"""Scene analysis that needs no simulator: visibility, roles, target choice, overlaps, validation.

Visibility uses the Warp ray caster at the camera's model resolution, i.e. exactly the view the model
gets (no anti-aliasing, no lighting), so "hidden" means hidden to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scene_physics.config import AuthoringConfig
from scene_physics.render import raycast
from scene_physics.scene.model import Scene, Visibility


@dataclass
class VisibilityReport:
    fractions: dict[str, float]
    visible_pixels: dict[str, int]
    silhouette_pixels: dict[str, int]
    amodal: dict[str, np.ndarray]  # object -> full silhouette mask at model resolution
    ids: np.ndarray  # first-hit object index image (-1 background)
    names: list[str]


def measure_visibility(scene: Scene, device: str | None = None) -> VisibilityReport:
    vis = raycast.visibility(scene, device=device)
    names = [n for n in vis["__names__"] if n in scene.objects]
    return VisibilityReport(
        fractions={n: vis[n]["fraction"] for n in names},
        visible_pixels={n: vis[n]["visible"] for n in names},
        silhouette_pixels={n: vis[n]["silhouette"] for n in names},
        amodal={n: vis[n]["amodal_mask"] for n in names},
        ids=vis["__ids__"],
        names=vis["__names__"],
    )


@dataclass
class RoleDecision:
    warnings: list[str] = field(default_factory=list)
    target: str | None = None


def assign_roles(scene: Scene, report: VisibilityReport, cfg: AuthoringConfig, *, target: str | None = None,
                 keep_authored: bool = True) -> RoleDecision:
    """Store visibility on every object, then set ``observed`` / ``hidden`` from it.

    A role the author set explicitly (``role_source == "authored"`` or ``"legacy"``) is kept when
    ``keep_authored``; if it disagrees with the measurement a warning is returned.
    """
    out = RoleDecision()
    for obj in scene:
        if obj.name in report.fractions:
            obj.visibility = Visibility(
                fraction=float(report.fractions[obj.name]),
                visible_pixels=int(report.visible_pixels[obj.name]),
                silhouette_pixels=int(report.silhouette_pixels[obj.name]),
            )
        if not obj.dynamic:
            obj.role = "static"
            continue
        measured = "hidden" if obj.visibility is not None and obj.visibility.fraction < cfg.hidden_threshold else "observed"
        if obj.visibility is not None and obj.visibility.silhouette_pixels == 0:
            out.warnings.append(f"{obj.name} is outside the camera view")
        if keep_authored and obj.role in ("observed", "hidden") and obj.role_source in ("authored", "legacy"):
            if obj.role != measured:
                out.warnings.append(
                    f"{obj.name}: role {obj.role!r} was set by hand but it is {obj.visibility.fraction:.1%} visible "
                    f"(threshold {cfg.hidden_threshold:.0%}) -> measured {measured!r}; keeping {obj.role!r}"
                )
            continue
        obj.role, obj.role_source = measured, "measured"

    # the target: explicit argument > existing tag > the single hidden object > least visible dynamic object
    if target is not None:
        if target not in scene:
            raise KeyError(f"--target {target!r} is not an object in {scene.name}: {sorted(scene.objects)}")
        for o in scene:
            if "target" in o.tags and o.name != target:
                o.tags.remove("target")
        if "target" not in scene[target].tags:
            scene[target].tags.append("target")
    tagged = scene.with_tag("target")
    if tagged:
        out.target = tagged[0].name
    else:
        hidden = scene.with_role("hidden")
        if len(hidden) == 1:
            out.target = hidden[0].name
        elif scene.dynamic_objects:
            cands = [o for o in scene.dynamic_objects if o.visibility is not None]
            best = min(cands, key=lambda o: (o.visibility.fraction, -o.visibility.silhouette_pixels)) if cands else None
            if best is not None:
                out.target = best.name
                if len(hidden) > 1:
                    out.warnings.append(
                        f"{len(hidden)} objects are hidden ({', '.join(o.name for o in hidden)}); picked {best.name} "
                        f"as the target (least visible, largest silhouette). Use --target to choose."
                    )
        if out.target is not None:
            scene[out.target].tags.append("target")
    if out.target is not None:
        t = scene[out.target]
        if t.role != "hidden" and t.visibility is not None:
            out.warnings.append(
                f"target {t.name} is {t.visibility.fraction:.1%} visible, above the hidden threshold "
                f"({cfg.hidden_threshold:.0%}): it is not occluded enough to be a hidden-object trial"
            )
    if not scene.with_role("hidden"):
        out.warnings.append("no object is hidden from the camera")
    return out


@dataclass
class Overlap:
    a: str
    b: str
    volume: float  # cubic metres of the smaller object inside the other (sampled)
    fraction: float  # that volume as a fraction of the smaller object's volume


def overlaps(scene: Scene, *, samples: int = 1024, min_fraction: float = 0.01, min_volume: float = 1e-6,
             seed: int = 0, device: str | None = None) -> list[Overlap]:
    """Pairs of objects that interpenetrate at their current poses (dynamic-dynamic and dynamic-static).

    Points are spread through each dynamic object's interior and tested against the other objects
    with winding numbers; a pair is reported when at least ``min_fraction`` of either object's
    volume (and at least ``min_volume`` m^3) lies inside the other. PhysX pushes such pairs apart, so
    an authored overlap turns into motion at the first physics step.
    """
    import warp as wp

    from scene_physics import geometry
    from scene_physics.likelihood.physics_terms import _inside_mesh, _interior_points, _wp_mesh

    device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    rng = np.random.default_rng(seed)
    objs = list(scene)
    meshes = {o.name: _wp_mesh(o.mesh, device) for o in objs}
    inside: dict[tuple[str, str], tuple[float, float]] = {}  # (a, b) -> (fraction of a inside b, volume)
    for a in scene.dynamic_objects:
        pts, vol = _interior_points(a.mesh, samples, rng, device)
        world = geometry.transform_points(a.pose, pts)
        for b in objs:
            if b.name == a.name:
                continue
            local = geometry.transform_points(geometry.inverse(b.pose), world).astype(np.float32)
            frac = float(_inside_mesh(meshes[b.name], local, device).mean())
            if frac > 0:
                inside[(a.name, b.name)] = (frac, frac * vol)
    out, seen = [], set()
    for (a, b), (frac, vol) in inside.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        other = inside.get((b, a), (0.0, 0.0))
        frac, vol = max((frac, vol), other)
        if frac >= min_fraction and vol >= min_volume:
            out.append(Overlap(a, b, vol, frac))
    return sorted(out, key=lambda o: -o.volume)


def overlap_lines(found: list[Overlap]) -> list[str]:
    return [f"{o.a} and {o.b} interpenetrate: {o.volume * 1e6:.1f} cm^3 ({o.fraction:.0%} of one of them)"
            for o in found]


def visibility_table(scene: Scene) -> str:
    lines = [f"  {'object':44s} {'role':9s} {'visible':>8s} {'px visible/silhouette':>22s}  tags"]
    for o in sorted(scene, key=lambda o: (o.role != "hidden", o.name)):
        if o.visibility is None:
            continue
        v = o.visibility
        lines.append(f"  {o.name:44s} {str(o.role):9s} {v.fraction:8.1%} {v.visible_pixels:>10d}/{v.silhouette_pixels:<11d}  "
                     f"{','.join(o.tags) or '-'}")
    return "\n".join(lines)


def validate(scene: Scene) -> list[str]:
    """Problems that make a scene unusable for inference (empty list = OK)."""
    problems = []
    if not scene.dynamic_objects:
        problems.append("no dynamic objects")
    if not scene.static_objects:
        problems.append("no static support (table)")
    for o in scene:
        if len(o.mesh.triangles) == 0:
            problems.append(f"{o.name}: empty mesh")
        if o.dynamic and o.role not in ("observed", "hidden"):
            problems.append(f"{o.name}: dynamic object without an observed/hidden role")
        if o.dynamic and o.prior is None:
            problems.append(f"{o.name}: no prior")
        if o.segmentation_id is None:
            problems.append(f"{o.name}: no segmentation id")
    ids = [o.segmentation_id for o in scene if o.segmentation_id is not None]
    if len(ids) != len(set(ids)):
        problems.append("duplicate segmentation ids")
    if not scene.with_role("hidden"):
        problems.append("no hidden object to infer")
    return problems


def stray_bodies(stage) -> list[str]:
    """Rigid bodies / colliders that are not under /World/Objects (the model only sees objects there).

    Happens when an asset is dragged into the Isaac Sim editor at the top level instead of into
    /World/Objects.
    """
    from pxr import UsdPhysics

    from scene_physics.scene import schema as S

    out = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.startswith(S.OBJECTS + "/") or path.startswith(S.GROUND) or path.startswith("/World/Look"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            out.append(path)
    return out
