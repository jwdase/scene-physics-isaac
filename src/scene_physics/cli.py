"""``sp``: the command line for authoring scenes and running the model.

    uv run sp import stage/scene053.usdc --exp data-set      # Blender export -> authoring scene
    uv run sp check  data-set/scene053                        # physics + occlusion report (+ GIF)
    uv run sp check  data-set/scene053 --gui                  # ... and watch it in Isaac Sim
    uv run sp open   data-set/scene053                        # edit in the Isaac Sim editor
    uv run sp finalize data-set/scene053                      # -> data/data-set/scene053/
    uv run sp info   data-set/scene053                        # readable summary of a scene
    uv run sp infer  data-set/scene053 --num-worlds 64        # the model (remote server)

    uv run sp new data-set/scene060                           # author without Blender: table + camera,
    uv run sp add data-set/scene060 jug04 --at 0.1 -0.2       # then objects from the library
    uv run sp library list

Scenes can be given as a path (``scene.usd`` or its directory) or as ``<experiment>/<scene>``, looked
up under ``data/`` and then ``authoring/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _config(args, scene_path: Path | None = None):
    from scene_physics.config import load_config, load_experiment

    exp = getattr(args, "exp", None)
    extra = [args.config] if getattr(args, "config", None) else []
    if exp is None and scene_path is not None:
        # experiment of an existing scene: data/<exp>/<scene>/scene.usd or authoring/<exp>/<scene>.usd
        from scene_physics import paths

        for base in (paths.data_dir(), paths.authoring_dir()):
            try:
                rel = scene_path.resolve().relative_to(base.resolve())
                cand = rel.parts[0]
                if (paths.experiments_dir() / f"{cand}.toml").exists():
                    exp = cand
                    break
            except ValueError:
                continue
    if exp:
        return load_experiment(exp.replace("/", "-"), *extra)
    return load_config(*extra)


def _tags(pairs: list[str] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--tag expects OBJECT=TAG, got {pair!r}")
        obj, tag = pair.split("=", 1)
        out.setdefault(obj, []).append(tag)
    return out


# ------------------------------------------------------------------------------------------ commands


def cmd_import(args):
    from scene_physics.authoring import workflow

    cfg = _config(args)
    workflow.import_scene(args.source, cfg, name=args.name, out=args.out, static=args.static,
                          tags=_tags(args.tag), overwrite=args.overwrite)


def cmd_check(args):
    from scene_physics import paths
    from scene_physics.authoring import workflow

    path = paths.resolve_scene(args.scene, prefer="authoring")
    workflow.check_scene(path, _config(args, path), gui=args.gui, gif=not args.no_gif)


def cmd_finalize(args):
    from scene_physics import paths
    from scene_physics.authoring import workflow

    path = paths.resolve_scene(args.scene, prefer="authoring")
    cfg = _config(args, path)
    workflow.finalize_scene(path, cfg, name=args.name, experiment=args.exp, settle=not args.no_settle,
                            target=args.target, render=not args.no_render, regenerate_priors=not args.keep_priors,
                            strict=args.strict, overwrite=args.overwrite)


def cmd_render(args):
    from scene_physics import paths
    from scene_physics.authoring import workflow

    path = paths.resolve_scene(args.scene)
    workflow.render_scene(path, _config(args, path), out_dir=args.out, quality=args.quality)


def cmd_open(args):
    from scene_physics import paths
    from scene_physics.authoring import workflow

    path = paths.resolve_scene(args.scene, prefer="authoring")
    workflow.open_in_editor(path, _config(args, path))


def cmd_migrate(args):
    from scene_physics.authoring import workflow

    cfg = _config(args)
    failed = []
    for d in args.dirs:
        try:
            workflow.migrate_legacy(d, cfg, render=not args.no_render, overwrite=args.overwrite)
        except Exception as exc:  # keep going through a dataset; report at the end
            failed.append((d, exc))
            print(f"[sp] FAILED {d}: {exc}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(args.dirs)} scenes failed")


def cmd_info(args):
    from scene_physics.scene import load_scene

    scene = load_scene(args.scene)
    print(scene.summary())
    if args.priors:
        for o in scene.dynamic_objects:
            if o.prior is not None:
                print(f"  prior {o.name}: rule={o.prior.rule} std=({o.prior.pos_std}, {o.prior.rot_std})")
                for m, w in zip(o.prior.means, o.prior.weights):
                    print(f"      w={w:.3f}  xyz=({m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f})")
    if args.json:
        import json

        out = {
            "name": scene.name,
            "camera": scene.camera.to_dict(),
            "metadata": {k: v for k, v in scene.metadata.items() if k != "physics_config"},
            "objects": {
                o.name: {
                    "role": o.role, "tags": o.tags, "dynamic": o.dynamic, "pose": o.pose.tolist(),
                    "segmentation_id": o.segmentation_id,
                    "visible_fraction": None if o.visibility is None else o.visibility.fraction,
                    "prior": None if o.prior is None else o.prior.to_legacy_dict(),
                }
                for o in scene
            },
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"[sp] wrote {args.json}")


def cmd_validate(args):
    from scene_physics.authoring import analysis
    from scene_physics.scene import load_scene

    bad = 0
    for spec in args.scenes:
        scene = load_scene(spec)
        problems = analysis.validate(scene)
        status = "OK" if not problems else "INVALID: " + "; ".join(problems)
        print(f"{scene.name}: {status}")
        bad += bool(problems)
    if bad:
        raise SystemExit(1)


def cmd_list(args):
    from scene_physics import paths
    from scene_physics.scene import load_scene

    roots = [paths.data_dir(), paths.authoring_dir()]
    for root in roots:
        base = root / args.exp if args.exp else root
        files = sorted(base.rglob("scene.usd")) + sorted(p for p in base.rglob("*.usd") if p.name != "scene.usd")
        if not files:
            continue
        print(f"{root}:")
        for f in files:
            try:
                s = load_scene(f)
                hidden = ",".join(o.name for o in s.with_role("hidden")) or "-"
                print(f"  {str(f.relative_to(root)):50s} {s.metadata.get('status', '?'):10s} hidden={hidden}")
            except Exception as exc:
                print(f"  {str(f.relative_to(root)):50s} unreadable: {exc}")


def cmd_priors(args):
    from pxr import Usd

    from scene_physics import paths
    from scene_physics.scene import load_scene, priors, usd_io

    path = paths.resolve_scene(args.scene)
    scene = load_scene(path)
    cfg = _config(args, path).authoring
    for key in ("rule", "z", "orientation"):
        v = getattr(args, key)
        if v is not None:
            setattr(cfg, {"rule": "prior_rule", "z": "prior_z", "orientation": "prior_orientation"}[key], v)
    if args.pos_std is not None:
        cfg.prior_pos_std = args.pos_std
    if args.rot_std is not None:
        cfg.prior_rot_std = args.rot_std
    priors.assign_priors(scene, cfg)
    stage = Usd.Stage.Open(str(path))
    usd_io.update_scene_file(stage, scene, poses=False)
    stage.GetRootLayer().Save()
    print(scene.summary())


def cmd_library(args):
    from scene_physics import paths
    from scene_physics.scene import library, load_scene

    if args.action == "list":
        names = library.list_assets()
        if not names:
            print("[sp] the library is empty; run `sp library extract` (after finalizing or migrating scenes)")
        for n in names:
            mesh, info = library.read_asset(n)
            e = mesh.extents * 100
            print(f"  {n:48s} {'dynamic' if info['dynamic'] else 'static '}  {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} cm")
        return
    specs = args.scenes or [str(p) for p in sorted(paths.data_dir().rglob("scene.usd"))]
    if not specs:
        raise SystemExit("no scenes found under data/; pass scene paths explicitly")
    scenes = [load_scene(s) for s in specs]
    library.extract_from_scenes(scenes, overwrite=args.overwrite, log=lambda m: print(f"[sp] {m}"))


def cmd_new(args):
    from scene_physics import paths
    from scene_physics.scene import library

    if "/" not in args.scene:
        raise SystemExit("name the scene as <experiment>/<name>, e.g. data-set/scene060")
    exp, name = args.scene.split("/", 1)
    cfg = _config(argparse.Namespace(exp=args.exp or exp, config=args.config))
    from scene_physics.camera import DEFAULT_CAMERA

    out = paths.authoring_dir() / exp / f"{name}.usd"
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists (use --overwrite)")
    library.new_scene(out, name, cfg.camera or DEFAULT_CAMERA, table=args.table, experiment=exp, physics=cfg.physics)
    print(f"[sp] new scene {out} (table {args.table}); add objects with `sp add {args.scene} <asset> --at X Y`")


def cmd_add(args):
    import numpy as np

    from scene_physics import paths
    from scene_physics.scene import library

    path = paths.resolve_scene(args.scene, prefer="authoring")
    cfg = _config(args, path)
    name = library.add_object(path, args.asset, at=tuple(args.at) if args.at else None, yaw_deg=args.yaw,
                              name=args.name, tags=args.tag, rng=np.random.default_rng(args.seed),
                              physics=cfg.physics)
    print(f"[sp] added {name} to {path}")


def cmd_infer(args):
    from scene_physics.inference import runner

    runner.main(args)


def cmd_lossmap(args):
    from scene_physics.inference import lossmap

    lossmap.main(args)


# ------------------------------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, exp_help="experiment config (experiments/<exp>.toml)"):
        sp.add_argument("--exp", help=exp_help)
        sp.add_argument("--config", help="extra TOML config applied on top")

    s = sub.add_parser("import", help="Blender USD export or legacy scene folder -> authoring scene")
    s.add_argument("source")
    s.add_argument("--name")
    s.add_argument("--out", help="output path (default authoring/<exp>/<name>.usd)")
    s.add_argument("--static", nargs="*", help="extra static (fixed) objects; the table is static automatically")
    s.add_argument("--tag", nargs="*", help="annotations OBJECT=TAG, e.g. square_wood_block_013=target")
    s.add_argument("--overwrite", action="store_true")
    common(s)
    s.set_defaults(func=cmd_import, needs_kit=False)

    s = sub.add_parser("check", help="simulate under the current physics + render + occlusion report")
    s.add_argument("scene")
    s.add_argument("--gui", action="store_true", help="watch it in an Isaac Sim window")
    s.add_argument("--no-gif", action="store_true")
    common(s)
    s.set_defaults(func=cmd_check, needs_kit=True)

    s = sub.add_parser("finalize", help="settle, assign roles/priors, render, write data/<exp>/<name>/")
    s.add_argument("scene")
    s.add_argument("--name")
    s.add_argument("--target", help="object that is the hidden target (default: automatic)")
    s.add_argument("--no-settle", action="store_true", help="keep the authored poses as ground truth")
    s.add_argument("--no-render", action="store_true")
    s.add_argument("--keep-priors", action="store_true", help="keep priors already in the scene")
    s.add_argument("--strict", action="store_true", help="fail if anything moves while settling or the scene is invalid")
    s.add_argument("--overwrite", action="store_true")
    common(s)
    s.set_defaults(func=cmd_finalize, needs_kit=True)

    s = sub.add_parser("render", help="render stimulus + segmentation for a scene")
    s.add_argument("scene")
    s.add_argument("--quality", choices=["final", "preview", "fast"], default="final")
    s.add_argument("--out")
    common(s)
    s.set_defaults(func=cmd_render, needs_kit=True)

    s = sub.add_parser("open", help="edit a scene in the Isaac Sim editor")
    s.add_argument("scene")
    common(s)
    s.set_defaults(func=cmd_open, needs_kit=True)

    s = sub.add_parser("migrate", help="convert legacy data-pipeline scene folders (poses kept)")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--no-render", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    common(s)
    s.set_defaults(func=cmd_migrate, needs_kit=True)

    s = sub.add_parser("info", help="print a scene summary")
    s.add_argument("scene")
    s.add_argument("--priors", action="store_true", help="also list prior candidates")
    s.add_argument("--json", help="also write a JSON summary (for people who do not read USD)")
    s.set_defaults(func=cmd_info, needs_kit=False)

    s = sub.add_parser("validate", help="check scenes are ready for inference")
    s.add_argument("scenes", nargs="+")
    s.set_defaults(func=cmd_validate, needs_kit=False)

    s = sub.add_parser("list", help="list scenes under data/ and authoring/")
    s.add_argument("exp", nargs="?")
    s.set_defaults(func=cmd_list, needs_kit=False)

    s = sub.add_parser("priors", help="regenerate the priors stored in a scene")
    s.add_argument("scene")
    s.add_argument("--rule", choices=["observed_objects", "occluder", "truth"])
    s.add_argument("--z", choices=["truth", "table_rest"])
    s.add_argument("--orientation", choices=["truth", "identity"])
    s.add_argument("--pos-std", type=float)
    s.add_argument("--rot-std", type=float)
    common(s)
    s.set_defaults(func=cmd_priors, needs_kit=False)

    s = sub.add_parser("library", help="object library: `extract` assets from scenes, or `list` them")
    s.add_argument("action", choices=["extract", "list"])
    s.add_argument("scenes", nargs="*", help="scenes to extract from (default: every scene under data/)")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_library, needs_kit=False)

    s = sub.add_parser("new", help="new authoring scene with the table and the experiment camera")
    s.add_argument("scene", help="<experiment>/<name>")
    s.add_argument("--table", default="dining_room_table", help="library asset used as the table")
    s.add_argument("--overwrite", action="store_true")
    common(s)
    s.set_defaults(func=cmd_new, needs_kit=False)

    s = sub.add_parser("add", help="add a library object to an authoring scene, standing on the table")
    s.add_argument("scene")
    s.add_argument("asset")
    s.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"), help="table position (default: random)")
    s.add_argument("--yaw", type=float, help="rotation about z in degrees (default: random)")
    s.add_argument("--name")
    s.add_argument("--tag", nargs="*", help="annotations, e.g. target occluder")
    s.add_argument("--seed", type=int)
    common(s)
    s.set_defaults(func=cmd_add, needs_kit=False)

    s = sub.add_parser("infer", help="run the model on scenes (importance sampling over N worlds)")
    s.add_argument("scene", nargs="+", help="one or more scenes (one Isaac Sim session for all)")
    s.add_argument("--num-worlds", "-N", type=int, help="parallel worlds (default: config / $NUM_WORLDS)")
    s.add_argument("--iterations", "-n", type=int, help="Gibbs iterations (default: config / $NUM_EPOCHS)")
    s.add_argument("--seed", type=int)
    s.add_argument("--out", help="results directory (default results/<exp>/<scene>/<timestamp>)")
    s.add_argument("--tag", help="name for this run (appended to the results directory)")
    common(s)
    s.set_defaults(func=cmd_infer, needs_kit=True, rtx=False)  # PhysX + Warp only

    s = sub.add_parser("lossmap", help="brute-force likelihood channels over the hidden object's XY prior box")
    s.add_argument("scene")
    s.add_argument("--resolution", type=float, default=0.01, help="grid spacing in metres")
    s.add_argument("--num-worlds", "-N", type=int)
    s.add_argument("--out")
    common(s)
    s.set_defaults(func=cmd_lossmap, needs_kit=True, rtx=False)  # PhysX + Warp only
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    code = 0
    try:
        if args.needs_kit and args.command != "open":
            # Boot Isaac Sim before any command code imports warp / pxr: Kit ships its own builds of both
            # and must load them first. (`sp open` boots the full editor experience itself.)
            from scene_physics.isaac import app

            app.launch(headless=not getattr(args, "gui", False), rendering=getattr(args, "rtx", True))
        args.func(args)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f"[sp] error: {exc}", file=sys.stderr, flush=True)
        code = 2
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if not isinstance(exc.code, int) and exc.code:
            print(f"[sp] {exc.code}", file=sys.stderr, flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    from scene_physics.isaac import app

    if app.is_running():
        app.shutdown(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
