"""Object appearance: categorical colours and simple PBR hints per library asset.

Ported from the Blender pipeline's ``material_specs.py`` so objects keep the colours they had in the
old stimuli (every dataset object gets a well-separated hue; the table stays wood). Assets without a
hand-picked spec get a deterministic colour from the same categorical palette, keyed by asset name, so
a given asset looks the same in every scene.
"""

from __future__ import annotations

import hashlib

# Hand-tuned specs from Scene-Physics/src/scene_physics/visualization/blender/material_specs.py.
# base_color is linear RGB; roughness/metallic/transmission follow UsdPreviewSurface semantics.
MATERIAL_SPECS: dict[str, dict] = {
    "dining_room_table": {"base_color": (0.52, 0.33, 0.19), "roughness": 0.55, "texture": "wood"},
    "square_wood_block": {"base_color": (0.90, 0.10, 0.10), "roughness": 0.45},
    "star_wood_block": {"base_color": (0.40, 0.55, 0.08), "roughness": 0.45},
    "bung": {"base_color": (0.45, 0.26, 0.10), "roughness": 0.80},
    "jug04": {"base_color": (0.10, 0.70, 0.95), "roughness": 0.30},
    "vase_05": {"base_color": (0.62, 0.12, 0.85), "roughness": 0.28},
    "coffeemug": {"base_color": (0.10, 0.70, 0.20), "roughness": 0.30},
    "int_kitchen_accessories_le_creuset_bowl_30cm": {"base_color": (0.88, 0.12, 0.78), "roughness": 0.28},
    "glass1": {"base_color": (0.12, 0.30, 0.90), "roughness": 0.16, "opacity": 0.35},
    "b04_candle_holder_metal": {"base_color": (0.80, 0.62, 0.08), "roughness": 0.42, "metallic": 1.0},
    "b05_coffee_grinder": {"base_color": (0.35, 0.15, 0.82), "roughness": 0.48, "metallic": 0.6},
    "f10_apple_iphone_4": {"base_color": (0.30, 0.45, 0.65), "roughness": 0.22},
    "round_coaster_stone": {"base_color": (0.10, 0.42, 0.30), "roughness": 0.70},
    "b03_loafbread": {"base_color": (0.95, 0.50, 0.05), "roughness": 0.85},
    "banana_fix2": {"base_color": (0.95, 0.85, 0.10), "roughness": 0.45},
    "pepper": {"base_color": (0.65, 0.05, 0.20), "roughness": 0.28},
    "shark": {"base_color": (0.05, 0.65, 0.60), "roughness": 0.55},
    "bee": {"base_color": (0.55, 0.80, 0.10), "roughness": 0.45},
    "heart": {"base_color": (0.95, 0.35, 0.60), "roughness": 0.40},
}

# The 18-colour categorical palette the specs above draw from (linear RGB).
PALETTE = [
    (0.90, 0.10, 0.10), (0.95, 0.50, 0.05), (0.95, 0.85, 0.10), (0.55, 0.80, 0.10),
    (0.10, 0.70, 0.20), (0.05, 0.65, 0.60), (0.10, 0.70, 0.95), (0.12, 0.30, 0.90),
    (0.35, 0.15, 0.82), (0.62, 0.12, 0.85), (0.88, 0.12, 0.78), (0.95, 0.35, 0.60),
    (0.65, 0.05, 0.20), (0.45, 0.26, 0.10), (0.80, 0.62, 0.08), (0.30, 0.45, 0.65),
    (0.40, 0.55, 0.08), (0.10, 0.42, 0.30),
]

DEFAULT_ROUGHNESS = 0.45


def _lookup(asset: str) -> dict | None:
    if asset in MATERIAL_SPECS:
        return MATERIAL_SPECS[asset]
    # longest spec key the name extends at an underscore boundary (e.g. f10_apple_iphone_4_006)
    keys = [k for k in MATERIAL_SPECS if asset.startswith(k + "_")]
    return MATERIAL_SPECS[max(keys, key=len)] if keys else None


def look_for(asset: str) -> dict:
    """Render hints for an asset: ``base_color``, ``roughness`` and optional ``metallic``/``opacity``."""
    spec = _lookup(asset)
    if spec is not None:
        out = {"roughness": DEFAULT_ROUGHNESS, **spec}
    else:
        digest = int(hashlib.sha1(asset.encode()).hexdigest(), 16)
        out = {"base_color": PALETTE[digest % len(PALETTE)], "roughness": DEFAULT_ROUGHNESS}
    out["base_color"] = tuple(float(c) for c in out["base_color"])
    return out


def color_for(asset: str) -> tuple[float, float, float]:
    return look_for(asset)["base_color"]
