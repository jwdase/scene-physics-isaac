"""Image outputs: segmentation PNGs, overlays, the target silhouette, GIFs and labelled panels.

``segmentation.png`` keeps the old convention (single channel, pixel value = object id, 0 = background)
and additionally carries its own ``{name: id}`` map in a PNG text chunk named ``labels``, so the image
is self-describing (:func:`read_segmentation`). The same ids are stored on each object in
``scene.usd`` (``sp:segmentationId``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


def linear_to_srgb(c) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def color_u8(linear_rgb) -> tuple[int, int, int]:
    return tuple(int(round(v * 255)) for v in linear_to_srgb(linear_rgb))


# --------------------------------------------------------------------------------- segmentation

def write_segmentation(path: str | Path, seg: np.ndarray, labels: dict[str, int]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(seg)
    arr = arr.astype(np.uint8) if arr.max(initial=0) < 256 else arr.astype(np.uint16)
    info = PngImagePlugin.PngInfo()
    info.add_text("labels", json.dumps({"background": 0, **{k: int(v) for k, v in labels.items() if k != "background"}}))
    Image.fromarray(arr).save(path, pnginfo=info)
    return path


def read_segmentation(path: str | Path) -> tuple[np.ndarray, dict[str, int]]:
    img = Image.open(path)
    labels = json.loads(img.text.get("labels", "{}")) if hasattr(img, "text") else {}
    return np.asarray(img), labels


def resize_nearest(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an id/mask image to ``(height, width)``."""
    if img.shape[:2] == tuple(shape):
        return img
    h, w = shape
    ys = (np.arange(h) + 0.5) * img.shape[0] / h
    xs = (np.arange(w) + 0.5) * img.shape[1] / w
    return img[ys.astype(int)[:, None], xs.astype(int)[None, :]]


def overlay(rgb: np.ndarray, seg: np.ndarray, id_colors: dict[int, tuple], *, alpha: float = 0.5,
            outlines: list[tuple[np.ndarray, tuple]] | None = None) -> np.ndarray:
    """Blend object colours over the image; draw ``outlines`` (``(mask, rgb_u8)``) on top."""
    out = rgb.astype(np.float32).copy()
    seg = resize_nearest(seg, rgb.shape[:2])
    for oid, col in id_colors.items():
        m = seg == oid
        if m.any():
            out[m] = (1 - alpha) * out[m] + alpha * np.asarray(col, dtype=np.float32)
    for mask, col in outlines or []:
        mask = resize_nearest(mask.astype(np.uint8), rgb.shape[:2]).astype(bool)
        edge = mask & ~_erode(mask)
        edge = _dilate(edge)
        out[edge] = col
    return np.clip(out, 0, 255).astype(np.uint8)


def _erode(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1, constant_values=False)
    return p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def _dilate(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1, constant_values=False)
    return p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def target_image(mask: np.ndarray, color_linear, shape: tuple[int, int] | None = None) -> np.ndarray:
    """RGBA image: the target's full (amodal) silhouette filled with its colour, transparent elsewhere.

    Same content as the old ``results/target.png``.
    """
    if shape is not None:
        mask = resize_nearest(mask.astype(np.uint8), shape).astype(bool)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color_u8(color_linear), 255)
    return rgba


def save_png(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(img)).save(path)
    return path


def write_depth(path: str | Path, depth_m: np.ndarray) -> Path:
    """Depth to the image plane as a 16-bit PNG in millimetres (0 = nothing hit), the usual RGB-D format."""
    d = np.asarray(depth_m, dtype=np.float64)
    mm = np.where(np.isfinite(d) & (d > 0), np.round(d * 1000.0), 0).clip(0, 65535).astype(np.uint16)
    return save_png(path, mm)


def read_depth(path: str | Path) -> np.ndarray:
    """Metres (NaN where nothing was hit)."""
    mm = np.asarray(Image.open(path)).astype(np.float64)
    return np.where(mm > 0, mm / 1000.0, np.nan)


def save_gif(path: str | Path, frames: list[np.ndarray], fps: int = 30) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=255) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


def labelled_panel(images: list[np.ndarray], titles: list[str], *, cols: int = 2, tile_width: int = 640) -> np.ndarray:
    """Grid of images with a title bar on each tile."""
    tiles = []
    for img, title in zip(images, titles):
        im = Image.fromarray(np.asarray(img)[..., :3] if np.asarray(img).ndim == 3 else np.asarray(img)).convert("RGB")
        h = int(round(im.height * tile_width / im.width))
        im = im.resize((tile_width, h))
        canvas = Image.new("RGB", (tile_width, h + 22), (24, 24, 24))
        canvas.paste(im, (0, 22))
        ImageDraw.Draw(canvas).text((6, 4), title, fill=(255, 235, 120), font=ImageFont.load_default())
        tiles.append(canvas)
    rows = (len(tiles) + cols - 1) // cols
    th = max(t.height for t in tiles)
    grid = Image.new("RGB", (cols * tile_width, rows * th), (24, 24, 24))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * tile_width, (i // cols) * th))
    return np.asarray(grid)


def id_colors_for(scene) -> dict[int, tuple]:
    """Segmentation id -> object's categorical colour (sRGB uint8)."""
    return {int(o.segmentation_id): color_u8(o.color) for o in scene if o.segmentation_id}


def qa_palette(ids) -> dict[int, tuple]:
    """Maximally spaced hues per id, like the old Blender ID pass (distinct from the object colours)."""
    import colorsys

    ids = sorted(int(i) for i in ids if int(i) > 0)
    n = max(len(ids), 1)
    return {i: tuple(int(255 * c) for c in colorsys.hsv_to_rgb(k / n, 0.9, 1.0)) for k, i in enumerate(ids)}
