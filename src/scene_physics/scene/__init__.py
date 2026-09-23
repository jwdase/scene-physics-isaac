"""The scene data model and the ``scene.usd`` file format.

    from scene_physics.scene import load_scene
    scene = load_scene("data/data-set/scene017")      # or a path to scene.usd
    print(scene.summary())
    scene.target.pose, scene["jug04"].role, scene.camera.fov_degree

Everything here is plain Python + OpenUSD; Isaac Sim is not needed to read or write scenes.
"""

from scene_physics.scene.model import (
    Mesh,
    Prior,
    Scene,
    SceneObject,
    Visibility,
    base_asset_name,
    safe_prim_name,
)


def load_scene(spec) -> Scene:
    """Load a scene from a path (file or scene directory) or ``<experiment>/<scene>`` under ``data/``."""
    from scene_physics import paths
    from scene_physics.scene.usd_io import read_scene

    return read_scene(paths.resolve_scene(spec))


def save_scene(scene: Scene, path, physics=None):
    from scene_physics.scene.usd_io import write_scene

    return write_scene(scene, path, physics)


__all__ = [
    "Mesh",
    "Prior",
    "Scene",
    "SceneObject",
    "Visibility",
    "base_asset_name",
    "safe_prim_name",
    "load_scene",
    "save_scene",
]
