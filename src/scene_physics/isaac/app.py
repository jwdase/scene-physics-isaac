"""Isaac Sim application lifecycle.

Kit has to be started before anything imports ``omni.*`` (and before Kit's own ``pxr`` is loaded), so
every entry point that needs the simulator calls :func:`launch` first and only then imports the rest
of :mod:`scene_physics.isaac`.

    from scene_physics.isaac import app
    app.launch(headless=True)
    ...
    app.shutdown()

Notes
-----
* Non-interactive runs need ``OMNI_KIT_ACCEPT_EULA=YES`` in the environment (Isaac Sim otherwise asks
  on stdin). It is not set here: accepting NVIDIA's EULA is the user's call.
* ``SimulationApp.close()`` in Isaac Sim 6.1 can abort with "Destroying busy TaskGroup!" after all work
  is done; :func:`shutdown` flushes output and hard-exits so the exit status reflects the real result.
"""

from __future__ import annotations

import os
import sys

_APP = None

QUIET_ARGS = [
    "--/log/level=warning",
    "--/log/outputStreamLevel=error",
    "--/log/fileLogLevel=warning",
    "--/app/extensions/fsWatcherEnabled=0",
]


def launch(
    headless: bool = True,
    *,
    renderer: str = "RaytracedLighting",
    experience: str | None = None,
    quiet: bool = True,
    width: int = 1280,
    height: int = 720,
    extra_args: list[str] | None = None,
    samples_per_pixel: int = 64,
):
    """Start Isaac Sim once per process and return the ``SimulationApp``.

    ``renderer``: ``RaytracedLighting`` (fast), ``PathTracing`` (stimulus quality) or
    ``RealTimePathTracing``. ``experience``: a ``.kit`` file name, e.g. ``isaacsim.exp.full.kit`` for
    the full editor UI.
    """
    global _APP
    if _APP is not None:
        return _APP
    if not os.environ.get("OMNI_KIT_ACCEPT_EULA") and not sys.stdin.isatty():
        print(
            "[scene_physics] Isaac Sim will ask you to accept the NVIDIA Omniverse EULA on stdin. For\n"
            "                non-interactive runs, export OMNI_KIT_ACCEPT_EULA=YES once you have accepted it.",
            file=sys.stderr,
        )
    from isaacsim import SimulationApp

    args = list(extra_args or [])
    if quiet:
        args = QUIET_ARGS + args
    config = {
        "headless": headless,
        "renderer": renderer,
        "width": width,
        "height": height,
        "multi_gpu": False,
        "samples_per_pixel_per_frame": samples_per_pixel,
        "extra_args": args,
    }
    # Kit parses sys.argv as its own command line; hide ours (CLI / pytest flags) while it starts.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        if experience is not None:
            from pathlib import Path

            import isaacsim

            exp = Path(experience)
            if not exp.is_file():
                exp = Path(isaacsim.__file__).parent / "apps" / experience
            _APP = SimulationApp(config, experience=str(exp))
        else:
            _APP = SimulationApp(config)
    finally:
        sys.argv = argv
    return _APP


def get():
    if _APP is None:
        raise RuntimeError("Isaac Sim is not running; call scene_physics.isaac.app.launch() first")
    return _APP


def is_running() -> bool:
    return _APP is not None


def update(n: int = 1) -> None:
    app = get()
    for _ in range(n):
        app.update()


def shutdown(code: int = 0) -> None:
    """Flush and hard-exit (see module notes)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
