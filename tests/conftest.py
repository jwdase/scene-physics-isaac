"""Test setup.

Pure tests (the default ``pytest`` run) never start Isaac Sim. Tests marked ``isaac`` need Kit, and Kit
must start before anything imports ``pxr``: the venv ships a standalone OpenUSD and Kit loads its own,
and importing both in one process crashes. So when the ``isaac`` marker is selected
(``uv run pytest -m isaac``), Kit is booted in ``pytest_configure``, before test modules are imported.
"""

import os
import sys

import pytest


def _wants_isaac(config) -> bool:
    expr = (config.option.markexpr or "").replace("not isaac", "")
    return "isaac" in expr or "multiworld" in expr


def pytest_configure(config):
    if _wants_isaac(config):
        from scene_physics.isaac import app

        app.launch(headless=True)


@pytest.fixture(scope="session")
def isaac():
    from scene_physics.isaac import app

    if not app.is_running():
        pytest.skip("Isaac Sim tests must be selected with -m isaac (Kit has to start before pxr is imported)")
    return app.get()


_EXIT_STATUS = 0


def pytest_sessionfinish(session, exitstatus):
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)


def pytest_unconfigure(config):
    # Runs after the terminal summary. Kit can abort in SimulationApp.close(); if it was started,
    # flush and hard-exit with pytest's status instead.
    from scene_physics.isaac import app

    if app.is_running():
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_EXIT_STATUS)
