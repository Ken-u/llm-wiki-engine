"""Runtime Web UI mounting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def resource_path(*parts: str) -> Path:
    if getattr(sys, "_MEIPASS", None):
        return Path(sys._MEIPASS).joinpath("app", "runtime", *parts)
    return Path(__file__).resolve().parent.joinpath(*parts)


def mount_runtime_ui(app: FastAPI) -> None:
    ui_dir = resource_path("ui_dist")
    if ui_dir.exists():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="runtime-ui")
