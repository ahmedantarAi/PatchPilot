"""Minimal FastAPI app (health only). Feature endpoints are out of scope for the scaffold."""

from __future__ import annotations

from fastapi import FastAPI

from patchpilot import __version__

app = FastAPI(title="PatchPilot", version=__version__)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
