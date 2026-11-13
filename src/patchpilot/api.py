"""Minimal FastAPI app (health only) with fail-closed path and payload boundaries."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from patchpilot import __version__
from patchpilot.sandbox.tar_and_diff import (
    MAX_PATCH_BYTES,
    TarSafetyError,
    validate_safe_relative_path,
)

MAX_PAYLOAD_BYTES = MAX_PATCH_BYTES  # 10 MB cap per SPEC §6


class WorkspacePathError(ValueError):
    """Raised when a path escapes or attempts traversal outside a configured workspace root."""


class _PayloadTooLargeError(Exception):
    """Internal signal raised when streamed HTTP request bytes exceed `MAX_PAYLOAD_BYTES`."""


def resolve_safe_workspace_path(
    raw_path: str | Path,
    workspace_root: Path,
    *,
    allow_root: bool = False,
) -> Path:
    """Resolve `raw_path` strictly inside `workspace_root`, rejecting traversal and symlinks."""
    if workspace_root.is_symlink():
        raise WorkspacePathError("Configured workspace root must not be a symlink")
    root_resolved = workspace_root.resolve()
    if not root_resolved.is_dir():
        raise WorkspacePathError(f"Workspace root directory does not exist: {workspace_root}")

    raw_str = str(raw_path)
    if allow_root and raw_str.replace("\\", "/") in (".", "./"):
        return root_resolved

    try:
        return validate_safe_relative_path(raw_str, root_resolved)
    except TarSafetyError as exc:
        raise WorkspacePathError(str(exc)) from exc


async def _send_json_error(
    scope: Scope,
    receive: Receive,
    send: Send,
    status_code: int,
    detail: str,
) -> None:
    resp = JSONResponse(status_code=status_code, content={"detail": detail})
    await resp(scope, receive, send)


class PayloadLimitMiddleware:
    """ASGI middleware enforcing a hard `MAX_PAYLOAD_BYTES` cap on incoming HTTP payloads."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_PAYLOAD_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit_msg = f"Request payload exceeds {self.max_bytes} byte limit"
        for key_bytes, val_bytes in scope.get("headers", ()):
            if key_bytes.lower() == b"content-length":
                try:
                    content_length = int(val_bytes.decode("ascii").strip())
                except (UnicodeDecodeError, ValueError):
                    await _send_json_error(
                        scope, receive, send, 400, "Invalid Content-Length header"
                    )
                    return
                if content_length < 0:
                    await _send_json_error(
                        scope, receive, send, 400, "Negative Content-Length header is invalid"
                    )
                    return
                if content_length > self.max_bytes:
                    await _send_json_error(scope, receive, send, 413, limit_msg)
                    return

        received_bytes = 0
        response_started = False

        async def bounded_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if isinstance(body, (bytes, bytearray)):
                    received_bytes += len(body)
                    if received_bytes > self.max_bytes:
                        raise _PayloadTooLargeError
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, tracked_send)
        except _PayloadTooLargeError:
            if not response_started:
                await _send_json_error(scope, receive, send, 413, limit_msg)


app = FastAPI(title="PatchPilot", version=__version__)
app.add_middleware(PayloadLimitMiddleware, max_bytes=MAX_PAYLOAD_BYTES)


@app.exception_handler(WorkspacePathError)
async def _workspace_path_error_handler(
    _request: Request,
    exc: WorkspacePathError,
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
