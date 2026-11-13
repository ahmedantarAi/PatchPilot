"""Unit tests for Phase 9 security hardening: API boundaries, safe XML, and dep pinning."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import Request

from patchpilot.api import (
    MAX_PAYLOAD_BYTES,
    WorkspacePathError,
    app,
    resolve_safe_workspace_path,
)
from patchpilot.sandbox import (
    SandboxRunner,
    validate_requirements,
    validate_requirements_file,
)
from patchpilot.sandbox.tar_and_diff import (
    MAX_JUNIT_CASES,
    MAX_OUTPUT_BYTES,
)


def _load_container_runner() -> ModuleType:
    runner_path = Path(__file__).resolve().parent.parent / "sandbox" / "runner.py"
    spec = importlib.util.spec_from_file_location("sandbox_runner_hardening", runner_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner_hardening"] = mod
    spec.loader.exec_module(mod)
    return mod


CONTAINER_RUNNER = _load_container_runner()


# ---------------------------------------------------------------------------
# 1. FastAPI workspace path containment & bounded payload size checks
# ---------------------------------------------------------------------------


def test_api_workspace_path_containment_rejects_traversal_and_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    target_file = workspace / "src" / "app.py"
    target_file.write_text("x = 1\n", encoding="utf-8")

    resolved = resolve_safe_workspace_path("src/app.py", workspace)
    assert resolved == workspace.resolve() / "src" / "app.py"
    assert resolve_safe_workspace_path(".", workspace, allow_root=True) == workspace.resolve()

    for unsafe in (
        "",
        "null\x00byte.py",
        ".",
        "../outside.py",
        "src/../../outside.py",
        "src/../src/app.py",
        "../outside\\file.py",
        "/etc/passwd",
        "\\Windows\\win.ini",
        "C:/evil.py",
    ):
        with pytest.raises(WorkspacePathError):
            resolve_safe_workspace_path(unsafe, workspace)

    # Real symlink escape if OS user has symlink privileges, otherwise simulated symlink check.
    escape_link = workspace / "src" / "link_escape"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    try:
        escape_link.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        orig_is_symlink = Path.is_symlink

        def _fake_is_symlink(self: Path) -> bool:
            if self.name == "link_escape":
                return True
            return orig_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", _fake_is_symlink)

    with pytest.raises(WorkspacePathError, match="symlink"):
        resolve_safe_workspace_path("src/link_escape/secret.txt", workspace)


def test_api_payload_size_middleware_enforces_10mb_limit() -> None:
    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            ok = await client.get("/healthz")
            assert ok.status_code == 200

            oversized = await client.post(
                "/healthz",
                headers={"Content-Length": str(MAX_PAYLOAD_BYTES + 1)},
            )
            assert oversized.status_code == 413

            bad_cl = await client.post("/healthz", headers={"Content-Length": "not-an-int"})
            assert bad_cl.status_code == 400

            neg_cl = await client.post("/healthz", headers={"Content-Length": "-5"})
            assert neg_cl.status_code == 400

        # Test streamed chunk enforcement when no Content-Length header is supplied.
        @app.post("/__test_echo_size__")
        async def _echo_size(request: Request) -> dict[str, int]:
            data = await request.body()
            return {"size": len(data)}

        try:

            async def _stream_too_large() -> object:
                chunk = b"x" * (1024 * 1024)
                for _ in range(11):
                    yield chunk

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                streamed = await client.post(
                    "/__test_echo_size__",
                    content=_stream_too_large(),  # type: ignore[arg-type]
                )
                assert streamed.status_code == 413
        finally:
            app.router.routes[:] = [
                r for r in app.router.routes if getattr(r, "path", "") != "/__test_echo_size__"
            ]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. Safe JUnit XML parsing (Expat non-DTD, zero ElementTree)
# ---------------------------------------------------------------------------


def test_container_runner_does_not_import_elementtree() -> None:
    runner_src = (Path(__file__).resolve().parent.parent / "sandbox" / "runner.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(runner_src)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert "xml.etree.ElementTree" not in imported_modules
    assert "xml.etree.cElementTree" not in imported_modules
    assert "xml.parsers.expat" in imported_modules


def test_safe_expat_junit_parser_handles_details_and_caps() -> None:
    cases_xml = [f'<testcase classname="mod" name="test_{i}" time="0.01"/>' for i in range(505)]
    cases_xml[0] = (
        '<testcase classname="mod" name="test_fail_text" time="invalid">'
        "<failure>AssertionError: 1 != 2</failure></testcase>"
    )
    cases_xml[1] = (
        '<testcase classname="mod" name="test_err_attr" time="0.2">'
        '<error message="RuntimeError: boom">ignored body</error></testcase>'
    )
    cases_xml[2] = (
        '<testcase classname="mod" name="test_skip" time="0.0">'
        '<skipped message="not run"/></testcase>'
    )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite>'
        + "".join(cases_xml)
        + "</testsuite></testsuites>"
    )

    summary = CONTAINER_RUNNER.parse_junit_xml(xml)
    assert summary["tests"] == 505
    assert summary["failures"] == 1
    assert summary["errors"] == 1
    assert summary["skipped"] == 1
    assert summary["passed"] == 502
    assert len(summary["cases"]) == MAX_JUNIT_CASES == 500
    assert summary["cases"][0]["status"] == "failed"
    assert summary["cases"][0]["message"] == "AssertionError: 1 != 2"
    assert summary["cases"][0]["time"] == 0.0
    assert summary["cases"][1]["status"] == "error"
    assert summary["cases"][1]["message"] == "RuntimeError: boom"
    assert summary["cases"][2]["status"] == "skipped"


def test_safe_expat_junit_parser_rejects_oversized_and_malformed_xml() -> None:
    oversized = "<testsuites>" + ("a" * (MAX_OUTPUT_BYTES + 10)) + "</testsuites>"
    res_big = CONTAINER_RUNNER.parse_junit_xml(oversized)
    assert "exceeds" in str(res_big.get("error", ""))
    assert res_big["tests"] == 0

    malformed = "<testsuites><testcase></testsuites>"
    res_bad = CONTAINER_RUNNER.parse_junit_xml(malformed)
    assert str(res_bad.get("error", "")).startswith("XMLParseError:")
    assert res_bad["tests"] == 0

    external_xxe = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE foo SYSTEM "http://127.0.0.1:9999/evil.dtd">\n'
        "<testsuites/>"
    )
    res_xxe = CONTAINER_RUNNER.parse_junit_xml(external_xxe)
    assert "Hostile XML" in str(res_xxe.get("error", ""))


# ---------------------------------------------------------------------------
# 3. Strict dependency pinning in both host and container installer
# ---------------------------------------------------------------------------


def test_validate_requirements_accepts_exact_pins_and_sha256_hashes(tmp_path: Path) -> None:
    digest = "a" * 64
    req_txt = tmp_path / "requirements.txt"
    req_txt.write_text(
        (
            "# pinned requirements\n"
            "\n"
            "pytest==9.2.0  # inline comment\n"
            "pydantic[email]==2.14.2 \\\n"
            f"    --hash=sha256:{digest}\n"
        ),
        encoding="utf-8",
    )
    validated = validate_requirements_file(req_txt)
    assert validated == [
        "pytest==9.2.0",
        f"pydantic[email]==2.14.2 --hash=sha256:{digest}",
    ]
    assert validate_requirements([f"httpx==0.28.1 --hash=sha256:{digest}"]) == [
        f"httpx==0.28.1 --hash=sha256:{digest}"
    ]


@pytest.mark.parametrize(
    "bad_req",
    [
        "requests",
        "requests>=2.31.0",
        "requests==2.*",
        "requests==*",
        "-e .",
        "--editable ../other",
        "http://evil.example/pkg.whl",
        "https://evil.example/pkg.whl",
        "git+https://github.com/org/repo.git",
        "file:///tmp/local_pkg",
        "../local_pkg",
        "pkg @ https://evil.example/pkg.whl",
        "pkg==1.0.0 --hash=md5:0123456789abcdef0123456789abcdef",
        "pkg==1.0.0 --extra-index-url https://evil.example/simple",
        f"--hash=sha256:{'a' * 64}",
        "pkg==1.0.0; os_name == 'posix'",
    ],
)
def test_validate_requirements_rejects_unpinned_urls_and_paths(bad_req: str) -> None:
    with pytest.raises(ValueError):
        validate_requirements([bad_req])


def test_install_dependencies_rejects_invalid_requirements_file_before_docker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from patchpilot.sandbox import runner as runner_mod

    def _fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("subprocess.Popen should not run on invalid requirements")

    monkeypatch.setattr(runner_mod.subprocess, "Popen", _fail_if_called)
    runner = SandboxRunner(lock_path=tmp_path / "lock", cleanup_on_start=False)

    bad_req_txt = tmp_path / "requirements.txt"
    bad_req_txt.write_text("git+https://github.com/org/repo.git\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.install_dependencies_sync(requirements_file=bad_req_txt)


def test_container_install_deps_rejects_unpinned_and_url_inputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("pip subprocess must not run when requirements are invalid")

    monkeypatch.setattr(CONTAINER_RUNNER.subprocess, "run", _fail_if_called)
    for bad_list in (["requests"], ["-e ."], ["git+https://github.com/org/repo.git"]):
        payload = json.dumps({"requirements": bad_list}).encode("utf-8")
        monkeypatch.setattr(
            CONTAINER_RUNNER.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload))
        )
        rc = CONTAINER_RUNNER.run_install_deps()
        assert rc == 2
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["exit_code"] == 2
        assert "Invalid requirement specification" in parsed["stderr"]
