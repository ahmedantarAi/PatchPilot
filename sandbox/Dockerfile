# Base image for running UNTRUSTED patches/tests. Launched by the host orchestrator with docker run; hardening flags are in SPEC.md section 6. Contains toolchains only; no secrets, no repo code.
FROM python:3.11.18-slim-bookworm
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin runner \
    && pip install --no-cache-dir pytest==9.2.0 \
    && mkdir -p /work /deps /usr/local/share/patchpilot-wheels \
    && python -c "import hashlib, base64, io, tarfile, zipfile, pathlib; \
wdir = pathlib.Path('/usr/local/share/patchpilot-wheels'); \
whl = wdir / 'patchpilot_dummy_dep-0.1.0-py3-none-any.whl'; \
mod = b'VALUE = \"installed_from_wheel\"\n'; \
meta = b'Metadata-Version: 2.1\nName: patchpilot-dummy-dep\nVersion: 0.1.0\n'; \
wheel = b'Wheel-Version: 1.0\nGenerator: patchpilot\nRoot-Is-Purelib: true\nTag: py3-none-any\n'; \
h = lambda b: 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode(); \
rec = f'patchpilot_dummy_dep/__init__.py,{h(mod)},{len(mod)}\npatchpilot_dummy_dep-0.1.0.dist-info/METADATA,{h(meta)},{len(meta)}\npatchpilot_dummy_dep-0.1.0.dist-info/WHEEL,{h(wheel)},{len(wheel)}\npatchpilot_dummy_dep-0.1.0.dist-info/RECORD,,\n'.encode(); \
z = zipfile.ZipFile(whl, 'w'); \
z.writestr('patchpilot_dummy_dep/__init__.py', mod); \
z.writestr('patchpilot_dummy_dep-0.1.0.dist-info/METADATA', meta); \
z.writestr('patchpilot_dummy_dep-0.1.0.dist-info/WHEEL', wheel); \
z.writestr('patchpilot_dummy_dep-0.1.0.dist-info/RECORD', rec); \
z.close(); \
sdist = wdir / 'malicious_sdist-0.1.0.tar.gz'; \
t = tarfile.open(sdist, 'w:gz'); \
setup_py = b'import pathlib; pathlib.Path(\"/deps/pwned_by_sdist\").write_text(\"pwned\"); raise SystemExit(\"setup.py ran\")\n'; \
ti = tarfile.TarInfo('malicious_sdist-0.1.0/setup.py'); ti.size = len(setup_py); \
t.addfile(ti, io.BytesIO(setup_py)); \
pkg_info = b'Metadata-Version: 2.1\nName: malicious-sdist\nVersion: 0.1.0\n'; \
ti2 = tarfile.TarInfo('malicious_sdist-0.1.0/PKG-INFO'); ti2.size = len(pkg_info); \
t.addfile(ti2, io.BytesIO(pkg_info)); \
t.close()" \
    && chown -R 10001:10001 /work /deps \
    && chmod 0555 /usr/local/share/patchpilot-wheels
COPY --chown=root:root --chmod=0555 runner.py /usr/local/bin/patchpilot-runner.py
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["python", "-u", "/usr/local/bin/patchpilot-runner.py"]
