# Base image for running UNTRUSTED patches/tests. Launched by the host orchestrator with docker run; hardening flags are in SPEC.md section 6. Contains toolchains only; no secrets, no repo code.
FROM python:3.11.18-slim-bookworm
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin runner \
    && pip install --no-cache-dir pytest==9.2.0
USER 10001
WORKDIR /work
