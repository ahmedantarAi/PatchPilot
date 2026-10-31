"""Sequential sandbox verification harness with per-case content+digest caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from patchpilot.evals.generator import load_pool_cases
from patchpilot.evals.models import CaseDefinition, VerifyCaseResult, VerifySummaryReport
from patchpilot.sandbox.runner import SANDBOX_IMAGE, SandboxRunner, get_image_digest

logger = logging.getLogger(__name__)


def compute_verify_cache_key(case_content_hash: str, image_digest: str) -> str:
    """Compute deterministic cache key: sha256(case_content_hash:image_digest)."""
    raw = f"{case_content_hash}:{image_digest}".encode()
    return hashlib.sha256(raw).hexdigest()


class VerifyCache:
    """Persistent on-disk cache for verification results keyed by (case_hash, image_digest)."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._entries: dict[str, dict[str, object]] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._entries = {k: v for k, v in data.items() if isinstance(v, dict)}
            except Exception as exc:
                logger.warning("Failed to load verify cache at %s: %s", self.cache_path, exc)
                self._entries = {}

    def get(self, key: str) -> dict[str, object] | None:
        return self._entries.get(key)

    def set(self, key: str, value: dict[str, object]) -> None:
        self._entries[key] = value
        self.save()

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")


def verify_single_case(
    case: CaseDefinition,
    runner: SandboxRunner,
    image_digest: str,
    cache: VerifyCache | None = None,
) -> VerifyCaseResult:
    """Verify one case sequentially inside a fresh sandbox container or load from cache."""
    cache_key = compute_verify_cache_key(case.metadata.content_hash, image_digest)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            raw_dur = cached.get("duration_s", 0.0)
            raw_err = cached.get("error")
            return VerifyCaseResult(
                case_id=case.metadata.case_id,
                pool=case.metadata.pool,
                content_hash=case.metadata.content_hash,
                verified=bool(cached.get("verified", False)),
                pre_fix_failed=bool(cached.get("pre_fix_failed", False)),
                patch_applied=bool(cached.get("patch_applied", False)),
                post_fix_passed=bool(cached.get("post_fix_passed", False)),
                duration_s=float(str(raw_dur)),
                cached=True,
                error=str(raw_err) if raw_err is not None else None,
                container_name="[cached]",
                image_digest=image_digest,
            )

    t0 = time.monotonic()
    result = runner.run_sync(
        files=case.files,
        symlinks=case.symlinks,
        patch_diff=case.patch_diff,
        pytest_args=case.metadata.pytest_args,
        verify=True,
    )
    duration_s = round(time.monotonic() - t0, 4)

    is_verified = bool(result.verified)
    pre_failed = bool(result.pre_fix_failed)
    patch_applied = bool(result.patch_applied)
    post_passed = bool(result.post_fix_passed)

    verify_res = VerifyCaseResult(
        case_id=case.metadata.case_id,
        pool=case.metadata.pool,
        content_hash=case.metadata.content_hash,
        verified=is_verified,
        pre_fix_failed=pre_failed,
        patch_applied=patch_applied,
        post_fix_passed=post_passed,
        duration_s=duration_s,
        cached=False,
        error=result.error,
        container_name=result.container_name,
        image_digest=result.image_digest,
    )

    if is_verified and cache is not None:
        cache.set(
            cache_key,
            {
                "verified": is_verified,
                "pre_fix_failed": pre_failed,
                "patch_applied": patch_applied,
                "post_fix_passed": post_passed,
                "duration_s": duration_s,
                "error": result.error,
            },
        )

    return verify_res


def run_verify(
    datasets_dir: Path,
    *,
    pools: list[str] | None = None,
    limit: int | None = None,
    cache_path: Path | None = None,
    report_path: Path | None = None,
    image: str = SANDBOX_IMAGE,
) -> VerifySummaryReport:
    """Run sequential sandbox verification on seeded cases (pre-fix fail, post-fix pass)."""
    if pools is None:
        pools = ["seeded_eval", "seeded_train"]

    effective_cache_path = cache_path or (datasets_dir.parent / ".cache" / "verify_cache.json")
    cache = VerifyCache(effective_cache_path)

    # 1. Check overlap between seeded_eval and seeded_train
    eval_cases = load_pool_cases(datasets_dir, "seeded_eval")
    train_cases = load_pool_cases(datasets_dir, "seeded_train")

    eval_hashes = {c.metadata.content_hash for c in eval_cases}
    train_hashes = {c.metadata.content_hash for c in train_cases}
    overlap_count = len(eval_hashes.intersection(train_hashes))

    # 2. Collect target cases
    target_cases: list[CaseDefinition] = []
    for pool_name in pools:
        if pool_name == "seeded_eval":
            target_cases.extend(eval_cases)
        elif pool_name == "seeded_train":
            target_cases.extend(train_cases)
        else:
            target_cases.extend(load_pool_cases(datasets_dir, pool_name))

    if limit is not None and limit > 0:
        target_cases = target_cases[:limit]

    runner = SandboxRunner(image=image, cleanup_on_start=True)
    image_digest = get_image_digest(image)

    t_total_start = time.monotonic()
    case_results: list[VerifyCaseResult] = []
    cached_count = 0
    verified_count = 0

    total_target = len(target_cases)
    for idx, case in enumerate(target_cases, start=1):
        c_res = verify_single_case(case, runner=runner, image_digest=image_digest, cache=cache)
        case_results.append(c_res)
        if c_res.cached:
            cached_count += 1
        if c_res.verified:
            verified_count += 1

        if idx % 20 == 0 or idx == total_target or not c_res.cached:
            logger.info(
                "[%d/%d] case=%s/%s verified=%s cached=%s duration=%.2fs",
                idx,
                total_target,
                case.metadata.pool,
                case.metadata.case_id,
                c_res.verified,
                c_res.cached,
                c_res.duration_s,
            )

    total_duration = round(time.monotonic() - t_total_start, 2)
    failed_count = total_target - verified_count

    report = VerifySummaryReport(
        total=total_target,
        verified=verified_count,
        failed=failed_count,
        cached=cached_count,
        duration_s=total_duration,
        train_eval_overlap=overlap_count,
        cases=case_results,
    )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return report
