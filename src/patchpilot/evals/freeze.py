"""Freezing and tampering-verification for the evaluation split (SPEC §7)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from patchpilot.evals.generator import load_pool_cases
from patchpilot.evals.models import FrozenManifest, FrozenPoolManifest
from patchpilot.evals.taxonomy import compute_case_content_hash

FREEZE_POOLS = ["seeded_eval", "safety"]


def compute_overall_freeze_sha256(pools: dict[str, FrozenPoolManifest]) -> str:
    """Compute deterministic SHA-256 over all frozen pool case hashes."""
    hasher = hashlib.sha256()
    for pool_name in sorted(pools.keys()):
        hasher.update(pool_name.encode("utf-8"))
        hasher.update(b"\x00")
        pool = pools[pool_name]
        for case_id in sorted(pool.cases.keys()):
            hasher.update(case_id.encode("utf-8"))
            hasher.update(b":")
            hasher.update(pool.cases[case_id].encode("utf-8"))
            hasher.update(b"\n")
    return hasher.hexdigest()


def freeze_eval_split(
    datasets_dir: Path,
    output_path: Path,
    base_seed: int = 1337,
) -> FrozenManifest:
    """Freeze all eval-only pools (seeded_eval, safety) into FROZEN.json."""
    frozen_pools: dict[str, FrozenPoolManifest] = {}

    for pool in FREEZE_POOLS:
        cases = load_pool_cases(datasets_dir, pool)
        if not cases:
            raise ValueError(f"Cannot freeze empty pool: {pool} in {datasets_dir}")
        case_hashes: dict[str, str] = {}
        for c in cases:
            # Recompute content hash directly from case files and patch to be certain
            content_hash = compute_case_content_hash(c.files, c.patch_diff)
            case_hashes[c.metadata.case_id] = content_hash

        frozen_pools[pool] = FrozenPoolManifest(
            count=len(cases),
            cases=case_hashes,
        )

    overall_sha256 = compute_overall_freeze_sha256(frozen_pools)
    manifest = FrozenManifest(
        generator_version="1.0.0",
        base_seed=base_seed,
        pools=frozen_pools,
        overall_sha256=overall_sha256,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def verify_frozen_eval_split(
    frozen_path: Path,
    datasets_dir: Path,
) -> tuple[bool, str]:
    """Verify that current on-disk eval cases strictly match FROZEN.json."""
    if not frozen_path.exists():
        return False, f"Frozen manifest not found: {frozen_path}"

    try:
        manifest = FrozenManifest.model_validate_json(frozen_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse frozen manifest: {exc}"

    recomputed_pools: dict[str, FrozenPoolManifest] = {}

    for pool_name, expected_pool in manifest.pools.items():
        current_cases = load_pool_cases(datasets_dir, pool_name)
        if len(current_cases) != expected_pool.count:
            return (
                False,
                f"Case count mismatch in pool '{pool_name}': "
                f"expected {expected_pool.count}, got {len(current_cases)}",
            )

        current_hashes: dict[str, str] = {}
        for case in current_cases:
            cid = case.metadata.case_id
            if cid not in expected_pool.cases:
                return False, f"Unexpected case '{cid}' found in frozen pool '{pool_name}'"

            actual_hash = compute_case_content_hash(case.files, case.patch_diff)
            expected_hash = expected_pool.cases[cid]
            if actual_hash != expected_hash:
                return (
                    False,
                    f"Content hash mismatch for case '{cid}' in pool '{pool_name}': "
                    f"expected {expected_hash}, got {actual_hash}",
                )
            current_hashes[cid] = actual_hash

        recomputed_pools[pool_name] = FrozenPoolManifest(
            count=len(current_cases),
            cases=current_hashes,
        )

    recomputed_sha256 = compute_overall_freeze_sha256(recomputed_pools)
    if recomputed_sha256 != manifest.overall_sha256:
        return (
            False,
            f"Overall SHA-256 mismatch: expected {manifest.overall_sha256}, "
            f"got {recomputed_sha256}",
        )

    return True, "OK: frozen eval split unchanged"
