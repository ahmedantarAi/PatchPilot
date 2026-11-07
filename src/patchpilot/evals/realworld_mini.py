"""Mini Phase 3b: realworld_mini dataset builder, sandbox verifier, and separate freeze guard."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from patchpilot.evals.freeze import compute_overall_freeze_sha256
from patchpilot.evals.generator import load_pool_cases, materialize_case
from patchpilot.evals.interfaces import RealworldEvalProvider
from patchpilot.evals.models import (
    CaseDefinition,
    CaseMetadata,
    FrozenManifest,
    FrozenPoolManifest,
    VerifyCaseResult,
)
from patchpilot.evals.taxonomy import compute_case_content_hash
from patchpilot.evals.verifier import VerifyCache, verify_single_case
from patchpilot.sandbox.runner import SANDBOX_IMAGE, SandboxRunner, get_image_digest

REALWORLD_MINI_POOL = "realworld_mini"
DEFAULT_FROZEN_REALWORLD_MINI = Path("evals/FROZEN_REALWORLD_MINI.json")
DEFAULT_CLONE_CACHE_DIR = Path(".work/realworld_src")
ALLOWED_LICENSES: frozenset[str] = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})


@dataclass(frozen=True)
class RealworldMiniSpec:
    """Curated specification for a verified real-world single-file bug-fix commit."""

    case_id: str
    repo_name: str
    repo_url: str
    fix_commit: str
    parent_commit: str
    source_file: str
    test_file_git: str
    test_file_repo: str
    failing_test: str
    license: str
    license_source: str
    taxonomy: str
    description: str
    borderline: bool = False
    support_files: tuple[str, ...] = ()


REALWORLD_MINI_SPECS: tuple[RealworldMiniSpec, ...] = (
    RealworldMiniSpec(
        case_id="0001",
        repo_name="dacite",
        repo_url="https://github.com/konradhalas/dacite",
        fix_commit="45c2993c7075ae69659b956f89404ed5ee9ef118",
        parent_commit="10a9ec40fc5874ae434aa68b975d1b1bf667a42f",
        source_file="dacite/cache.py",
        test_file_git="tests/test_types.py",
        test_file_repo="tests/test_types.py",
        failing_test="tests/test_types.py::test_optional_and_union_none_does_not_pollute_scope_via_caching",
        license="MIT",
        license_source="LICENSE",
        taxonomy="caching_type_collision",
        description=(
            "cache() wraps lru_cache without typed=True, causing Union and Optional "
            "type checks with None to collide (#258)."
        ),
        support_files=("tests/__init__.py", "tests/common.py"),
    ),
    RealworldMiniSpec(
        case_id="0002",
        repo_name="dacite",
        repo_url="https://github.com/konradhalas/dacite",
        fix_commit="eae039bd25b2d0415de8e0d37cd07aca6c481c5c",
        parent_commit="d2206b2e4711859da0ea5862c395940f33693e80",
        source_file="dacite/core.py",
        test_file_git="tests/core/test_config.py",
        test_file_repo="tests/core/test_config.py",
        failing_test="tests/core/test_config.py::test_from_dict_with_type_hook_exception",
        license="MIT",
        license_source="LICENSE",
        taxonomy="exception_handling",
        description=(
            "from_dict wraps transform_value in try/except KeyError, swallowing "
            "KeyError raised inside user type_hooks (#140)."
        ),
    ),
    RealworldMiniSpec(
        case_id="0003",
        repo_name="schema",
        repo_url="https://github.com/keleshev/schema",
        fix_commit="a651b394a23bb7081d56befa8512aa2cb5d76736",
        parent_commit="23b729ec91cf12cbaf076e3171b68dd6ed49de0a",
        source_file="schema.py",
        test_file_git="test_schema.py",
        test_file_repo="test_schema.py",
        failing_test="test_schema.py::test_schema",
        license="MIT",
        license_source="LICENSE-MIT",
        taxonomy="type_check",
        description=(
            "Schema(int).validate(True) accepts bool values as int because "
            "isinstance(True, int) is True in Python (#176)."
        ),
        support_files=("LICENSE-MIT",),
    ),
    RealworldMiniSpec(
        case_id="0004",
        repo_name="schema",
        repo_url="https://github.com/keleshev/schema",
        fix_commit="d8dc368dd016107c777b8a00559d43af772177a7",
        parent_commit="b17b7b2c44c2936955f485f6555b4986303c5101",
        source_file="schema.py",
        test_file_git="test_schema.py",
        test_file_repo="test_schema.py",
        failing_test="test_schema.py::test_or",
        license="MIT",
        license_source="LICENSE-MIT",
        taxonomy="none_handling",
        description=(
            "Empty Or().validate(data) initializes x = None and raises TypeError "
            "on raise x instead of SchemaError([], []) (#13)."
        ),
    ),
    RealworldMiniSpec(
        case_id="0005",
        repo_name="jsonpointer",
        repo_url="https://github.com/stefankoegl/python-json-pointer",
        fix_commit="9b78c3b495b719b981a14f9aa55a8616551ecd73",
        parent_commit="ead884aeae7c721904797f8402dba263762ac869",
        source_file="jsonpointer.py",
        test_file_git="tests.py",
        test_file_repo="test_jsonpointer.py",
        failing_test="test_jsonpointer.py::SetTests::test_set_create_immutable_sequence",
        license="BSD-3-Clause",
        license_source="LICENSE.txt",
        taxonomy="exception_handling",
        description=(
            "Appending via '-' pointer to an immutable Sequence (tuple) leaks "
            "AttributeError instead of raising JsonPointerException."
        ),
        support_files=("bin/jsonpointer",),
    ),
    RealworldMiniSpec(
        case_id="0006",
        repo_name="jsonpointer",
        repo_url="https://github.com/stefankoegl/python-json-pointer",
        fix_commit="d3966dd34c681a5e1c790ca212a020d83bca26ac",
        parent_commit="490c7a25147b0674fc0234d0462726595bd4fd87",
        source_file="jsonpointer.py",
        test_file_git="tests.py",
        test_file_repo="test_jsonpointer.py",
        failing_test="test_jsonpointer.py::ComparisonTests::test_contains",
        license="BSD-3-Clause",
        license_source="COPYING",
        taxonomy="wrong_operator",
        description=(
            "[borderline] JsonPointer.contains() uses inverted length check and "
            "omits __contains__, returning False for valid sub-pointers."
        ),
        borderline=True,
    ),
    RealworldMiniSpec(
        case_id="0007",
        repo_name="parse",
        repo_url="https://github.com/r1chardj0n3s/parse",
        fix_commit="52bf0b3862c6e21271e9582a3a98915247228763",
        parent_commit="cc86df74f565265a4c03326bbcef06e08f01cec2",
        source_file="parse/__init__.py",
        test_file_git="tests/test_bugs.py",
        test_file_repo="tests/test_bugs.py",
        failing_test="tests/test_bugs.py::test_time_type_requires_an_actual_time",
        license="MIT",
        license_source="LICENSE",
        taxonomy="regex_pattern",
        description=(
            "Time format specifier 'tt' marks TIME_PAT as optional (r'%s?%s?%s?'), "
            "matching empty strings and yielding time(0, 0)."
        ),
    ),
    RealworldMiniSpec(
        case_id="0008",
        repo_name="parse",
        repo_url="https://github.com/r1chardj0n3s/parse",
        fix_commit="ba6803d57179a4b026c8ed51d970156653153b62",
        parent_commit="738447715c76f8ebc460b33320307faccafbc599",
        source_file="parse.py",
        test_file_git="tests/test_bugs.py",
        test_file_repo="tests/test_bugs.py",
        failing_test="tests/test_bugs.py::test_microsecond_precision_issue162",
        license="MIT",
        license_source="LICENSE",
        taxonomy="precision_loss",
        description=(
            "date_convert computes microseconds via int(float('.' + u) * 1000000), "
            "losing precision due to float rounding (#162)."
        ),
    ),
    RealworldMiniSpec(
        case_id="0009",
        repo_name="docopt-ng",
        repo_url="https://github.com/jazzband/docopt-ng",
        fix_commit="6b06b5a7305153e9823cd0e6b32d8ff7d5762403",
        parent_commit="5112ecc663a18a0b25c361b2a30c55b64311d285",
        source_file="docopt.py",
        test_file_git="test_docopt.py",
        test_file_repo="test_docopt.py",
        failing_test="test_docopt.py::test_issue_126_defaults_not_parsed_correctly_when_tabs",
        license="MIT",
        license_source="LICENSE-MIT",
        taxonomy="regex_pattern",
        description=(
            "parse_defaults splits option lines on spaces only ('\\n *(-\\S+?)'), "
            "failing when option lines use tabs (#126)."
        ),
    ),
    RealworldMiniSpec(
        case_id="0010",
        repo_name="docopt-ng",
        repo_url="https://github.com/jazzband/docopt-ng",
        fix_commit="fa484c3e457108d03a03930d83d2729b8c659129",
        parent_commit="78bdf1814abbfd8cfd5d5ba35fcaf6cdea61b190",
        source_file="docopt.py",
        test_file_git="test_docopt.py",
        test_file_repo="test_docopt.py",
        failing_test="test_docopt.py::test_printable_and_formal_usage",
        license="MIT",
        license_source="LICENSE-MIT",
        taxonomy="string_parsing",
        description=(
            "formal_usage does not strip leading/trailing pipes around usage tokens "
            "before wrapping alternatives."
        ),
    ),
    RealworldMiniSpec(
        case_id="0011",
        repo_name="python-semver",
        repo_url="https://github.com/python-semver/python-semver",
        fix_commit="f332326e54a5582092b50c8fa113d11bbdf1a9e6",
        parent_commit="69d1cf205aba68f2e20f0d321d740db5ddce36a1",
        source_file="semver.py",
        test_file_git="test_semver.py",
        test_file_repo="test_semver.py",
        failing_test="test_semver.py::test_version_info_should_throw_index_error",
        license="BSD-3-Clause",
        license_source="LICENSE.txt",
        taxonomy="none_handling",
        description=(
            "VersionInfo.__getitem__ uses filter(None, ...) and mis-checks negative "
            "slice indices, dropping valid 0 version parts (#260)."
        ),
    ),
    RealworldMiniSpec(
        case_id="0012",
        repo_name="python-semver",
        repo_url="https://github.com/python-semver/python-semver",
        fix_commit="f55559acfe25e6edf57614a4650b9b7d47bd4879",
        parent_commit="cf18005a6ca1fa2c4d45f2011d05d4f733d81182",
        source_file="semver.py",
        test_file_git="test_semver.py",
        test_file_repo="test_semver.py",
        failing_test="test_semver.py::test_next_version_with_versioninfo",
        license="BSD-3-Clause",
        license_source="LICENSE.txt",
        taxonomy="return_type",
        description=(
            "VersionInfo.next_version() wraps bump_<part>() in str(...), returning "
            "str instead of VersionInfo (#251)."
        ),
    ),
    RealworldMiniSpec(
        case_id="0013",
        repo_name="tabulate",
        repo_url="https://github.com/astanin/python-tabulate",
        fix_commit="41a6db29b666f0c0eedec40efb5bc3af4f411a56",
        parent_commit="7d6dfc1126e4da7d7a7547f1e1626fe676a5462f",
        source_file="tabulate.py",
        test_file_git="test/test_regression.py",
        test_file_repo="tests/test_regression.py",
        failing_test="tests/test_regression.py::test_simple_separated_format_with_headers",
        license="MIT",
        license_source="LICENSE",
        taxonomy="none_handling",
        description=(
            "simple_separated_format sets headerrow=None instead of "
            "DataRow('', separator, ''), raising TypeError with headers (#15)."
        ),
    ),
    RealworldMiniSpec(
        case_id="0014",
        repo_name="mistune",
        repo_url="https://github.com/lepture/mistune",
        fix_commit="eb32fe5e9c99dc15ebfd15a69f82c73f3c96c111",
        parent_commit="24bc03137cfc3b0c7103ff9277df15f6030e7c90",
        source_file="mistune/renderers.py",
        test_file_git="tests/test_misc.py",
        test_file_repo="tests/test_misc.py",
        failing_test="tests/test_misc.py::TestMiscCases::test_harmful_links",
        license="BSD-3-Clause",
        license_source="LICENSE",
        taxonomy="security_sanitization",
        description=(
            "HTMLRenderer._safe_url checks url.startswith(s) without .lower(), "
            "allowing mixed-case JavaScript: links through."
        ),
    ),
    RealworldMiniSpec(
        case_id="0015",
        repo_name="tinydb",
        repo_url="https://github.com/msiemens/tinydb",
        fix_commit="dcf0a0137b62d5a2d7bfea842d60cc529ebfba2e",
        parent_commit="10644a0e07ad180c5b756aba272ee6b0dbd12df8",
        source_file="tinydb/utils.py",
        test_file_git="tests/test_utils.py",
        test_file_repo="tests/test_utils.py",
        failing_test="tests/test_utils.py::test_lru_cache_falsy_values_bug",
        license="MIT",
        license_source="LICENSE",
        taxonomy="falsy_value_handling",
        description=(
            "LRUCache.set checks if self.cache.get(key): instead of if key in "
            "self.cache:, evicting keys with falsy values."
        ),
    ),
)

SPEC_BY_CASE_ID: dict[str, RealworldMiniSpec] = {s.case_id: s for s in REALWORLD_MINI_SPECS}


def _git(repo_dir: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.longpaths=true", "-C", str(repo_dir), *args],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} in {repo_dir} failed: {stderr}")
    return proc.stdout.decode("utf-8", "replace")


def ensure_repo_cloned(repo_name: str, repo_url: str, clone_cache_dir: Path) -> Path:
    """Ensure the repository is cloned on the host under clone_cache_dir/<repo_name>."""
    clone_cache_dir.mkdir(parents=True, exist_ok=True)
    dest = clone_cache_dir / repo_name
    if (dest / ".git").exists() or (dest / "HEAD").exists():
        return dest

    proc = subprocess.run(
        [
            "git",
            "-c",
            "core.longpaths=true",
            "clone",
            "--quiet",
            "--no-checkout",
            "--no-tags",
            repo_url,
            str(dest),
        ],
        capture_output=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"Failed to clone {repo_url} into {dest}: {stderr}")
    return dest


class RealworldMiniProvider(RealworldEvalProvider):
    """Materializes and validates the `realworld_mini` evaluation pool."""

    def __init__(
        self,
        specs: tuple[RealworldMiniSpec, ...] = REALWORLD_MINI_SPECS,
        clone_cache_dir: Path = DEFAULT_CLONE_CACHE_DIR,
    ) -> None:
        self.specs = specs
        self.clone_cache_dir = clone_cache_dir

    def load_manifest(self, manifest_path: Path | None = None) -> list[CaseDefinition]:
        """Build `CaseDefinition` objects from the verified realworld_mini specs."""
        cases: list[CaseDefinition] = []
        for spec in self.specs:
            case, _ = self._build_single_case(spec)
            cases.append(case)
        if not self.validate_licenses(cases):
            raise ValueError("One or more realworld_mini cases failed SPDX license validation")
        return cases

    def validate_licenses(self, cases: list[CaseDefinition]) -> bool:
        """Assert all cases specify an allowlisted SPDX license and license_source."""
        for case in cases:
            if case.metadata.license not in ALLOWED_LICENSES:
                return False
            if not case.metadata.license_source.strip():
                return False
        return True

    def _build_single_case(self, spec: RealworldMiniSpec) -> tuple[CaseDefinition, str]:
        """Extract buggy files at parent_commit, test at fix_commit, diff, and LICENSE text."""
        repo_dir = ensure_repo_cloned(spec.repo_name, spec.repo_url, self.clone_cache_dir)

        # Verify both commits exist in git
        if _git(repo_dir, "cat-file", "-t", spec.fix_commit).strip() != "commit":
            raise ValueError(f"Fix commit {spec.fix_commit} not found in {spec.repo_name}")
        if _git(repo_dir, "cat-file", "-t", spec.parent_commit).strip() != "commit":
            raise ValueError(f"Parent commit {spec.parent_commit} not found in {spec.repo_name}")

        # Collect package .py files at parent_commit
        files: dict[str, str] = {}
        top_entry = spec.source_file.split("/")[0] if "/" in spec.source_file else spec.source_file
        tree_out = _git(
            repo_dir, "ls-tree", "-r", "--name-only", spec.parent_commit, "--", top_entry
        )
        for rel_path in tree_out.splitlines():
            rel_clean = rel_path.strip()
            if not rel_clean.endswith(".py"):
                continue
            # Exclude optional mypy plugin that imports third-party `mypy`
            if rel_clean.endswith("mypy_plugin.py"):
                continue
            files[rel_clean] = _git(repo_dir, "show", f"{spec.parent_commit}:{rel_clean}")

        # Load any required repo support files at parent_commit
        for sup_path in spec.support_files:
            files[sup_path] = _git(repo_dir, "show", f"{spec.parent_commit}:{sup_path}")

        # Load the regression test file from fix_commit
        test_content = _git(repo_dir, "show", f"{spec.fix_commit}:{spec.test_file_git}")
        files[spec.test_file_repo] = test_content

        # Extract the ground-truth unified diff for source_file only
        patch_diff = _git(
            repo_dir,
            "diff",
            spec.parent_commit,
            spec.fix_commit,
            "--",
            spec.source_file,
        )
        if not patch_diff.strip():
            raise ValueError(f"Empty patch_diff for {spec.case_id} ({spec.repo_name})")

        # Read the upstream LICENSE file at fix_commit
        license_text = _git(repo_dir, "show", f"{spec.fix_commit}:{spec.license_source}")
        if not license_text.strip():
            raise ValueError(f"Empty license file {spec.license_source} in {spec.repo_name}")

        content_hash = compute_case_content_hash(files, patch_diff)
        meta = CaseMetadata(
            case_id=spec.case_id,
            pool=REALWORLD_MINI_POOL,
            taxonomy=spec.taxonomy,
            template=f"{spec.repo_name}:{spec.source_file}",
            mutation_signature=f"{spec.repo_name}@{spec.fix_commit[:12]}",
            seed=0,
            content_hash=content_hash,
            patch_diff=patch_diff,
            pytest_args=["-q"],
            license=spec.license,
            license_source=f"{spec.repo_url}/blob/{spec.fix_commit}/{spec.license_source}",
            description=spec.description,
        )
        case = CaseDefinition(
            metadata=meta,
            files=files,
            symlinks={},
            patch_diff=patch_diff,
        )
        return case, license_text


def build_and_verify_realworld_mini(
    datasets_dir: Path,
    *,
    clone_cache_dir: Path = DEFAULT_CLONE_CACHE_DIR,
    cache_path: Path | None = None,
    image: str = SANDBOX_IMAGE,
) -> tuple[list[VerifyCaseResult], list[VerifyCaseResult]]:
    """Build `realworld_mini`, copy LICENSE into each case dir, and verify in sandbox.

    Any case that does not fail before the fix (`pre_fix_failed == True`) and pass after
    the fix (`post_fix_passed == True`) inside the `--network none` sandbox is dropped.
    """
    provider = RealworldMiniProvider(clone_cache_dir=clone_cache_dir)
    pool_dir = datasets_dir / REALWORLD_MINI_POOL
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    built_cases: list[CaseDefinition] = []
    for spec in provider.specs:
        case, license_text = provider._build_single_case(spec)
        case_dir = materialize_case(case, datasets_dir)
        # Requirement 5: Copy each repo's LICENSE file into its case folder (outside repo/)
        (case_dir / "LICENSE").write_text(license_text, encoding="utf-8")
        built_cases.append(case)

    if not provider.validate_licenses(built_cases):
        raise ValueError("License validation failed for realworld_mini")

    effective_cache_path = cache_path or (
        datasets_dir.parent / ".cache" / "verify_realworld_mini_cache.json"
    )
    cache = VerifyCache(effective_cache_path)
    runner = SandboxRunner(image=image, cleanup_on_start=True)
    image_digest = get_image_digest(image)

    kept_results: list[VerifyCaseResult] = []
    dropped_results: list[VerifyCaseResult] = []

    for case in built_cases:
        res = verify_single_case(case, runner=runner, image_digest=image_digest, cache=cache)
        if res.verified and res.pre_fix_failed and res.patch_applied and res.post_fix_passed:
            kept_results.append(res)
        else:
            dropped_results.append(res)
            bad_dir = pool_dir / case.metadata.case_id
            if bad_dir.exists():
                shutil.rmtree(bad_dir)

    return kept_results, dropped_results


def freeze_realworld_mini(
    datasets_dir: Path,
    output_path: Path = DEFAULT_FROZEN_REALWORLD_MINI,
) -> FrozenManifest:
    """Freeze `realworld_mini` into a separate manifest (`evals/FROZEN_REALWORLD_MINI.json`)."""
    cases = load_pool_cases(datasets_dir, REALWORLD_MINI_POOL)
    if not cases:
        raise ValueError(f"Cannot freeze empty pool '{REALWORLD_MINI_POOL}' in {datasets_dir}")

    case_hashes: dict[str, str] = {}
    for case in cases:
        content_hash = compute_case_content_hash(case.files, case.patch_diff)
        case_hashes[case.metadata.case_id] = content_hash

    frozen_pools = {
        REALWORLD_MINI_POOL: FrozenPoolManifest(
            count=len(cases),
            cases=case_hashes,
        )
    }
    overall_sha256 = compute_overall_freeze_sha256(frozen_pools)
    manifest = FrozenManifest(
        generator_version="1.0.0",
        base_seed=0,
        pools=frozen_pools,
        overall_sha256=overall_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def verify_frozen_realworld_mini(
    frozen_path: Path = DEFAULT_FROZEN_REALWORLD_MINI,
    datasets_dir: Path = Path("evals/datasets"),
) -> tuple[bool, str]:
    """Fail-closed verification of `realworld_mini` against `FROZEN_REALWORLD_MINI.json`."""
    if not frozen_path.exists():
        return False, f"Frozen realworld_mini manifest not found: {frozen_path}"

    try:
        manifest = FrozenManifest.model_validate_json(frozen_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse frozen realworld_mini manifest: {exc}"

    if REALWORLD_MINI_POOL not in manifest.pools:
        return False, f"Pool '{REALWORLD_MINI_POOL}' missing from {frozen_path}"

    expected_pool = manifest.pools[REALWORLD_MINI_POOL]
    current_cases = load_pool_cases(datasets_dir, REALWORLD_MINI_POOL)
    if len(current_cases) != expected_pool.count:
        return (
            False,
            f"Case count mismatch in '{REALWORLD_MINI_POOL}': "
            f"expected {expected_pool.count}, got {len(current_cases)}",
        )

    current_hashes: dict[str, str] = {}
    for case in current_cases:
        cid = case.metadata.case_id
        if cid not in expected_pool.cases:
            return False, f"Unexpected case '{cid}' in '{REALWORLD_MINI_POOL}'"
        if case.metadata.license not in ALLOWED_LICENSES:
            return False, f"Disallowed license '{case.metadata.license}' in case '{cid}'"
        license_file = datasets_dir / REALWORLD_MINI_POOL / cid / "LICENSE"
        if not license_file.is_file() or not license_file.read_text(encoding="utf-8").strip():
            return False, f"Missing or empty LICENSE file in case '{cid}'"

        actual_hash = compute_case_content_hash(case.files, case.patch_diff)
        expected_hash = expected_pool.cases[cid]
        if actual_hash != expected_hash:
            return (
                False,
                f"Content hash mismatch for '{REALWORLD_MINI_POOL}/{cid}': "
                f"expected {expected_hash}, got {actual_hash}",
            )
        current_hashes[cid] = actual_hash

    recomputed_pools = {
        REALWORLD_MINI_POOL: FrozenPoolManifest(
            count=len(current_cases),
            cases=current_hashes,
        )
    }
    recomputed_sha256 = compute_overall_freeze_sha256(recomputed_pools)
    if recomputed_sha256 != manifest.overall_sha256:
        return (
            False,
            f"Overall SHA-256 mismatch in {frozen_path}: "
            f"expected {manifest.overall_sha256}, got {recomputed_sha256}",
        )

    return True, f"OK: frozen {REALWORLD_MINI_POOL} split unchanged"
