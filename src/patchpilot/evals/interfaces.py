"""Abstract interfaces and protocol definitions for Phase 3b dataset extensions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

from patchpilot.evals.models import CaseDefinition, CaseMetadata


@runtime_checkable
class DatasetBuilder(Protocol):
    """Protocol for dataset builders across evaluation and training pools."""

    @property
    def pool_name(self) -> str:
        """Name of the target pool (e.g. 'realworld_eval', 'findings_eval')."""
        ...

    def build(self, output_dir: Path) -> list[CaseDefinition]:
        """Build and materialize cases into the target directory."""
        ...


class RealworldEvalProvider(ABC):
    """Interface for Phase 3b real-world evaluation dataset loader.

    Materializes offline curated bug-fix commits from allowlisted open-source repositories
    under MIT, Apache-2.0, BSD-2-Clause, or BSD-3-Clause licenses.
    """

    @abstractmethod
    def load_manifest(self, manifest_path: Path) -> list[CaseDefinition]:
        """Load curated real-world bug-fix cases from an offline manifest."""
        raise NotImplementedError

    @abstractmethod
    def validate_licenses(self, cases: list[CaseDefinition]) -> bool:
        """Assert all cases specify SPDX license identifier in allowlist."""
        raise NotImplementedError


class FindingsEvalProvider(ABC):
    """Interface for Phase 3b PR findings review evaluation dataset loader.

    Materializes labelled PR review findings with human true/false ground truth labels.
    """

    @abstractmethod
    def load_findings(self, findings_path: Path) -> list[CaseMetadata]:
        """Load labelled PR review findings from curated offline files."""
        raise NotImplementedError

    @abstractmethod
    def compute_review_metrics(
        self,
        predicted: list[CaseMetadata],
        ground_truth: list[CaseMetadata],
    ) -> dict[str, float]:
        """Compute review precision, recall, and F1 score against ground truth."""
        raise NotImplementedError
