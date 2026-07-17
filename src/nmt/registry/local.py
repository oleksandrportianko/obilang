"""File-locked JSON model registry with directional production pointers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

from nmt.utils.io import atomic_write_json, load_json
from nmt.utils.paths import ProjectPaths, validate_identifier

VersionStatus = Literal[
    "training", "candidate", "approved", "production", "rejected", "archived", "failed"
]
VERSION_SUFFIX = re.compile(r"-v(\d+)\.(\d+)\.(\d+)$")


class ModelVersion(BaseModel):
    """Complete JSON metadata for one immutable directional model artifact."""

    model_config = ConfigDict(extra="allow")

    version_id: str
    version_label: str
    parent_version: str | None = None
    created_at: str
    language_pair: str
    direction: str
    status: VersionStatus = "training"
    protected: bool = False
    notes: str = ""
    experiment_id: str
    model_configuration: dict[str, Any]
    training_configuration: dict[str, Any]
    tokenizer_version: str
    dataset_versions: list[str]
    checkpoint_path: str | None = None
    best_validation_score: float | None = None
    test_metrics: dict[str, float] = Field(default_factory=dict)
    regression_metrics: dict[str, float] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    random_seed: int
    training_duration_seconds: float | None = None
    failure_reason: str | None = None


class RegistryError(ValueError):
    """Raised for invalid version transitions, unknown IDs, or protected artifacts."""


class LocalModelRegistry:
    """Cross-platform local JSON implementation behind a replaceable registry API."""

    def __init__(self, paths: ProjectPaths, pair: str, direction: str) -> None:
        """Resolve and initialize one pair/direction registry location."""
        self.paths = paths
        self.pair = validate_identifier(pair, "pair")
        self.direction = validate_identifier(direction, "direction")
        self.root = paths.model_direction(pair, direction)
        self.registry_path = self.root / "registry.json"
        self.lock = FileLock(str(self.root / ".registry.lock"))

    def _empty(self) -> dict[str, Any]:
        """Return the on-disk schema for a new directional registry."""
        return {"schema_version": 1, "production_version": None, "versions": {}}

    def _read(self) -> dict[str, Any]:
        """Load the registry document or return a new empty schema."""
        document = load_json(self.registry_path, self._empty())
        if not isinstance(document, dict) or not isinstance(document.get("versions"), dict):
            raise RegistryError(f"Registry is corrupted or has an unsupported schema: {self.registry_path}")
        return document

    def _write(self, document: dict[str, Any]) -> None:
        """Atomically persist the registry while its caller holds the file lock."""
        atomic_write_json(self.registry_path, document)

    def list_versions(self) -> list[ModelVersion]:
        """Return all versions sorted by semantic version tuple."""
        with self.lock:
            versions = [ModelVersion.model_validate(item) for item in self._read()["versions"].values()]
        return sorted(versions, key=lambda item: self._version_tuple(item.version_id))

    @staticmethod
    def _version_tuple(version_id: str) -> tuple[int, int, int]:
        """Extract the numeric semantic suffix used for ordering and allocation."""
        match = VERSION_SUFFIX.search(version_id)
        if not match:
            raise RegistryError(f"Version ID has no semantic suffix: {version_id}")
        return tuple(int(value) for value in match.groups())  # type: ignore[return-value]

    def resolve(self, version: str) -> ModelVersion:
        """Resolve an exact ID, semantic label, or the `production` alias.

        Raises:
            RegistryError: If no production pointer or matching version exists.
        """
        with self.lock:
            document = self._read()
            requested = document.get("production_version") if version == "production" else version
            if not requested:
                raise RegistryError(f"No production model is set for {self.pair}/{self.direction}.")
            exact = document["versions"].get(requested)
            if exact:
                return ModelVersion.model_validate(exact)
            matches = [
                value
                for value in document["versions"].values()
                if value.get("version_label") == requested
            ]
            if len(matches) == 1:
                return ModelVersion.model_validate(matches[0])
        raise RegistryError(f"Unknown model version {version!r} for {self.pair}/{self.direction}.")

    def allocate_version(self, parent_version: str | None = None) -> tuple[str, str]:
        """Allocate the next fresh-major or child-minor semantic version.

        The allocation is advisory until ``add`` is called. Training commands hold
        no long transaction, so ``add`` also rejects a rare concurrent collision.
        """
        versions = self.list_versions()
        tuples = [self._version_tuple(item.version_id) for item in versions]
        if parent_version is None:
            major = max((item[0] for item in tuples), default=0) + 1
            numbers = (major, 0, 0)
        else:
            parent = self.resolve(parent_version)
            parent_numbers = self._version_tuple(parent.version_id)
            same_major_minors = [item[1] for item in tuples if item[0] == parent_numbers[0]]
            numbers = (parent_numbers[0], max(same_major_minors, default=parent_numbers[1]) + 1, 0)
        label = f"v{numbers[0]}.{numbers[1]}.{numbers[2]}"
        return f"{self.pair}-{self.direction}-{label}", label

    def add(self, version: ModelVersion) -> None:
        """Register one new immutable identity and write its metadata snapshot.

        Raises:
            RegistryError: If the identity already exists or pair/direction differ.
        """
        if version.language_pair != self.pair or version.direction != self.direction:
            raise RegistryError("Version pair/direction does not match this registry.")
        with self.lock:
            document = self._read()
            if version.version_id in document["versions"]:
                raise RegistryError(f"Version already exists: {version.version_id}")
            document["versions"][version.version_id] = version.model_dump(mode="json")
            self._write(document)
        self._write_version_snapshot(version)

    def _write_version_snapshot(self, version: ModelVersion) -> None:
        """Mirror metadata beside immutable version artifacts for portability."""
        destination = self.root / "versions" / version.version_id / "metadata.json"
        atomic_write_json(destination, version.model_dump(mode="json"))

    def update(self, version_id: str, **changes: Any) -> ModelVersion:
        """Update lifecycle/metrics fields without changing identity or lineage.

        Identity, pair, direction, creation time, tokenizer, and parent references
        cannot change after registration.
        """
        immutable_fields = {
            "version_id",
            "version_label",
            "parent_version",
            "created_at",
            "language_pair",
            "direction",
            "tokenizer_version",
        }
        forbidden = immutable_fields.intersection(changes)
        if forbidden:
            raise RegistryError(f"Cannot mutate immutable version fields: {sorted(forbidden)}")
        with self.lock:
            document = self._read()
            if version_id not in document["versions"]:
                raise RegistryError(f"Unknown model version: {version_id}")
            candidate = dict(document["versions"][version_id])
            candidate.update(changes)
            version = ModelVersion.model_validate(candidate)
            document["versions"][version_id] = version.model_dump(mode="json")
            self._write(document)
        self._write_version_snapshot(version)
        return version

    def promote(self, version_id: str, manual_override: bool = False) -> ModelVersion:
        """Make an approved candidate production and demote the previous pointer.

        Args:
            version_id: Exact registered identity.
            manual_override: Allow a candidate or rejected model to bypass gate status.

        Returns:
            Updated production metadata.

        Raises:
            RegistryError: If status is ineligible without an override.
        """
        with self.lock:
            document = self._read()
            raw = document["versions"].get(version_id)
            if not raw:
                raise RegistryError(f"Unknown model version: {version_id}")
            if raw["status"] not in {"approved", "production"} and not manual_override:
                raise RegistryError(
                    f"Version {version_id} is {raw['status']}; approve it or use an explicit manual override."
                )
            previous_id = document.get("production_version")
            if previous_id and previous_id != version_id:
                previous = dict(document["versions"][previous_id])
                previous["status"] = "approved"
                document["versions"][previous_id] = previous
            updated = dict(raw)
            updated["status"] = "production"
            updated["protected"] = True
            document["versions"][version_id] = updated
            document["production_version"] = version_id
            self._write(document)
        if previous_id and previous_id != version_id:
            self._write_version_snapshot(ModelVersion.model_validate(previous))
        version = ModelVersion.model_validate(updated)
        self._write_version_snapshot(version)
        return version

    def rollback(self, version_id: str) -> ModelVersion:
        """Move the production pointer to a prior approved/protected model."""
        version = self.resolve(version_id)
        if version.status not in {"approved", "production", "archived"} and not version.protected:
            raise RegistryError(
                f"Rollback target {version.version_id} must be approved, archived, production, or protected."
            )
        return self.promote(version.version_id, manual_override=True)

    def protect(self, version_id: str, protected: bool = True) -> ModelVersion:
        """Set deletion protection metadata for an important version."""
        return self.update(version_id, protected=protected)


def new_model_version(
    registry: LocalModelRegistry,
    experiment_id: str,
    model_configuration: dict[str, Any],
    training_configuration: dict[str, Any],
    tokenizer_version: str,
    dataset_versions: list[str],
    environment: dict[str, Any],
    random_seed: int,
    parent_version: str | None = None,
    notes: str = "",
) -> ModelVersion:
    """Allocate and construct metadata for a newly starting training run."""
    version_id, label = registry.allocate_version(parent_version)
    return ModelVersion(
        version_id=version_id,
        version_label=label,
        parent_version=registry.resolve(parent_version).version_id if parent_version else None,
        created_at=datetime.now(timezone.utc).isoformat(),
        language_pair=registry.pair,
        direction=registry.direction,
        experiment_id=experiment_id,
        model_configuration=model_configuration,
        training_configuration=training_configuration,
        tokenizer_version=tokenizer_version,
        dataset_versions=dataset_versions,
        environment=environment,
        random_seed=random_seed,
        notes=notes,
    )
