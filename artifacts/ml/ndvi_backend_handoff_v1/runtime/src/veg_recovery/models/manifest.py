"""Versioned, strict metadata for locally trusted model bundles."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"


class ModelFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["json", "joblib"]

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or not value:
            raise ValueError("Bundle paths must be safe relative paths")
        if path.name == "manifest.json":
            raise ValueError("Manifest cannot recursively include itself")
        return value


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    model_version: str = "baseline-v1"
    bundle_kind: Literal["baseline", "trained"] = "baseline"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    git_commit: str
    train_fingerprints: dict[str, str]
    feature_version: str
    model_files: list[ModelFile]
    seeds: list[int]
    cv_summary: dict[str, Any]
    package_versions: dict[str, str]

    @field_validator("created_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value

    @field_validator("model_files")
    @classmethod
    def unique_paths(cls, value: list[ModelFile]) -> list[ModelFile]:
        if len({item.path for item in value}) != len(value):
            raise ValueError("Duplicate model file paths")
        return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("numpy", "pandas", "pydantic", "scikit-learn", "scipy", "joblib", "catboost", "lightgbm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
