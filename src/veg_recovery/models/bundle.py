"""Checked JSON baseline artifacts and explicitly trusted estimator artifacts.

SHA256 detects corruption, not authenticity. Only load trained bundles produced
by your own controlled training job; a joblib file can execute Python code.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .manifest import ModelManifest, ModelFile, SCHEMA_VERSION, git_commit, package_versions, sha256_file


@dataclass(frozen=True)
class LoadedBundle:
    manifest: ModelManifest
    state: Any
    config: dict[str, Any]
    estimators: dict[str, Any] | None = None


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _save_bundle(path, state, config, metadata, estimators=None) -> ModelManifest:
    from veg_recovery.features.builder import FEATURE_VERSION
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {root}")
    _json_write(root / "feature_state.json", state.to_dict() if hasattr(state, "to_dict") else state)
    _json_write(root / "config.json", config)
    files = [ModelFile(path=name, sha256=sha256_file(root / name), format="json")
             for name in ("feature_state.json", "config.json")]
    if estimators is not None:
        import joblib
        joblib.dump(estimators, root / "estimators.joblib", compress=3)
        files.append(ModelFile(path="estimators.joblib", sha256=sha256_file(root / "estimators.joblib"), format="joblib"))
    values = {"git_commit": git_commit(), "train_fingerprints": {}, "feature_version": FEATURE_VERSION,
              "seeds": [], "cv_summary": {}, "package_versions": package_versions(), **(metadata or {})}
    values.update(bundle_kind="trained" if estimators is not None else "baseline", model_files=files)
    manifest = ModelManifest.model_validate(values)
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def save_baseline_bundle(path, state, config=None, metadata=None) -> ModelManifest:
    return _save_bundle(path, state, config or {"method": "mean_neighbors", "clip": None}, metadata)


def save_trained_bundle(path, state, config, estimators, metadata=None) -> ModelManifest:
    return _save_bundle(path, state, config, metadata, estimators=estimators)


def load_bundle(path, trusted: bool = False) -> LoadedBundle:
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Manifest symlinks are forbidden")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Incompatible schema_version: {payload.get('schema_version')!r}")
    manifest = ModelManifest.model_validate(payload)
    files = {item.path: item for item in manifest.model_files}
    required = {"feature_state.json", "config.json"}
    if manifest.bundle_kind == "trained":
        required.add("estimators.joblib")
    if set(files) != required:
        raise ValueError("Manifest file set does not match bundle kind")
    if any(files[name].format != ("joblib" if name.endswith(".joblib") else "json") for name in required):
        raise ValueError("Model file format does not match its role")
    # Verify every byte first, before constructing state or invoking joblib.
    for item in manifest.model_files:
        candidate = root / item.path
        if candidate.is_symlink() or candidate.resolve().parent != root or not candidate.is_file():
            raise ValueError(f"Invalid bundle file: {item.path}")
        if sha256_file(candidate) != item.sha256:
            raise ValueError(f"SHA256 mismatch: {item.path}")
    from veg_recovery.features.builder import FEATURE_VERSION, FeatureState
    if manifest.feature_version != FEATURE_VERSION:
        raise ValueError(f"Incompatible feature_version: {manifest.feature_version}")
    if manifest.bundle_kind == "trained" and not trusted:
        raise ValueError("Trained bundle requires trusted=True after provenance review")
    state = FeatureState.from_dict(json.loads((root / "feature_state.json").read_text(encoding="utf-8")))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    estimators = None
    if manifest.bundle_kind == "trained":
        import joblib
        estimators = joblib.load(root / "estimators.joblib")
    return LoadedBundle(manifest=manifest, state=state, config=config, estimators=estimators)
