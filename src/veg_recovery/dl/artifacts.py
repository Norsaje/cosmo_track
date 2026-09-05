"""Локальные research checkpoints с хешами; не production candidate bundle."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import importlib.metadata

from .data import WindowPreprocessor


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_info() -> dict:
    import torch

    packages = {}
    for name in ("numpy", "pandas", "torch", "pypots", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    source_root = Path(__file__).resolve().parents[1]
    source_files = sorted(
        [
            *source_root.joinpath("dl").rglob("*.py"),
            *source_root.joinpath("anomalies").glob("*.py"),
        ]
    )
    source_hashes = {
        p.relative_to(source_root).as_posix(): file_sha256(p) for p in source_files
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_worktree_dirty": bool(status.stdout.strip())
        if status.returncode == 0
        else None,
        "source_sha256": source_hashes,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def save_research_checkpoint(
    path, model, preprocessor: WindowPreprocessor, metadata: dict
):
    import torch

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    weights = destination / "weights.pt"
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        weights,
    )
    prep = destination / "preprocessor.json"
    prep.write_text(
        json.dumps(preprocessor.to_dict(), sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "dl-research-0.1",
        "production_approved": False,
        "model": "residual_tcn",
        "model_config": asdict(model.config),
        "files": {
            "weights.pt": file_sha256(weights),
            "preprocessor.json": file_sha256(prep),
        },
        "preprocessor_fingerprint": preprocessor.fingerprint,
        "metadata": metadata,
        "environment": environment_info(),
    }
    (destination / "manifest.json").write_text(
        json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    return manifest


def load_research_checkpoint(path, *, device="cpu"):
    import torch
    from .models.tcn import ResidualTCN, TCNConfig

    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "dl-research-0.1"
        or manifest.get("model") != "residual_tcn"
    ):
        raise ValueError("Unsupported DL research checkpoint schema/model")
    if set(manifest["files"]) != {"weights.pt", "preprocessor.json"}:
        raise ValueError("Unexpected checkpoint files")
    for name, expected_hash in manifest["files"].items():
        if file_sha256(root / name) != expected_hash:
            raise ValueError(f"Checkpoint SHA256 mismatch: {name}")
    prep = WindowPreprocessor.from_dict(
        json.loads((root / "preprocessor.json").read_text(encoding="utf-8"))
    )
    if prep.fingerprint != manifest["preprocessor_fingerprint"]:
        raise ValueError("Preprocessor fingerprint mismatch")
    model = ResidualTCN(TCNConfig(**manifest["model_config"]))
    model.load_state_dict(
        torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    return model, prep, manifest
