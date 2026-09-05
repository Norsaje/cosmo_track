"""Собирает автономный Kaggle Dataset для полного C-03 прогона DL.

Внутрь кладутся три части с раздельным происхождением:

* `src_dl/`  — код ветки DL (владелец: DL);
* `src_ml/`  — код опубликованной ветки ML, read-only (владелец: ML);
* `inputs/`  — производный C-03 manifest и ключи (`veg_recovery.dl.c03_bridge`).

Ничего не скачивается из сети, скрытые test labels не попадают в архив:
он содержит только `train_dataset.csv`, уже опубликованный обеими ветками.

    python configs/dl/kaggle/package_dataset.py \
        --ml-root tmp/dl-review-ml --inputs artifacts/dl/c03_derived \
        --output artifacts/dl/kaggle_bundle
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SKIP = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
RUN_COMMAND = (
    "PYTHONPATH=src_dl:src_ml CUBLAS_WORKSPACE_CONFIG=:4096:8 "
    "python -m veg_recovery.dl.train --fold-manifest inputs/dl_c03.json "
    "--device cuda --seeds 17 42 73 --epochs 24 --epoch-policy fixed "
    "--base-mode anchored --window 61 --output /kaggle/working/dl_tcn_run"
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*SKIP),
        dirs_exist_ok=False,
    )


def _commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build(ml_root, inputs, output, *, repo=".", archive=True):
    repo, ml_root = Path(repo).resolve(), Path(ml_root).resolve()
    inputs, output = Path(inputs).resolve(), Path(output)
    manifest_path = inputs / "dl_c03.json"
    if not manifest_path.is_file():
        raise SystemExit(
            "Run veg_recovery.dl.c03_bridge first: the derived C-03 manifest is missing"
        )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _copy_tree(repo / "src", output / "src_dl")
    _copy_tree(ml_root / "src", output / "src_ml")
    _copy_tree(repo / "tests", output / "tests")
    _copy_tree(inputs, output / "inputs")
    shutil.copyfile(repo / "configs/dl/tcn.yaml", output / "tcn.yaml")
    shutil.copyfile(
        repo / "configs/dl/kaggle/run_tcn.ipynb", output / "run_tcn.ipynb"
    )
    files = sorted(p for p in output.rglob("*") if p.is_file())
    payload = {
        "kind": "dl_kaggle_bundle",
        "schema_version": "dl-kaggle-bundle-0.1",
        "dl_commit": _commit(repo),
        "ml_commit": _commit(ml_root),
        "c03_review_status": json.loads(
            manifest_path.read_text(encoding="utf-8")
        )["review_status"],
        "run_command": RUN_COMMAND,
        "python": sys.version.split()[0],
        "files": {
            p.relative_to(output).as_posix(): _sha256(p)
            for p in files
            if p.name != "BUNDLE_MANIFEST.json"
        },
    }
    (output / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {"directory": str(output), "files": len(payload["files"])}
    if archive:
        zip_path = output.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(p for p in output.rglob("*") if p.is_file()):
                bundle.write(path, path.relative_to(output).as_posix())
        result["archive"] = str(zip_path)
        result["archive_sha256"] = _sha256(zip_path)
        result["archive_mb"] = round(zip_path.stat().st_size / 2**20, 2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", required=True)
    parser.add_argument("--inputs", default="artifacts/dl/c03_derived")
    parser.add_argument("--output", default="artifacts/dl/kaggle_bundle")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(
                args.ml_root,
                args.inputs,
                args.output,
                repo=args.repo,
                archive=not args.no_archive,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
