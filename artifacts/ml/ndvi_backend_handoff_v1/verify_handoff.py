#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files() -> int:
    checked = 0
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"missing file: {rel}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"SHA256 mismatch: {rel}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-inference", action="store_true")
    args = parser.parse_args()
    count = verify_files()

    from veg_recovery.inference import load_reconstructor
    model = load_reconstructor(ROOT / "bundle", trusted=True)
    print(f"integrity OK: {count} files; model={model.bundle.manifest.model_version}")

    if not args.full_inference:
        return

    import numpy as np
    import pandas as pd
    from veg_recovery.contracts import KEY_COLUMNS, ReconstructionRequest
    from veg_recovery.data import read_dataset

    frame = read_dataset(ROOT / "examples/test_data.csv", kind="test", strict_current=True)
    mask = frame["is_synthetic_gap"].astype(bool)
    result = model.predict(ReconstructionRequest(frame, mask, "competition"))
    expected = pd.read_csv(ROOT / "examples/expected_submission.csv", parse_dates=["date"])
    actual = result.predictions[KEY_COLUMNS + ["primary_ndvi_pred"]].reset_index(drop=True)
    if len(actual) != 3112 or not actual[KEY_COLUMNS].equals(expected[KEY_COLUMNS]):
        raise SystemExit("prediction keys/count differ from expected output")
    delta = np.max(np.abs(actual["primary_ndvi_pred"].to_numpy() - expected["primary_ndvi_pred"].to_numpy()))
    if not np.isfinite(delta) or delta > 1e-10:
        raise SystemExit(f"prediction mismatch: max_abs_delta={delta}")
    print(f"full inference OK: rows={len(actual)}; max_abs_delta={delta:.3g}")


if __name__ == "__main__":
    main()
