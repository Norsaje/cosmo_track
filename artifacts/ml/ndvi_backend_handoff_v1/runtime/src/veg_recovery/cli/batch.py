"""Strict batch inference: python -m veg_recovery.cli.batch --help."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from veg_recovery.contracts import KEY_COLUMNS, SUBMISSION_COLUMNS, ReconstructionRequest


def validate_submission(table: pd.DataFrame, expected_keys: pd.DataFrame) -> None:
    if list(table.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be exactly {SUBMISSION_COLUMNS}")
    if table[KEY_COLUMNS].isna().any().any() or table.duplicated(KEY_COLUMNS).any():
        raise ValueError("null or duplicate submission key")
    dates = table["date"].astype(str)
    if not dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("date must be YYYY-MM-DD")
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    if not pd.api.types.is_numeric_dtype(table["primary_ndvi_pred"]):
        raise ValueError("prediction column must be numeric")
    if pd.api.types.is_bool_dtype(table["primary_ndvi_pred"]):
        raise ValueError("boolean prediction is not a real-valued prediction")
    if not np.isfinite(table["primary_ndvi_pred"].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("prediction contains NaN/inf")
    actual = pd.DataFrame({"anon_polygon_id": table["anon_polygon_id"].astype(str).to_numpy(), "date": parsed.to_numpy()})
    expected = expected_keys[KEY_COLUMNS].copy().reset_index(drop=True)
    expected["anon_polygon_id"] = expected["anon_polygon_id"].astype(str)
    expected["date"] = pd.to_datetime(expected["date"])
    if expected.duplicated(KEY_COLUMNS).any() or not actual.equals(expected):
        raise ValueError("submission must exactly match the ordered synthetic-gap keys")


def validate_submission_file(path: str | Path, expected_keys: pd.DataFrame) -> pd.DataFrame:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UTF-8 BOM is not accepted; write plain UTF-8")
    decoded = raw.decode("utf-8", errors="strict")
    rows = list(csv.reader(io.StringIO(decoded), delimiter=",", strict=True))
    if not rows or rows[0] != SUBMISSION_COLUMNS or any(len(row) != 3 for row in rows[1:]):
        raise ValueError("invalid CSV header or row width")
    table = pd.read_csv(io.StringIO(decoded), dtype={"anon_polygon_id": "string", "date": "string"})
    # Empty output has no values from which pandas can infer a numeric dtype.
    if table.empty: table["primary_ndvi_pred"] = pd.Series(dtype=float)
    validate_submission(table, expected_keys)
    return table


def write_submission(predictions: pd.DataFrame, expected_keys: pd.DataFrame, output: str | Path) -> None:
    table = predictions[SUBMISSION_COLUMNS].copy()
    table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
    validate_submission(table, expected_keys)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".csv", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            table.to_csv(stream, index=False, float_format="%.17g", lineterminator="\n")
        validate_submission_file(temporary, expected_keys)
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--trusted-bundle", action="store_true", help="Allow verified joblib from your own training job")
    parser.add_argument("--expected-count", type=int, default=None, help="Use 3112 for the current competition test")
    args = parser.parse_args(argv)
    try:
        from veg_recovery.data import read_dataset
        from veg_recovery.inference import load_reconstructor
        inputs, output, diagnostic = Path(args.input).resolve(), Path(args.output).resolve(), Path(args.diagnostics).resolve()
        if len({inputs, output, diagnostic}) != 3:
            raise ValueError("input, submission and diagnostics paths must be distinct")
        bundle = Path(args.bundle).resolve()
        if any(p == bundle or bundle in p.parents for p in [output, diagnostic]):
            raise ValueError("outputs may not overwrite the model bundle")
        frame = read_dataset(inputs, kind="test")
        mask = frame["is_synthetic_gap"].astype(bool)
        if args.expected_count is not None and int(mask.sum()) != args.expected_count:
            raise ValueError(f"expected {args.expected_count} gaps, found {int(mask.sum())}")
        model = load_reconstructor(bundle, trusted=args.trusted_bundle)
        result = model.predict(ReconstructionRequest(frame, mask, "competition"))
        write_submission(result.predictions, frame.loc[mask, KEY_COLUMNS], output)
        diagnostics = result.diagnostics.copy()
        diagnostics["quality_flags"] = diagnostics["quality_flags"].map(json.dumps)
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(diagnostic, index=False, encoding="utf-8")
        print(json.dumps(dict(rows=len(result.predictions), model_version=result.model_version,
                              submission=str(output), diagnostics=str(diagnostic))))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"batch inference failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
