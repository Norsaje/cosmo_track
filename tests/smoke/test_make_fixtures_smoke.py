"""Smoke-проверки scripts/make_test_fixtures.py.

Покрывает:
- CLI --help exit 0;
- end-to-end: запуск на data/*.csv → 5 файлов созданы;
- детерминизм: повторный запуск → побайтно одинаковые файлы;
- generated submission_for_test_tiny.csv валиден против test_tiny.csv
  через scripts.validate_submission;
- generated submission_invalid_duplicate.csv имеет дубликат и fail в
  validate_submission;
- manifest.json schema_version есть, без timestamp;
- absolute path → exit 2.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.make_test_fixtures import MANIFEST_SCHEMA_VERSION, SCRIPT_VERSION
from scripts.validate_submission import (
    extract_test_gap_keys,
)
from scripts.validate_submission import (
    main as vs_main,
)

ROOT = Path(__file__).resolve().parents[2]
TMP = "tests/smoke/_tmp"
TMP_OUT = f"{TMP}/make_fixtures_out"
TMP_MANIFEST = f"{TMP}/make_fixtures_manifest.json"


@pytest.fixture
def smoke_tmp(monkeypatch):
    monkeypatch.chdir(ROOT)
    (ROOT / TMP).mkdir(parents=True, exist_ok=True)
    (ROOT / TMP_OUT).mkdir(parents=True, exist_ok=True)
    yield ROOT / TMP


# ───────────────────────────── CLI --help ─────────────────────────────

class TestCliHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--out" in result.stdout
        assert "--manifest" in result.stdout
        assert "--train-source" in result.stdout
        assert "--test-source" in result.stdout


# ───────────────────────────── end-to-end ─────────────────────────────

class TestEndToEnd:
    def test_run_creates_all_files(self, smoke_tmp):
        subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures",
             "--out", TMP_OUT, "--manifest", TMP_MANIFEST],
            cwd=ROOT, check=True,
        )
        expected = {
            f"{TMP_OUT}/train_tiny.csv",
            f"{TMP_OUT}/test_tiny.csv",
            f"{TMP_OUT}/submission_for_test_tiny.csv",
            f"{TMP_OUT}/submission_invalid_duplicate.csv",
            TMP_MANIFEST,
        }
        for p in expected:
            assert (ROOT / p).exists(), f"missing file: {p}"

    def test_manifest_schema_version_and_no_timestamp(self, smoke_tmp):
        subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures",
             "--out", TMP_OUT, "--manifest", TMP_MANIFEST],
            cwd=ROOT, check=True,
        )
        payload = json.loads((ROOT / TMP_MANIFEST).read_text(encoding="utf-8"))
        assert payload["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
        assert payload["script_version"] == SCRIPT_VERSION
        assert "timestamp" not in payload
        assert "created_at" not in payload
        # design_assumptions + requires_ml_confirmation
        assert "design_assumptions" in payload
        assert payload["design_assumptions_requires_ml_confirmation"] is True
        # Source fingerprints заполнены.
        assert payload["source_fingerprints"]["train_dataset.csv"]["sha256"]
        assert payload["source_fingerprints"]["test_data.csv"]["sha256"]
        # Selected keys описаны.
        assert "selected_keys" in payload
        assert "known_polygon" in payload["selected_keys"]

    def test_determinism_byte_to_byte(self, smoke_tmp):
        # Run 1.
        subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures",
             "--out", TMP_OUT, "--manifest", TMP_MANIFEST],
            cwd=ROOT, check=True,
        )
        # Capture SHA1.
        import hashlib
        shas_run1 = {
            p: hashlib.sha1((ROOT / p).read_bytes()).hexdigest()
            for p in [
                f"{TMP_OUT}/train_tiny.csv",
                f"{TMP_OUT}/test_tiny.csv",
                f"{TMP_OUT}/submission_for_test_tiny.csv",
                f"{TMP_OUT}/submission_invalid_duplicate.csv",
                TMP_MANIFEST,
            ]
        }
        # Run 2.
        subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures",
             "--out", TMP_OUT, "--manifest", TMP_MANIFEST],
            cwd=ROOT, check=True,
        )
        shas_run2 = {
            p: hashlib.sha1((ROOT / p).read_bytes()).hexdigest()
            for p in shas_run1
        }
        assert shas_run1 == shas_run2

    def test_generated_submission_is_valid(self, smoke_tmp):
        """submission_for_test_tiny.csv должен проходить validate_submission
        против сгенерированного test_tiny.csv."""
        subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures",
             "--out", TMP_OUT, "--manifest", TMP_MANIFEST],
            cwd=ROOT, check=True,
        )
        test_tiny = pd.read_csv(ROOT / f"{TMP_OUT}/test_tiny.csv")
        gap_keys = extract_test_gap_keys(test_tiny)
        if not gap_keys:
            pytest.skip("test_tiny has no gaps; nothing to validate")

        sys.argv = [
            "validate_submission",
            "--test", f"{TMP_OUT}/test_tiny.csv",
            "--submission", f"{TMP_OUT}/submission_for_test_tiny.csv",
        ]
        assert vs_main() == 0

    def test_generated_duplicate_submission_fails(self, smoke_tmp):
        """submission_invalid_duplicate.csv должен падать на duplicate_key."""
        subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures",
             "--out", TMP_OUT, "--manifest", TMP_MANIFEST],
            cwd=ROOT, check=True,
        )
        test_tiny = pd.read_csv(ROOT / f"{TMP_OUT}/test_tiny.csv")
        gap_keys = extract_test_gap_keys(test_tiny)
        if not gap_keys:
            pytest.skip("test_tiny has no gaps; nothing to validate")

        sys.argv = [
            "validate_submission",
            "--test", f"{TMP_OUT}/test_tiny.csv",
            "--submission", f"{TMP_OUT}/submission_invalid_duplicate.csv",
        ]
        assert vs_main() == 1


# ───────────────────────────── exit codes ─────────────────────────────

class TestExitCodes:
    def test_absolute_out_path_returns_2(self, smoke_tmp):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures",
             "--out", "C:/absolute/out",
             "--manifest", TMP_MANIFEST],
            cwd=ROOT, check=False, capture_output=True, text=True,
        )
        assert result.returncode == 2

    def test_missing_train_source_returns_2(self, smoke_tmp):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures",
             "--out", TMP_OUT, "--manifest", TMP_MANIFEST,
             "--train-source", "nonexistent_train.csv"],
            cwd=ROOT, check=False, capture_output=True, text=True,
        )
        assert result.returncode == 2