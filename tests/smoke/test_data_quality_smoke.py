"""Smoke-проверки scripts/data_quality_report.py.

Покрывает:
- CLI --help exit 0;
- integration на реальных CSV (data/train_dataset.csv, data/test_data.csv)
  с проверкой основных counts и отсутствия warnings;
- детерминизм JSON: повторный запуск → побайтно одинаковый;
- exit code 2 на абсолютный путь;
- exit code 2 на отсутствующий файл;
- exit code 1 при mismatch, exit 0 с --allow-mismatch.

Эти тесты НЕ требуют subprocess-вызовов data_quality_report.py —
используют main() напрямую через monkeypatch sys.argv. Но есть один
тест с реальным subprocess, чтобы поймать регрессии в CLI.

Пути: train/test передаются ОТНОСИТЕЛЬНО корня проекта (как требует
ed_part.md). out-md/out-json пишутся в tests/smoke/_tmp/ (gitignored),
тоже относительными путями. Корневая директория проекта вычисляется
через ROOT ниже и подразумевается cwd для всех вызовов.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.data_quality_report import main as dq_main

ROOT = Path(__file__).resolve().parents[2]
TMP = "tests/smoke/_tmp"
TRAIN = "data/train_dataset.csv"
TEST = "data/test_data.csv"


# ───────────────────────────── CLI --help ─────────────────────────────

class TestCliHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.data_quality_report", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--train" in result.stdout
        assert "--test" in result.stdout
        assert "--out-md" in result.stdout
        assert "--out-json" in result.stdout


# ───────────────────────────── integration on real CSVs ─────────────────────────────

@pytest.fixture
def smoke_tmp(monkeypatch):
    """Создаёт tests/smoke/_tmp, переходит в ROOT (корень проекта),
    чистит после теста.
    """
    monkeypatch.chdir(ROOT)
    tmp_path = ROOT / TMP
    tmp_path.mkdir(parents=True, exist_ok=True)
    yield tmp_path
    # Не удаляем файлы — они могут быть полезны для дебага; .gitignore их
    # скрывает от репозитория. Если нужно — добавить rmtree.


class TestIntegrationRealCsv:
    def test_main_on_real_data_exits_zero(self, smoke_tmp):
        out_md = f"{TMP}/out.md"
        out_json = f"{TMP}/out.json"
        sys.argv = [
            "data_quality_report",
            "--train", TRAIN,
            "--test", TEST,
            "--out-md", out_md,
            "--out-json", out_json,
        ]
        assert dq_main() == 0

        payload = json.loads((smoke_tmp / "out.json").read_text(encoding="utf-8"))
        assert payload["report_schema_version"] == "1.0.0"
        assert payload["train"]["rows"] == 99_955
        assert payload["train"]["columns"] == 21
        assert payload["train"]["polygon_count"] == 39
        assert payload["train"]["primary_ndvi_finite"] == 30_520
        assert payload["train"]["warnings"] == []
        assert payload["test"]["rows"] == 57_185
        assert payload["test"]["columns"] == 20
        assert payload["test"]["polygon_count"] == 78
        assert payload["test"]["primary_ndvi_finite"] == 17_641
        assert payload["test"]["is_synthetic_gap"]["true"] == 3_112
        assert payload["test"]["warnings"] == []

        md = (smoke_tmp / "out.md").read_text(encoding="utf-8")
        for section in (
            "# Data Quality Report",
            "## 1. Summary",
            "## 2. Schema",
            "## 3. Missingness",
            "## 4. Gaps",
            "## 5. Polygon overlap",
            "## 6. Target-source hierarchy",
            "## 7. Warnings",
        ):
            assert section in md, f"missing section in MD: {section}"
        assert "No warnings" in md

    def test_determinism_byte_to_byte_on_real_csv(self, smoke_tmp):
        sys.argv = [
            "data_quality_report",
            "--train", TRAIN,
            "--test", TEST,
            "--out-md", f"{TMP}/a.md",
            "--out-json", f"{TMP}/a.json",
        ]
        assert dq_main() == 0
        sys.argv = [
            "data_quality_report",
            "--train", TRAIN,
            "--test", TEST,
            "--out-md", f"{TMP}/b.md",
            "--out-json", f"{TMP}/b.json",
        ]
        assert dq_main() == 0

        a_md = (smoke_tmp / "a.md").read_bytes()
        b_md = (smoke_tmp / "b.md").read_bytes()
        a_json = (smoke_tmp / "a.json").read_bytes()
        b_json = (smoke_tmp / "b.json").read_bytes()
        assert a_md == b_md
        assert a_json == b_json

        payload = json.loads((smoke_tmp / "a.json").read_text(encoding="utf-8"))
        assert "timestamp" not in payload
        assert "created_at" not in payload


# ───────────────────────────── exit codes ─────────────────────────────

class TestExitCodes:
    def test_missing_train_file_returns_2(self, smoke_tmp):
        sys.argv = [
            "data_quality_report",
            "--train", "nonexistent_train.csv",
            "--test", TEST,
            "--out-md", f"{TMP}/out.md",
            "--out-json", f"{TMP}/out.json",
        ]
        rc = dq_main()
        assert rc == 2

    def test_absolute_train_path_returns_2(self, smoke_tmp):
        sys.argv = [
            "data_quality_report",
            "--train", "C:/absolute/path.csv",
            "--test", TEST,
            "--out-md", f"{TMP}/out.md",
            "--out-json", f"{TMP}/out.json",
        ]
        with pytest.raises(SystemExit) as exc_info:
            dq_main()
        assert exc_info.value.code == 2

    def test_allow_mismatch_does_not_fail(self, smoke_tmp):
        (smoke_tmp / "bad_train.csv").write_text(
            "anon_polygon_id,date,primary_ndvi\nX,2020-04-01,0.5\n",
            encoding="utf-8",
        )
        (smoke_tmp / "bad_test.csv").write_text(
            "anon_polygon_id,date,primary_ndvi,is_synthetic_gap\n"
            "Y,2020-04-01,0.5,False\n",
            encoding="utf-8",
        )
        sys.argv = [
            "data_quality_report",
            "--train", f"{TMP}/bad_train.csv",
            "--test", f"{TMP}/bad_test.csv",
            "--out-md", f"{TMP}/out.md",
            "--out-json", f"{TMP}/out.json",
            "--allow-mismatch",
        ]
        assert dq_main() == 0

    def test_no_allow_mismatch_returns_1(self, smoke_tmp):
        (smoke_tmp / "bad_train.csv").write_text(
            "anon_polygon_id,date,primary_ndvi\nX,2020-04-01,0.5\n",
            encoding="utf-8",
        )
        (smoke_tmp / "bad_test.csv").write_text(
            "anon_polygon_id,date,primary_ndvi,is_synthetic_gap\n"
            "Y,2020-04-01,0.5,False\n",
            encoding="utf-8",
        )
        sys.argv = [
            "data_quality_report",
            "--train", f"{TMP}/bad_train.csv",
            "--test", f"{TMP}/bad_test.csv",
            "--out-md", f"{TMP}/out.md",
            "--out-json", f"{TMP}/out.json",
        ]
        assert dq_main() == 1