"""Синтетический producer fixture проверяет DL consumer, не утверждает C-03 готовым."""

import json
from pathlib import Path
import subprocess
import sys
import types

import pandas as pd
import pytest

from veg_recovery.dl.artifacts import file_sha256
from veg_recovery.dl.data import KEY, prepare_context
from veg_recovery.dl.fixtures import training_fixture
from veg_recovery.dl.train import main, preflight


@pytest.fixture
def consumer_manifest(tmp_path, monkeypatch):
    frame = training_fixture()
    observed = frame.loc[frame.primary_ndvi.notna()].reset_index(drop=True)
    outer = observed.iloc[[10, 30, 55, 65]][KEY]
    inner = observed.iloc[[8, 28, 52, 69]][KEY]
    holdout = pd.concat([outer, inner]).assign(_heldout=True)
    fit = frame[KEY].merge(holdout, on=KEY, how="left")
    fit = fit.loc[fit._heldout.isna(), KEY]
    train = observed.iloc[[2, 5, 15, 20, 50, 58, 70, 80]][KEY]
    baseline = outer.merge(frame[KEY + ["primary_ndvi"]], on=KEY).rename(
        columns={"primary_ndvi": "y_true"}
    )
    baseline = baseline.assign(
        split="matched", fold="f0", primary_ndvi_pred=lambda f: f.y_true + 0.1
    )

    def write(name, data):
        path = tmp_path / (name + ".csv")
        data.to_csv(path, index=False)
        return {"path": path.name, "sha256": file_sha256(path)}

    fold = {
        "split": "matched",
        "fold": "f0",
        "context_policy": "interpolation",
        "fit_context_keys": write("fit", fit),
        "inference_context_keys": write("infer", frame[KEY]),
        "censored_context_keys": write("censored", outer),
        "train_target_keys": write("train", train.assign(block=[0, 1] * 4)),
        "inner_target_keys": write("inner", inner),
        "evaluation_keys": write("outer", outer),
    }
    manifest = {
        "schema_version": "dl-c03-consumer-0.2",
        "producer": "ML",
        "review_status": "accepted",
        "fold_version": "TEST_FIXTURE_ONLY",
        "mask_version": "TEST_FIXTURE_ONLY",
        "mask_callable": "veg_recovery.validation.fixture:apply_mask",
        "data": write("data", frame),
        "baseline_oof": write("baseline", baseline),
        "baseline_metrics": {"overall_rmse": 0.1, "gap_score": 0},
        "folds": [fold],
    }
    module = types.ModuleType("veg_recovery.validation.fixture")
    module.apply_mask = lambda f, keys: prepare_context(f, keys).frame
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / "consumer.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_manifest_preflight_uses_immutable_ml_keys(consumer_manifest):
    path, _ = consumer_manifest
    manifest, frame, baseline, folds, _ = preflight(path)
    assert manifest["fold_version"] == "TEST_FIXTURE_ONLY"
    assert len(folds) == 1 and len(frame) == 122 and len(baseline) == 4


@pytest.mark.parametrize(
    "problem",
    [
        "draft",
        "hash",
        "metrics",
        "overlap",
        "missing_mask",
        "uncensored_evaluation",
        "censored_fit",
        "derived_without_evidence",
        "unknown_schema",
    ],
)
def test_bad_handoff_fails_before_training(consumer_manifest, problem):
    path, manifest = consumer_manifest
    fold = manifest["folds"][0]
    if problem == "draft":
        manifest["review_status"] = "draft"
    elif problem == "hash":
        manifest["data"]["sha256"] = "0" * 64
    elif problem == "metrics":
        manifest["baseline_metrics"]["overall_rmse"] = 0.06
    elif problem == "overlap":
        fold["inner_target_keys"] = fold["train_target_keys"]
    elif problem == "uncensored_evaluation":
        # Без producer censoring inference увидел бы скрытые ML значения.
        fold["censored_context_keys"] = fold["inner_target_keys"]
    elif problem == "censored_fit":
        fold["censored_context_keys"] = fold["fit_context_keys"]
    elif problem == "derived_without_evidence":
        manifest["review_status"] = "derived_from_producer_artifacts"
    elif problem == "unknown_schema":
        manifest["schema_version"] = "dl-c03-consumer-9.9"
    else:
        manifest["mask_callable"] = "veg_recovery.validation.missing:apply_mask"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        preflight(path)


def test_derived_manifest_needs_producer_evidence(consumer_manifest):
    path, manifest = consumer_manifest
    manifest["review_status"] = "derived_from_producer_artifacts"
    manifest["producer_evidence"] = {
        "commit": "0" * 40,
        "input_sha256": {"train": "1" * 64},
        "ml_acknowledgement": "pending",
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result, _, _, folds, _ = preflight(path)
    assert result["producer_evidence"]["ml_acknowledgement"] == "pending"
    assert len(folds) == 1


def test_causal_fold_rejects_uncensored_future_observation(consumer_manifest):
    path, manifest = consumer_manifest
    # Тот же контекст, но объявленный causal: будущие наблюдения не зацензурированы.
    manifest["folds"][0]["context_policy"] = "causal"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="uncensored future"):
        preflight(path)


def test_fixture_cv_runner_and_research_outputs(consumer_manifest, tmp_path):
    pytest.importorskip("torch")
    path, _ = consumer_manifest
    output = tmp_path / "run"
    code = main(
        [
            "--fold-manifest",
            str(path),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--seeds",
            "17",
            "--window",
            "15",
            "--hidden-size",
            "16",
            "--layers",
            "2",
        ]
    )
    assert code == 0
    report = json.loads((output / "cv_report.json").read_text())
    assert report["decision"] == "PENDING_REVIEW"
    assert report["seeds"][0]["dl"]["composite_rmse"] is None
    assert report["seeds"][0]["complete_cv"] is True
    assert report["epoch_policy"] == "fixed"
    oof = pd.read_csv(output / "oof_seed_17.csv")
    assert len(oof) == 4 and "base_pred" in oof
    assert len(pd.read_csv(output / "experiments.csv")) == 1
    manifest = json.loads((output / "seed_17_fold_0/manifest.json").read_text())
    assert manifest["production_approved"] is False


def test_missing_c03_cli_exits_without_torch_or_output(tmp_path):
    code = """
import sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('Preflight imported torch')
sys.meta_path.insert(0, NoTorch())
from veg_recovery.dl.train import main
raise SystemExit(main(['--fold-manifest', 'MISSING_C03.json', '--preflight-only']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "C-03" in result.stderr
    assert "Preflight imported torch" not in result.stderr


def test_kaggle_notebook_is_valid_python():
    # В репозитории и внутри собранного Kaggle bundle notebook лежит по-разному.
    candidates = [Path("configs/dl/kaggle/run_tcn.ipynb"), Path("run_tcn.ipynb")]
    found = next((p for p in candidates if p.is_file()), None)
    assert found is not None, "run_tcn.ipynb not found in repo or bundle layout"
    notebook = json.loads(found.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "kaggle_cell", "exec")
