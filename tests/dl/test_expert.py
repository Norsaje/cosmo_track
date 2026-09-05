"""DL-эксперт за интерфейсом C-02: контракт соблюдён, интервалы не выдуманы."""

from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from veg_recovery.dl.data import KEY, canonical_keys
from veg_recovery.dl.expert import METHOD, ResidualTCNExpert
from veg_recovery.dl.fixtures import training_fixture

CHECKPOINT = "artifacts/dl/cpu_smoke_v3/seed_17"


@dataclass(frozen=True)
class _Request:
    frame: pd.DataFrame
    gap_mask: pd.Series
    context_mode: str = "competition"


@pytest.fixture
def contracts(monkeypatch):
    """Минимальная заглушка C-02: тесты DL не тянут код ML из другого worktree."""
    module = types.ModuleType("veg_recovery.contracts")

    @dataclass(frozen=True)
    class ReconstructionResult:
        predictions: pd.DataFrame
        diagnostics: pd.DataFrame
        model_version: str

    def validate_request(request):
        if request.context_mode not in ("competition", "web"):
            raise ValueError("context_mode must be competition or web")
        if not request.gap_mask.index.equals(request.frame.index):
            raise ValueError("gap_mask must be a Series with the exact frame index")
        return request.frame.copy(deep=True), request.gap_mask

    module.ReconstructionResult = ReconstructionResult
    module.validate_request = validate_request
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def expert():
    pytest.importorskip("torch")
    if not Path(CHECKPOINT).is_dir():
        pytest.skip(f"Нет offline smoke checkpoint: {CHECKPOINT}")
    frame = training_fixture()
    reference = frame.loc[frame.primary_ndvi.notna()]
    return ResidualTCNExpert(
        checkpoint=CHECKPOINT,
        reference_frame=frame,
        reference_keys=canonical_keys(reference),
    )


def _request(frame, positions):
    mask = pd.Series(False, index=frame.index)
    mask.iloc[positions] = True
    return _Request(frame=frame, gap_mask=mask)


def test_expert_returns_c02_shaped_result(contracts, expert):
    frame = training_fixture()
    observed = np.flatnonzero(frame.primary_ndvi.notna().to_numpy())[[3, 11, 25]]
    hidden = frame.copy()
    hidden.loc[hidden.index[observed], "primary_ndvi"] = np.nan
    result = expert.predict(_request(hidden, observed))
    assert result.model_version.startswith("research_residual_tcn_")
    assert len(result.predictions) == 3 and len(result.diagnostics) == 3
    required = set(KEY) | {
        "primary_ndvi_pred", "lower", "upper", "method",
        "primary_ndvi_reconstructed", "ndvi_harmonized",
    }
    assert required.issubset(result.predictions.columns)
    assert (result.predictions.method == METHOD).all()
    assert np.isfinite(result.predictions.primary_ndvi_pred).all()


def test_uncalibrated_interval_is_declared_not_invented(contracts, expert):
    frame = training_fixture()
    observed = np.flatnonzero(frame.primary_ndvi.notna().to_numpy())[[5, 20]]
    hidden = frame.copy()
    hidden.loc[hidden.index[observed], "primary_ndvi"] = np.nan
    result = expert.predict(_request(hidden, observed))
    predictions, diagnostics = result.predictions, result.diagnostics
    assert (predictions.lower == predictions.primary_ndvi_pred).all()
    assert (predictions.upper == predictions.primary_ndvi_pred).all()
    assert (diagnostics.interval_status == "unavailable_uncalibrated").all()
    assert (diagnostics.interval_level == 0.0).all()
    # Источник DL не классифицирует и не притворяется, что классифицирует.
    assert (diagnostics.p_unknown == 1.0).all()
    assert (diagnostics[["p_s2", "p_landsat", "p_modis"]] == 0.0).all().all()
    assert (diagnostics.harmonization_status == "not_applied").all()
    assert diagnostics.quality_flags.map(lambda f: "research_model" in f).all()
    assert (diagnostics.model_disagreement >= 0).all()
    keys = canonical_keys(diagnostics)
    assert keys.equals(canonical_keys(predictions.assign(date=predictions.date)))


def test_expert_rejects_empty_gap_mask(contracts, expert):
    frame = training_fixture()
    with pytest.raises(ValueError, match="Empty gap mask"):
        expert.predict(_Request(frame=frame, gap_mask=pd.Series(False, index=frame.index)))


def test_expert_requires_shared_contracts(expert, monkeypatch):
    monkeypatch.setitem(sys.modules, "veg_recovery.contracts", None)
    frame = training_fixture()
    observed = np.flatnonzero(frame.primary_ndvi.notna().to_numpy())[[3]]
    with pytest.raises(RuntimeError, match="C-02"):
        expert.predict(_request(frame, observed))
