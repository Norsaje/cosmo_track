"""DL-эксперт за общим интерфейсом ML `NDVIReconstructor`. Research, не production.

Класс реализует `predict(ReconstructionRequest) -> ReconstructionResult`, поэтому
Backend и ML-ансамбль могут вызвать его тем же контрактом, что и основную модель.
Регистрация в production запрещена до прохождения adoption gate: `model_version`
всегда начинается с `research_`.

Сезонный prior — статистика обучающего фолда, как `feature_state` у ML, поэтому
он строится из **явного reference-контекста**, переданного вызывающей стороной,
а не из строк, которые сейчас восстанавливаются.

Интервалы не калиброваны: `lower`/`upper` равны точечному прогнозу, а в
диагностике стоит `interval_status="unavailable_uncalibrated"`. Ширина не
выдумывается. Источник наблюдения DL не классифицирует: `p_unknown = 1`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .artifacts import load_research_checkpoint
from .data import KEY, SeasonalPrior, WindowDatasetAdapter, canonical_keys, prepare_context

METHOD = "research_residual_tcn"


def _contracts():
    try:
        from veg_recovery.contracts import ReconstructionResult, validate_request
    except ImportError as exc:  # pragma: no cover - зависит от наличия кода ML
        raise RuntimeError(
            "veg_recovery.contracts (C-02) недоступен: добавьте код ML в PYTHONPATH"
        ) from exc
    return ReconstructionResult, validate_request


@dataclass
class ResidualTCNExpert:
    """Один research checkpoint плюс reference-контекст обучающего фолда."""

    checkpoint: Path
    reference_frame: pd.DataFrame
    reference_keys: pd.DataFrame
    device: str = "cpu"
    base_mode: str = "anchored"
    apply_mask: object = None
    _model: object = field(default=None, init=False, repr=False)
    _prep: object = field(default=None, init=False, repr=False)
    _prior: SeasonalPrior = field(default=None, init=False, repr=False)
    _manifest: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._model, self._prep, self._manifest = load_research_checkpoint(
            self.checkpoint, device=self.device
        )
        if self._manifest.get("production_approved"):
            raise ValueError("Research checkpoint must not claim production approval")
        reference = prepare_context(
            self.reference_frame,
            self.reference_keys.iloc[:0],
            apply_mask=self.apply_mask,
        )
        self._prior = SeasonalPrior.fit(reference, self.reference_keys)

    @property
    def window_days(self) -> int:
        return int(self._manifest["metadata"]["window"])

    @property
    def model_version(self) -> str:
        digest = self._manifest["files"]["weights.pt"][:12]
        return f"research_residual_tcn_{digest}"

    def predict(self, request):
        from .training import predict_dataset

        ReconstructionResult, validate_request = _contracts()
        frame, mask = validate_request(request)
        if not mask.any():
            raise ValueError("Empty gap mask: nothing to reconstruct")
        keys = canonical_keys(frame.loc[mask.to_numpy()])
        context = prepare_context(frame, keys, apply_mask=self.apply_mask)
        dataset = WindowDatasetAdapter(
            context,
            keys,
            self._prep,
            window_days=self.window_days,
            prior=self._prior,
            base_mode=self.base_mode,
        )
        values, _ = predict_dataset(self._model, dataset, device=self.device)
        predictions = self._predictions(values)
        return ReconstructionResult(
            predictions=predictions,
            diagnostics=self._diagnostics(dataset, predictions),
            model_version=self.model_version,
        )

    def _predictions(self, values: pd.DataFrame) -> pd.DataFrame:
        out = values.copy()
        out["date"] = out.date.dt.date
        point = out.primary_ndvi_pred.to_numpy(float)
        # Интервал не калиброван, поэтому границы совпадают с точкой,
        # а статус в диагностике говорит об этом явно.
        out["lower"] = point
        out["upper"] = point
        out["method"] = METHOD
        out["primary_ndvi_reconstructed"] = point
        # Гармонизация здесь не применяется: C-09 получает отдельный ряд.
        out["ndvi_harmonized"] = point
        return out

    def _diagnostics(self, dataset, predictions: pd.DataFrame) -> pd.DataFrame:
        center = dataset.center_index
        rows = []
        for i in range(len(dataset)):
            item = dataset[i]
            valid = item["valid_calendar_mask"]
            observed = item["observation_mask"][:, 0] & valid
            left = float(item["delta_since"][center, 0])
            right = float(item["delta_until"][center, 0])
            base = float(item["base"][center])
            no_context = not observed.any()
            rows.append(
                {
                    "p_s2": 0.0,
                    "p_landsat": 0.0,
                    "p_modis": 0.0,
                    "p_unknown": 1.0,
                    "left_distance_days": None if left >= 366 else left,
                    "right_distance_days": None if right >= 366 else right,
                    # Расхождение сети и её собственной residual base.
                    "model_disagreement": abs(
                        float(predictions.primary_ndvi_pred.iloc[i]) - base
                    ),
                    "fallback_reason": "seasonal_prior_only" if no_context else "",
                    "context_quality": float(observed.sum() / max(int(valid.sum()), 1)),
                    "source_confidence": 0.0,
                    "quality_flags": ["research_model", "uncalibrated_interval"]
                    + (["no_visible_observation_in_polygon_year"] if no_context else []),
                    "interval_status": "unavailable_uncalibrated",
                    "interval_level": 0.0,
                    "harmonization_status": "not_applied",
                }
            )
        diagnostics = pd.DataFrame(rows)
        for column in KEY:
            diagnostics.insert(
                KEY.index(column), column, predictions[column].to_numpy()
            )
        if len(diagnostics) != len(predictions):
            raise ValueError("Diagnostics must align one to one with predictions")
        return diagnostics


def prior_from_fold(frame: pd.DataFrame, fit_keys: pd.DataFrame, *, apply_mask=None):
    """Утилита для тестов и ансамбля: prior строго из разрешённого fit-контекста."""
    context = prepare_context(frame, canonical_keys(fit_keys).iloc[:0], apply_mask=apply_mask)
    return SeasonalPrior.fit(context, fit_keys)


__all__ = ["ResidualTCNExpert", "prior_from_fold", "METHOD"]
