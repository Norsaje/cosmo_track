"""Мост к публичному PyPOTS impute API без импорта библиотеки в core.

Stock SAITS.fit использует MCAR по отдельным ячейкам, несовместимый с C-03.
Этот модуль отвечает за безопасные массивы/выравнивание готовой модели.
Matched-mask training bridge и проверка конкретной версии остаются DL-004.
"""

from dataclasses import dataclass

import numpy as np

from ..data import WindowDatasetAdapter


@dataclass(frozen=True)
class PyPOTSArrays:
    X: np.ndarray
    X_ori: np.ndarray | None
    target_indices: np.ndarray


def to_pypots_arrays(
    dataset: WindowDatasetAdapter, *, include_validation_labels=False
) -> PyPOTSArrays:
    if not len(dataset):
        raise ValueError("PyPOTS requires a nonempty dataset")
    samples = [dataset[i] for i in range(len(dataset))]
    values = np.stack([s["values"] for s in samples]).copy()
    masks = np.stack([s["observation_mask"] for s in samples])
    valid = np.stack([s["valid_calendar_mask"] for s in samples])
    values[~masks | ~valid[..., None]] = np.nan
    original = None
    if include_validation_labels:
        if not np.isfinite(dataset.y).all():
            raise ValueError("Finite validation labels are required")
        original = values.copy()
        # Единственное различие X/X_ori — оцениваемый primary_ndvi, не все dynamic fields.
        original[:, dataset.center_index, 0] = (
            dataset.y - dataset.preprocessor.center[0]
        ) / dataset.preprocessor.scale[0]
    positions = np.full(len(dataset), dataset.center_index, dtype=np.int64)
    return PyPOTSArrays(values, original, positions)


class PyPOTSInferenceAdapter:
    """Принимает уже обученный SAITS/BRITS с impute({'X': ...})."""

    def __init__(self, fitted_model, *, model_name: str):
        if model_name not in {"saits", "brits"} or not callable(
            getattr(fitted_model, "impute", None)
        ):
            raise ValueError("Expected fitted SAITS/BRITS public impute API")
        self.model, self.model_name = fitted_model, model_name

    def predict(self, dataset: WindowDatasetAdapter):
        arrays = to_pypots_arrays(dataset)
        output = np.asarray(self.model.impute({"X": arrays.X.copy()}))
        if output.shape != arrays.X.shape:
            raise ValueError(
                "PyPOTS output shape changed; refuse positional misalignment"
            )
        target = output[np.arange(len(dataset)), arrays.target_indices, 0]
        target = target * dataset.preprocessor.scale[0] + dataset.preprocessor.center[0]
        return dataset.predictions_frame(target)
