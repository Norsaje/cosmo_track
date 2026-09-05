"""Сервисный слой: оркестрация анализа.

`reconstructor` — единая точка доступа к предсказателю C-02 (BE-003): загрузка бандла
Разработчика 1 на старте процесса, классификация отказов и заглушка `ModelStub` на
время, пока бандла нет. Своей модели здесь не будет никогда (инвариант 6, решение D-002).

Оркестратор анализа появляется в BE-006. Он собирает `ReconstructionRequest` и зовёт
`NDVIReconstructor`; веб-путь всегда передаёт `context_mode="web"`.
"""

from veg_recovery.service.reconstructor import (
    CONTRACT_ID,
    CONTRACT_SCHEMA_VERSION,
    MODEL_ERROR_CODE,
    STUB_MODEL_VERSION,
    LoadedReconstructor,
    ModelContractMissing,
    ModelStub,
    ModelUnavailable,
    build_reconstructor,
    classify_bundle_failure,
    contract_is_available,
)

__all__ = [
    "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION",
    "MODEL_ERROR_CODE",
    "STUB_MODEL_VERSION",
    "LoadedReconstructor",
    "ModelContractMissing",
    "ModelStub",
    "ModelUnavailable",
    "build_reconstructor",
    "classify_bundle_failure",
    "contract_is_available",
]
