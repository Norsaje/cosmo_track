"""Адаптеры внешних источников данных.

Протоколы `FieldBoundaryProvider`, `OpticalProvider`, `WeatherProvider` и
нормализованный `ObservationFrame` (контракт C-13) появляются в BE-007.
Адаптеры не знают ни о FastAPI, ни о БД — ими управляет оркестратор.
"""
