"""Общий пакет проекта cosmo_track.

Один пакет на всю команду: batch-инференс и веб-сервис зовут одну и ту же
реализацию, дублирование feature/model-кода запрещено (решение D-002, инвариант 6).

Импорт этого модуля обязан работать без torch, GDAL и клиентов БД — на это
опираются offline smoke-тесты (SH-005). Тяжёлые зависимости подключаются
только внутри своих подпакетов.

Границы владения (§6 координации):
  contracts.py, inference.py, features/, models/, validation/, cli/,
  anomalies/baseline.py  — Разработчик 1 (ML)
  dl/, anomalies/advanced.py|events.py|explain.py — Разработчик 2 (DL)
  providers/, geospatial/, service/ — Разработчик 3 (Backend, мы)
"""

__version__ = "0.1.0"
