"""Сервисный слой: оркестрация анализа.

Оркестратор появляется в BE-006. Он собирает `ReconstructionRequest` и вызывает
`NDVIReconstructor` из `veg_recovery.inference` (зона ML) — свою модель здесь
не реализуем никогда. Веб-путь всегда передаёт `context_mode="web"`.
"""
