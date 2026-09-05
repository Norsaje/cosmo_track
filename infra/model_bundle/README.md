# Точка монтирования model bundle

Каталог намеренно пуст. Сюда монтируется bundle Разработчика 1 (ML, контракт C-04),
который физически лежит в `artifacts/ml/final_bundle/` — а это его зона владения,
и создавать её у себя мы не имеем права.

Пока в каталоге нет `manifest.json`, `GET /health/ready` отвечает `model_bundle: degraded`
(каталог смонтирован, но бандла нет) либо `not_configured` (каталога нет вовсе).
Готовым (`ok`) bundle объявляется только при наличии манифеста.

Переопределить путь: `MODEL_BUNDLE_HOST_PATH=./artifacts/ml/final_bundle docker compose up`.
