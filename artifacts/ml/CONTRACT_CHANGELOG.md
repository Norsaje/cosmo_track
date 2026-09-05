# ML contract 1.0

Initial boundary; no existing backend contracts or routes were changed.

Backend imports `ReconstructionRequest`, `ReconstructionResult`, `NDVIReconstructor`
from `veg_recovery.contracts` and `load_reconstructor` from `veg_recovery.inference`.
Load a bundle once at process startup, then pass an aligned boolean gap mask per call.
Competition mode requires the mask to equal `is_synthetic_gap`; web mode accepts an
explicit mask. Dates are timezone-naive calendar days. Keys must be unique.

`ReconstructionPayload.from_result(result)` is the Pydantic 2 JSON boundary.
Schema is versioned `1.0`; missing distances serialize to JSON null, not infinity.
Predictions preserve requested gap order. The batch export selects only
`anon_polygon_id,date,primary_ndvi_pred`. Product harmonization is a separate field.
Diagnostics expose fallback, interval and harmonization limitations explicitly.

The anomaly baseline returns separate point/event tables; advanced detection remains
the DL team's responsibility. This package contains no DL implementation.
Integrators should approve the DTO schema when connecting their own routes; route
implementation and deployment are outside this workspace change.
