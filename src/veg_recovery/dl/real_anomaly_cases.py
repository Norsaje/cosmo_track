"""Exploratory retrospective anomaly audit; не DL CV и не размеченный benchmark.

Train CSV используется read-only. Калибровка/reference только на годах < query
year; supplied status/zscore/climatology не используются ни для fit, ни как labels.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from veg_recovery.anomalies.advanced import AdvancedAnomalyDetector, SensorHarmonizer
from veg_recovery.anomalies.events import ALGORITHM_VERSION
from .artifacts import file_sha256


def _write_json(path, value):
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )


def _observations(frame):
    frame = frame.copy()
    frame["is_observed"] = np.isfinite(frame.primary_ndvi)
    frame["selected_source"] = np.select(
        [np.isfinite(frame[c]) for c in ("s2_ndvi", "landsat_ndvi", "modis_ndvi")],
        ["s2", "landsat", "modis"],
        default="unknown",
    )
    hierarchy = frame.s2_ndvi.combine_first(frame.landsat_ndvi).combine_first(
        frame.modis_ndvi
    )
    visible = frame.is_observed
    if not np.allclose(
        frame.loc[visible, "primary_ndvi"], hierarchy[visible], rtol=0, atol=1e-10
    ):
        raise ValueError(
            "CSV primary/source hierarchy differs; use shared ML adapter instead"
        )
    # Это только invalid-value proxy. Спутниковых QA/cloud flags в CSV нет.
    frame["quality"] = np.where(frame.primary_ndvi.between(-1, 1), 1.0, 0.0)
    return frame


def sensor_alignment_diagnostics(query_raw, harmonizer, output):
    """Диагностика шкалы на синхронных sensor pairs, НЕ primary OOF metric."""
    results = []
    for source in ("landsat", "modis"):
        column = source + "_ndvi"
        pair = query_raw.loc[
            query_raw.s2_ndvi.between(-1, 1) & query_raw[column].between(-1, 1)
        ].copy()
        if pair.empty:
            continue
        evaluation = pair.assign(primary_ndvi=pair[column], selected_source=source)
        corrected = harmonizer.transform(evaluation).ndvi_harmonized.to_numpy()
        truth = pair.s2_ndvi.to_numpy()
        raw = pair[column].to_numpy()
        results.append(
            {
                "source": source,
                "pairs": len(pair),
                "raw_mae": float(np.mean(np.abs(raw - truth))),
                "harmonized_mae": float(np.mean(np.abs(corrected - truth))),
                "raw_rmse": float(np.sqrt(np.mean((raw - truth) ** 2))),
                "harmonized_rmse": float(np.sqrt(np.mean((corrected - truth) ** 2))),
                "raw_median_bias": float(np.median(raw - truth)),
                "harmonized_median_bias": float(np.median(corrected - truth)),
                "metric_scope": "same_day_sensor_alignment_not_primary_ndvi_oof",
            }
        )
    pd.DataFrame(results).to_csv(Path(output) / "sensor_alignment.csv", index=False)
    return results


def generate_real_cases(input_path, output_path, *, year=2024, limit_polygons=None):
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path, parse_dates=["date"])
    if "is_synthetic_gap" in raw and raw.is_synthetic_gap.fillna(False).any():
        raise ValueError(
            "Use train CSV for this exploratory audit, not competition test gaps"
        )
    ref_raw = raw.loc[raw.date.dt.year < year].copy()
    query_raw = raw.loc[raw.date.dt.year == year].copy()
    if ref_raw.empty or query_raw.empty:
        raise ValueError(
            "Both earlier reference years and requested query year are required"
        )
    counts = (
        query_raw.groupby("anon_polygon_id")
        .primary_ndvi.count()
        .sort_values(ascending=False, kind="stable")
    )
    polygons = counts.index.tolist()
    if limit_polygons is not None:
        polygons = polygons[:limit_polygons]
    started = time.perf_counter()
    harmonizer = SensorHarmonizer().fit(ref_raw)
    ref = harmonizer.transform(_observations(ref_raw))
    detector = AdvancedAnomalyDetector().fit(ref)
    fit_sec = time.perf_counter() - started
    _write_json(output / "calibration.json", harmonizer.to_dict())
    _write_json(output / "config.json", asdict(detector.config))
    sensor_alignment_diagnostics(query_raw, harmonizer, output)
    candidates = []
    for polygon in polygons:
        query = harmonizer.transform(
            _observations(query_raw.loc[query_raw.anon_polygon_id == polygon])
        )
        started = time.perf_counter()
        points, result = detector.analyze(query)
        seconds = time.perf_counter() - started
        events = list(result.events)
        strongest = max(events, key=lambda e: e.score, default=None)
        suspicious = sum(
            "SOURCE_SWITCH_RISK" in e.reason_codes or e.confidence < 0.7 for e in events
        )
        summary = {
            "anon_polygon_id": polygon,
            "crop_type": str(query.crop_type.iloc[0]),
            "rows": len(query),
            "observed": int(query.is_observed.sum()),
            "events": len(events),
            "critical_events": sum(e.severity == "critical" for e in events),
            "max_score": strongest.score if strongest else 0.0,
            "max_confidence": strongest.confidence if strongest else None,
            "source_risk_or_low_confidence_events": suspicious,
            "unsupported_observed": int(
                (query.is_observed & ~query.calibration_supported).sum()
            ),
            "warnings": list(result.warnings),
            "detect_sec": seconds,
        }
        candidates.append((summary, points, result))
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "anon_polygon_id",
                        "events",
                        "critical_events",
                        "detect_sec",
                    )
                }
            ),
            flush=True,
        )
    ranked = sorted(
        candidates, key=lambda x: (-x[0]["max_score"], x[0]["anon_polygon_id"])
    )
    selected = [("strong_candidate", item) for item in ranked[:3] if item[0]["events"]]
    selected_ids = {item[0]["anon_polygon_id"] for _, item in selected}
    ambiguous = sorted(
        (item for item in candidates if item[0]["anon_polygon_id"] not in selected_ids),
        key=lambda x: (
            -x[0]["source_risk_or_low_confidence_events"],
            -x[0]["unsupported_observed"],
            -x[0]["max_score"],
            x[0]["anon_polygon_id"],
        ),
    )
    selected.extend(("questionable_control", item) for item in ambiguous[:3])
    review_rows = []
    for group, (summary, points, result) in selected:
        polygon = summary["anon_polygon_id"]
        name = f"{group}_{polygon}_{year}"
        payload = result.to_dict()
        payload.update(
            case_kind="real_exploratory_no_ground_truth",
            review_group=group,
            anon_polygon_id=polygon,
            year=year,
            reference_max_year=year - 1,
            human_review="pending",
            qa_available=False,
            weather_reason_codes="not_computed",
            caveats=[
                "No labeled anomaly truth: these are candidates, not confirmed crop damage",
                "Supplied train status/zscore/climatology were not used as labels or features",
                "Cloud QA, causal attribution and region metadata are unavailable",
                "Thresholds unchanged from synthetic config; no tuning on this query year",
            ],
        )
        _write_json(output / (name + ".json"), payload)
        columns = [
            "anon_polygon_id",
            "date",
            "crop_type",
            "primary_ndvi_raw",
            "ndvi_harmonized",
            "selected_source",
            "calibration_supported",
            "is_observed",
            "quality",
            "expected_ndvi",
            "climatology_scale",
            "reference_years",
            "reference_level",
            "q10",
            "q25",
            "q50",
            "q75",
            "q90",
            "residual",
            "robust_z",
            "effective_z",
            "source_switch_risk",
            "confidence",
            "candidate",
        ]
        points[columns].to_csv(output / (name + ".csv"), index=False)
        _plot(points, result, output / (name + ".png"), polygon, year, group)
        review_rows.append({"case": name, "review_group": group, **summary})
    pd.DataFrame([summary for summary, _, _ in candidates]).to_csv(
        output / "polygon_summary.csv", index=False
    )
    summary = {
        "kind": "real_exploratory_no_ground_truth",
        "algorithm_version": ALGORITHM_VERSION,
        "input_sha256": file_sha256(input_path),
        "input_name": Path(input_path).name,
        "query_year": year,
        "reference_years": sorted(int(y) for y in ref_raw.date.dt.year.unique()),
        "fit_rows": len(ref_raw),
        "fit_visible": int(np.isfinite(ref_raw.primary_ndvi).sum()),
        "evaluated_polygons": len(candidates),
        "evaluated_rows": sum(s["rows"] for s, _, _ in candidates),
        "fit_sec": fit_sec,
        "detect_sec_total": sum(s["detect_sec"] for s, _, _ in candidates),
        "detector_events": sum(s["events"] for s, _, _ in candidates),
        "critical_events": sum(s["critical_events"] for s, _, _ in candidates),
        "precision_recall": None,
        "false_alert_rate": None,
        "human_review": "pending",
        "selection_rule": "3 largest event scores; 3 other polygons ranked by source-risk/low-confidence count",
        "selected_cases": review_rows,
    }
    _write_json(output / "summary.json", summary)
    return summary


def _plot(points, result, path, polygon, year, group):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, zx) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1], layout="constrained"
    )
    for source, color in (
        ("s2", "#176b91"),
        ("landsat", "#8755a3"),
        ("modis", "#bb7c2d"),
    ):
        rows = points.loc[points.is_observed & points.selected_source.eq(source)]
        ax.scatter(
            rows.date, rows.ndvi_harmonized, color=color, s=17, label=source, zorder=3
        )
    ax.plot(
        points.date,
        points.primary_ndvi_raw,
        color="0.65",
        linestyle="--",
        linewidth=0.8,
        label="raw",
    )
    ax.plot(
        points.date,
        points.expected_ndvi,
        color="#222222",
        label="expected (past years)",
    )
    ax.fill_between(
        points.date,
        points.q10,
        points.q90,
        color="#222222",
        alpha=0.1,
        label="reference p10-p90",
    )
    observed = points.loc[points.is_observed]
    zx.scatter(observed.date, observed.robust_z, s=12, color="#176b91")
    for event in result.events:
        color = "#b94141" if event.severity == "critical" else "#c98c24"
        for axis in (ax, zx):
            axis.axvspan(event.start_date, event.end_date, color=color, alpha=0.15)
    zx.axhline(-1, color="#c98c24", linestyle="--", linewidth=0.8)
    zx.axhline(-2, color="#b94141", linestyle="--", linewidth=0.8)
    ax.set_title(f"{polygon} / {year} | {group} | no ground truth", loc="left", pad=32)
    ax.set_ylabel("NDVI on S2 scale")
    zx.set_ylabel("robust z")
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1), ncols=6, fontsize=8, frameon=False
    )
    for axis in (ax, zx):
        axis.grid(alpha=0.15)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/train_dataset.csv")
    parser.add_argument("--output", default="reports/anomaly_cases/real_2024")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit-polygons", type=int)
    args = parser.parse_args(argv)
    summary = generate_real_cases(
        args.input, args.output, year=args.year, limit_polygons=args.limit_polygons
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "evaluated_polygons",
                    "detector_events",
                    "critical_events",
                    "detect_sec_total",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
