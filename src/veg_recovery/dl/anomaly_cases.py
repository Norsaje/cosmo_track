"""Воспроизводимые synthetic cases + графики; не оценка реальной точности."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from veg_recovery.anomalies.advanced import AdvancedAnomalyDetector
from .fixtures import anomaly_case, anomaly_reference


def generate_cases(destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    reference = anomaly_reference()
    reference.to_csv(output / "reference.csv", index=False)
    detector = AdvancedAnomalyDetector().fit(reference)
    (output / "config.json").write_text(
        json.dumps(asdict(detector.config), indent=2), encoding="utf-8"
    )
    results = []
    names = (
        "mild_pulse",
        "medium_pulse",
        "strong_pulse",
        "source_switch",
        "single_outlier",
        "wide_uncertainty",
        "normal",
    )
    for name in names:
        query, truth = anomaly_case(name)
        points, result = detector.analyze(query)
        points.to_csv(output / f"{name}.csv", index=False)
        payload = result.to_dict()
        payload["case_kind"] = "synthetic_proxy_not_real_accuracy"
        payload["truth_interval"] = [d.isoformat() for d in truth] if truth else None
        iou = None
        delay = None
        if truth and result.events:
            event = result.events[0]
            intersection = max(
                0,
                (min(truth[1], event.end_date) - max(truth[0], event.start_date)).days
                + 1,
            )
            union = (truth[1] - truth[0]).days + 1 + event.duration_days - intersection
            iou = intersection / union
            supported = points.loc[
                points.candidate & points.date.between(str(truth[0]), str(truth[1]))
            ]
            if len(supported) >= detector.config.min_event_points:
                delay = (
                    supported.date.iloc[detector.config.min_event_points - 1].date()
                    - truth[0]
                ).days
        payload["event_iou"] = iou
        payload["earliest_support_delay_days"] = delay
        payload["delay_caveat"] = (
            "Retrospective detector; earliest support is not a measured online alarm delay"
        )
        (output / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
        ax.plot(
            points.date,
            points.primary_ndvi_raw,
            color="0.65",
            linestyle="--",
            label="raw",
        )
        ax.plot(
            points.date,
            points.ndvi_harmonized,
            color="#176b91",
            marker=".",
            label="harmonized",
        )
        ax.plot(
            points.date, points.expected_ndvi, color="#20252b", label="LOYO expected"
        )
        ax.fill_between(
            points.date,
            points.q10,
            points.q90,
            color="#20252b",
            alpha=0.1,
            label="reference p10-p90",
        )
        for event in result.events:
            ax.axvspan(event.start_date, event.end_date, color="#b94141", alpha=0.2)
        ax.set(ylabel="NDVI")
        ax.set_title(f"{name} | synthetic fixture", loc="left", pad=34)
        ax.legend(
            loc="lower left", bbox_to_anchor=(0, 1), ncols=4, fontsize=8, frameon=False
        )
        ax.grid(alpha=0.15)
        fig.savefig(output / f"{name}.png", dpi=130)
        plt.close(fig)
        results.append(
            {
                "case": name,
                "expected_event": bool(truth),
                "events": len(result.events),
                "severity": [event.severity for event in result.events],
                "iou": iou,
                "earliest_support_delay_days": delay,
            }
        )
    summary = {
        "kind": "synthetic_only",
        "config_fingerprint": detector.config_fingerprint,
        "cases": results,
        "false_alerts_in_four_negative_controls": sum(
            r["events"] for r in results if not r["expected_event"]
        ),
        "real_world_precision_recall": None,
        "human_review": "pending",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/anomaly_cases/synthetic_v1")
    args = parser.parse_args(argv)
    print(json.dumps(generate_cases(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
