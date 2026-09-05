"""Build metric figures from the committed result summary."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "model_metrics.csv"
OUTPUT = ROOT / "figures"
COLORS = {"baseline": "#64748b", "catboost": "#7c3aed", "ensemble": "#ec4899"}
LABELS = {"baseline": "Baseline", "catboost": "CatBoost", "ensemble": "Ансамбль"}


def read_metrics() -> list[dict[str, str]]:
    with DATA.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def build_composite(rows: list[dict[str, str]]) -> None:
    selected = {row["model"]: float(row["rmse"]) for row in rows if row["evaluation"] == "composite"}
    models = ["baseline", "catboost", "ensemble"]
    values = [selected[model] for model in models]
    improvement = [(selected["baseline"] - value) / selected["baseline"] * 100 for value in values]

    fig, axis = plt.subplots(figsize=(9.6, 4.8))
    bars = axis.barh(range(3), values, color=[COLORS[model] for model in models], height=0.58)
    axis.set_yticks(range(3), [LABELS[model] for model in models])
    axis.invert_yaxis()
    axis.set_xlim(0, 0.12)
    axis.set_xlabel("Composite RMSE — меньше лучше")
    axis.set_title("Ансамбль снижает ошибку baseline на 8.70%", loc="left", pad=16)
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)
    for index, (bar, value) in enumerate(zip(bars, values)):
        suffix = "" if index == 0 else f"   −{improvement[index]:.2f}%"
        axis.text(
            value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.6f}{suffix}",
            va="center",
            fontweight="bold",
        )
    fig.text(0.01, 0.01, "OOF composite: 50% A + 25% B + 15% C + 10% D", color="#64748b", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT / "composite_rmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_modes(rows: list[dict[str, str]]) -> None:
    modes = ["A", "B", "C", "D"]
    models = ["baseline", "catboost", "ensemble"]
    lookup = {(row["evaluation"], row["model"]): float(row["rmse"]) for row in rows}
    x = np.arange(len(modes))
    width = 0.24

    fig, axis = plt.subplots(figsize=(10.5, 5.7))
    for offset, model in zip((-width, 0.0, width), models):
        values = [lookup[(mode, model)] for mode in modes]
        bars = axis.bar(x + offset, values, width, label=LABELS[model], color=COLORS[model])
        axis.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=3,
            fontsize=8.5,
            rotation=90,
        )
    axis.set_xticks(x, ["A · matched mask", "B · unseen", "C · temporal", "D · hard"])
    axis.set_ylim(0, 0.285)
    axis.set_ylabel("RMSE — меньше лучше")
    axis.set_title("Качество в четырёх режимах проверки", loc="left", pad=16)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=3, loc="upper left")
    fig.text(
        0.01,
        0.01,
        "CatBoost усиливает A/B; temporal-C остаётся главным ограничением.",
        color="#64748b",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT / "cv_modes_rmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()
    rows = read_metrics()
    build_composite(rows)
    build_modes(rows)


if __name__ == "__main__":
    main()
