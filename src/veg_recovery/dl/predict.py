"""Research inference; не регистрируется автоматически в production core."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .artifacts import load_research_checkpoint
from .data import WindowDatasetAdapter, prepare_context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--keys", required=True, help="Explicit gap keys CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    from .training import predict_dataset

    model, prep, manifest = load_research_checkpoint(
        args.checkpoint, device=args.device
    )
    frame, keys = pd.read_csv(args.input), pd.read_csv(args.keys)
    context = prepare_context(frame, keys)
    dataset = WindowDatasetAdapter(
        context, keys, prep, window_days=manifest["metadata"]["window"]
    )
    predictions, seconds = predict_dataset(model, dataset, device=args.device)
    predictions["method"] = "research_residual_tcn_not_production"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(
        json.dumps(
            {
                "research_only": True,
                "rows": len(predictions),
                "infer_sec": seconds,
                "uncertainty": "not_calibrated",
                "masking_backend": context.masking_backend,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
