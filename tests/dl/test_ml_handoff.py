from hashlib import sha256
import json

import pandas as pd
import pytest

from veg_recovery.dl.ml_handoff import adapt_baseline_oof, bundle_integrity


def example():
    return pd.DataFrame(
        {
            "anon_polygon_id": ["p", "p"],
            "date": ["2024-07-01"] * 2,
            "mode": ["A", "A"],
            "repeat": [0, 1],
            "fold": [0, 0],
            "primary_ndvi": [0.5, 0.5],
            "pred_mean_neighbors": [0.4, 0.6],
            "seen_polygon": [True, False],
        }
    )


def test_repeat_identity_and_unseen_are_preserved():
    result = adapt_baseline_oof(example())
    assert result.fold.tolist() == ["r0_f0", "r1_f0"]
    assert result.is_unseen.tolist() == [False, True]
    assert result.split.tolist() == ["matched", "matched"]
    assert result.y_true.tolist() == [0.5, 0.5]
    assert result.primary_ndvi_pred.tolist() == [0.4, 0.6]


@pytest.mark.parametrize(
    "column,value",
    [("mode", "E"), ("seen_polygon", "False"), ("repeat", -1), ("fold", 0.5)],
)
def test_invalid_producer_metadata_is_rejected(column, value):
    frame = example().astype({column: "object"})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        adapt_baseline_oof(frame)


def test_duplicate_fold_key_is_not_silently_collapsed():
    frame = example()
    frame["repeat"] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        adapt_baseline_oof(frame)


@pytest.mark.parametrize("crlf", [False, True])
def test_bundle_hash_transport_damage_is_reported_without_repair(tmp_path, crlf):
    original = b'{\n  "value": 1\n}'
    actual = original.replace(b"\n", b"\r\n") if crlf else original
    (tmp_path / "state.json").write_bytes(actual)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_files": [
                    {"path": "state.json", "sha256": sha256(original).hexdigest()}
                ]
            }
        )
    )
    result = bundle_integrity(tmp_path)
    assert result["loadable_hashes"] is not crlf
    assert result["files"][0]["lf_normalized_matches"]
    assert not result["repair_applied"]
    assert (tmp_path / "state.json").read_bytes() == actual
