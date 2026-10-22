"""Reproduce grouped bootstrap intervals and the paired exact test without DFT files."""

import argparse
import json
from pathlib import Path

import polars as pl
from production_core import _bootstrap_balanced_accuracy, _mcnemar, classification_metrics


def compute(path, replicates=10000, seed=20260923):
    frame = pl.read_parquet(path)
    names = ["mattersim_5m", "equflashv2_45m_oam"]
    # The original records are ordered by benchmark_id. Clusters and predictions
    # must use the same order; sorting each model independently by frequency is invalid.
    reference = frame.filter(pl.col("model") == names[0]).sort("benchmark_id")
    truth = reference["pbe_stable"].to_numpy().astype(bool)
    predictions = {}
    for name in names:
        rows = frame.filter(pl.col("model") == name).sort("benchmark_id")
        if rows["benchmark_id"].to_list() != reference["benchmark_id"].to_list():
            raise ValueError("Reference and model sample IDs differ")
        if not (rows["status"] == "complete").all():
            raise ValueError("Archived full-coverage comparison contains a failed prediction")
        predictions[name] = rows["predicted_stable_primary"].to_numpy().astype(bool)
    return {
        "records": len(truth),
        "seed": seed,
        "replicates": replicates,
        "metrics": {
            name: classification_metrics(truth, values) for name, values in predictions.items()
        },
        "cluster_bootstrap": _bootstrap_balanced_accuracy(
            truth,
            predictions,
            reference["geometry_representation_sha256"].to_list(),
            replicates=replicates,
            seed=seed,
        ),
        "paired_exact_mcnemar": _mcnemar(
            truth == predictions[names[0]], truth == predictions[names[1]]
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    result = compute(args.predictions, args.replicates, args.seed)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
