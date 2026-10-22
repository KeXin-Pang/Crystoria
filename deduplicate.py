"""Bounded continuous candidate search with strict orientation-preserving matching."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from production_core import (
    ProperStructureMatcher,
    continuous_candidates,
    direct_representatives,
    geometry_features,
    structure_from_canonical_json,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--maximum-pairs", type=int, default=1000000)
    args = p.parse_args()
    settings = json.loads((Path(__file__).parent / "configs/identity.json").read_text())
    frame = pl.read_parquet(args.input).sort("record_id").with_row_index("record_index")
    if "canonical_structure_json" not in frame.columns and "structure_json" in frame.columns:
        frame = frame.rename({"structure_json": "canonical_structure_json"})
    features = []
    for row in frame.iter_rows(named=True):
        composition, values = geometry_features(row["canonical_structure_json"], settings)
        features.append(
            {"record_index": row["record_index"], "composition": composition, "features": values}
        )
    candidates = continuous_candidates(pl.DataFrame(features), args.maximum_pairs)
    matcher = ProperStructureMatcher(**settings["matcher"])
    structures = {}
    verified = []
    for left, right in candidates.iter_rows():
        for key in (left, right):
            if key not in structures:
                structures[key] = structure_from_canonical_json(
                    frame["canonical_structure_json"][key]
                )
        if matcher.fit(structures[left], structures[right]):
            verified.append((left, right))
    edges = np.asarray(verified, dtype=int).reshape(-1, 2)
    mirrors = (
        frame["is_derived_mirror"].to_numpy()
        if "is_derived_mirror" in frame.columns
        else np.zeros(frame.height, dtype=bool)
    )
    representatives = direct_representatives(frame.height, edges, mirrors)
    frame.select("record_id").with_columns(
        pl.Series("retained_record_id", frame["record_id"][representatives])
    ).write_parquet(args.output)
    print(
        json.dumps(
            {
                "records": frame.height,
                "candidate_pairs": candidates.height,
                "verified_pairs": len(verified),
                "removed": int(np.sum(representatives != np.arange(frame.height))),
            }
        )
    )


if __name__ == "__main__":
    main()
