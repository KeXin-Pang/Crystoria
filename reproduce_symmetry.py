"""Reproduce Fig. S4 counts from the released paired symmetry assignments."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

SYSTEMS = (
    (2, "Triclinic"),
    (15, "Monoclinic"),
    (74, "Orthorhombic"),
    (142, "Tetragonal"),
    (167, "Trigonal"),
    (194, "Hexagonal"),
    (230, "Cubic"),
)


def crystal_system(number):
    if not 1 <= int(number) <= 230:
        raise ValueError(f"Invalid space group: {number}")
    return next(name for upper, name in SYSTEMS if number <= upper)


def summarize(frame):
    if frame["record_id"].n_unique() != frame.height:
        raise ValueError("Repeated record IDs")
    for name in ("source_spacegroup", "relaxed_spacegroup"):
        values = frame[name]
        if values.null_count() or not values.is_between(1, 230).all():
            raise ValueError(f"Missing or invalid assignments in {name}")
    matrix = np.zeros((230, 230), dtype=np.int64)
    a = frame["source_spacegroup"].to_numpy().astype(int)
    b = frame["relaxed_spacegroup"].to_numpy().astype(int)
    np.add.at(matrix, (a - 1, b - 1), 1)
    rows = []
    lower = 1
    for upper, name in SYSTEMS:
        total = int(matrix[lower - 1 : upper].sum())
        same = int(np.diag(matrix)[lower - 1 : upper].sum())
        rows.append(
            {
                "crystal_system": name,
                "structures": total,
                "same_spacegroup": same,
                "changed_spacegroup": total - same,
                "same_percent": 100 * same / total if total else None,
            }
        )
        lower = upper + 1
    summary = {
        "structures": frame.height,
        "same_spacegroup": int(np.trace(matrix)),
        "changed_spacegroup": int(matrix.sum() - np.trace(matrix)),
        "same_percent": 100 * np.trace(matrix).item() / frame.height if frame.height else None,
        "symprec_angstrom": 0.01,
        "angle_tolerance_degree": 5.0,
    }
    return matrix, rows, summary


def coordinate_check(data, frame, limit):
    import spglib

    selected = frame.sort("record_id").head(limit)
    keys = selected["record_id"].to_list()
    checked = 0
    for geometry in ("source", "relaxed"):
        structures = (
            pl.scan_parquet(data / "structures" / geometry / "*.parquet")
            .filter(pl.col("record_id").is_in(keys))
            .select("record_id", "structure_json")
            .collect()
        )
        joined = selected.join(structures, on="record_id", validate="1:1")
        if joined.height != len(keys):
            raise ValueError(f"Missing coordinates: {geometry}")
        for row in joined.iter_rows(named=True):
            payload = json.loads(row["structure_json"])
            symmetry = spglib.get_symmetry_dataset(
                (
                    np.asarray(payload["lattice_matrix_angstrom"]),
                    np.asarray(payload["fractional_coordinates"]),
                    np.asarray(payload["atomic_numbers"]),
                ),
                symprec=0.01,
                angle_tolerance=5.0,
            )
            if symmetry is None or int(symmetry.number) != row[f"{geometry}_spacegroup"]:
                raise ValueError(f"Symmetry mismatch: {row['record_id']} / {geometry}")
            checked += 1
    return {"structures": len(keys), "coordinate_views": checked, "status": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--check-coordinates", type=int, default=0)
    args = parser.parse_args()
    frame = pl.read_parquet(args.results / "paired-assignments" / "*.parquet")
    matrix, rows, summary = summarize(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    actual = pl.DataFrame(rows)
    expected = pl.read_csv(args.results / "spacegroup_consistency_by_system.csv")
    if not actual.equals(expected):
        raise ValueError("Crystal-system counts differ from the archived figure inputs")
    archived = pl.read_csv(args.results / "spacegroup_transition_matrix.csv")
    if not np.array_equal(matrix, archived.drop("source_spacegroup").to_numpy()):
        raise ValueError("Transition matrix differs from the archived figure inputs")
    actual.write_csv(args.output / "spacegroup_consistency_by_system.csv")
    pl.DataFrame(
        {
            "source_spacegroup": range(1, 231),
            **{f"relaxed_{n}": matrix[:, n - 1] for n in range(1, 231)},
        }
    ).write_csv(args.output / "spacegroup_transition_matrix.csv")
    if args.check_coordinates:
        if args.data is None or args.check_coordinates < 0:
            raise ValueError("A positive coordinate sample requires --data")
        summary["coordinate_check"] = coordinate_check(args.data, frame, args.check_coordinates)
    summary["status"] = "pass"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
