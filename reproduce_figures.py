"""Recompute numeric summaries and readable plots from released data, not server paths."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from crystoria import metric_table, statistics


def figure_data(data, output):
    source = pl.scan_parquet(data / "structures/source/*.parquet")
    species_type = pl.Struct({"atomic_numbers": pl.List(pl.Int64)})
    species = source.select(
        pl.col("structure_json")
        .str.json_decode(species_type)
        .struct.field("atomic_numbers")
        .alias("z")
    ).collect()
    elements = (
        species.select(pl.col("z").list.unique().explode().alias("atomic_numbers"))
        .group_by("atomic_numbers")
        .len(name="structures_containing_element")
        .sort("atomic_numbers")
    )
    order = (
        species.select(pl.col("z").list.n_unique().alias("chemical_system_order"))
        .group_by("chemical_system_order")
        .len(name="structures")
        .sort("chemical_system_order")
    )
    atoms = (
        source.group_by("primitive_num_sites")
        .len(name="structures")
        .sort("primitive_num_sites")
        .collect()
    )
    groups = (
        source.group_by("spacegroup_number")
        .len(name="total_count")
        .sort("spacegroup_number")
        .collect()
        .rename({"spacegroup_number": "space_group_number"})
    )
    for name, frame in (
        ("element_occurrence", elements),
        ("chemical_system_order_distribution", order),
        ("atom_count_distribution", atoms),
        ("space_group_distribution", groups),
    ):
        frame.write_csv(output / f"{name}.csv")
    quantities = pl.read_parquet(data / "labels/phonons.parquet").join(
        pl.read_parquet(data / "labels/thermodynamics.parquet"), on="record_id", validate="1:1"
    )
    omega = quantities["minimum_frequency_thz"].to_numpy().astype(float)
    hull = quantities["endpoint_hull_mev_atom"].to_numpy().astype(float)
    source_hull = quantities["screening_source_hull_mev_atom"].to_numpy().astype(float)
    hull_upper = max(100.0, math.ceil(float(np.quantile(hull, 0.99)) / 25) * 25.0)
    omega_lower = math.floor(float(np.quantile(omega, 0.05)) / 0.5) * 0.5
    omega_upper = max(0.8, math.ceil(float(np.quantile(omega, 0.995)) / 0.1) * 0.1)
    hull_edges = np.arange(0, hull_upper + 0.5, 1.0)
    source_edges = np.arange(0, 101, 1.0)
    omega_edges = np.arange(omega_lower, omega_upper + 0.025, 0.05)
    thermodynamic_mask = hull <= hull_upper
    joint_mask = thermodynamic_mask & (omega >= omega_lower) & (omega <= omega_upper)
    thermo_counts = np.histogram2d(
        source_hull[thermodynamic_mask], hull[thermodynamic_mask], bins=(source_edges, hull_edges)
    )[0]
    joint_counts = np.histogram2d(
        hull[joint_mask], omega[joint_mask], bins=(hull_edges, omega_edges)
    )[0]
    frequency_counts = np.histogram(omega, bins=omega_edges)[0]
    np.savez_compressed(
        output / "Fig3_numeric_grids.npz",
        source_edges=source_edges,
        hull_edges=hull_edges,
        frequency_edges=omega_edges,
        source_endpoint_counts=thermo_counts,
        energy_frequency_counts=joint_counts,
        frequency_counts=frequency_counts,
    )
    summary = {
        "structures": quantities.height,
        "endpoint_display_upper_mev_atom": hull_upper,
        "frequency_display_range_thz": [omega_lower, omega_upper],
        "endpoint_overflow": int((~thermodynamic_mask).sum()),
        "frequency_outside_display": int(quantities.height - frequency_counts.sum()),
        "joint_displayed": int(joint_mask.sum()),
        "endpoint_median_mev_atom": float(np.median(hull)),
        "endpoint_p95_mev_atom": float(np.quantile(hull, 0.95)),
        "hull_difference_over_20_mev_atom_percent": float(
            100 * np.mean(np.abs(hull - source_hull) > 20)
        ),
        "source_energy_definition": "source hull energy carried with the original "
        "relaxation/phonon calculation",
        "scope": "All valid release records, including inherited inversion scalars; "
        "not independent observations.",
    }
    (output / "Fig3_population_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = statistics(args.data)
    summary["figure3"] = figure_data(args.data, args.output)
    summary["benchmark"] = metric_table(
        args.evidence / "validation/pbe_full_reference_predictions.parquet"
    )
    (args.output / "recomputed_statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    source = pl.scan_parquet(args.data / "structures/source/*.parquet")
    sg = source.group_by("spacegroup_number").len().sort("spacegroup_number").collect()
    sg.write_csv(args.output / "spacegroup_counts.csv")
    labels = pl.read_parquet(args.data / "labels/phonons.parquet")
    thermo = pl.read_parquet(args.data / "labels/thermodynamics.parquet")
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    axs[0].bar(sg["spacegroup_number"], sg["len"], color="#167c9b")
    axs[0].set(xlabel="Source space group", ylabel="Structures")
    axs[1].hist(thermo["endpoint_hull_mev_atom"], bins=100, range=(0, 300), color="#167c9b")
    axs[1].set(xlabel="Relaxed hull estimate (meV/atom)", ylabel="Structures")
    axs[2].hist(labels["minimum_frequency_thz"], bins=100, color="#167c9b")
    axs[2].axvline(-0.10, color="#d77d46", linewidth=0.8)
    axs[2].axvline(-0.30, color="#ad515b", linewidth=0.8)
    axs[2].set(xlabel="Minimum frequency (THz)", ylabel="Structures")
    fig.savefig(args.output / "numeric_overview.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
