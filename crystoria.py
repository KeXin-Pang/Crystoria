"""Portable Crystoria loader and calculation entrypoints (no workspace imports)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def labels(root):
    root = Path(root)
    return pl.scan_parquet(root / "labels/phonons.parquet").join(
        pl.scan_parquet(root / "labels/thermodynamics.parquet"), on="record_id", how="inner"
    )


def read_structure(root, record_id, geometry="relaxed"):
    result = (
        pl.scan_parquet(Path(root) / "structures" / geometry / "*.parquet")
        .filter(pl.col("record_id") == record_id)
        .collect()
    )
    if result.height != 1:
        raise ValueError(f"Expected one structure for {record_id}, found {result.height}")
    return json.loads(result["structure_json"][0])


def ase_atoms(payload):
    from ase import Atoms

    return Atoms(
        numbers=payload["atomic_numbers"],
        cell=payload["lattice_matrix_angstrom"],
        scaled_positions=payload["fractional_coordinates"],
        pbc=True,
    )


def verify(candidate):
    candidate = Path(candidate).resolve()
    failures, count = [], 0
    for line in (candidate / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (candidate / name).resolve()
        if not path.is_relative_to(candidate) or not path.is_file() or checksum(path) != expected:
            failures.append(name)
        count += 1
    if failures:
        raise ValueError(f"Invalid files: {failures}")
    return {"checksummed_files": count, "status": "pass"}


def statistics(data):
    data = Path(data)
    phonons = pl.read_parquet(data / "labels/phonons.parquet")
    source = pl.scan_parquet(data / "structures/source/*.parquet")
    return {
        "structures": phonons.height,
        "classes": dict(phonons.group_by("phonon_class_name").len().rows()),
        "atoms": source.select(pl.col("primitive_num_sites").sum()).collect().item(),
        "independently_evaluated": int(phonons["label_independently_evaluated"].sum()),
    }


def metric_table(path):
    frame = pl.read_parquet(path)
    output = {}
    for key, group in frame.partition_by("model", as_dict=True).items():
        truth = group["pbe_stable"].to_numpy().astype(bool)
        completed = group["status"].to_numpy() == "complete"
        predictions = group["predicted_stable_primary"].fill_null(False).to_numpy().astype(bool)
        correct = (truth == predictions) & completed
        recalls = [float(np.mean(correct[truth == label])) for label in (False, True)]
        output[key[0]] = {
            "structures": group.height,
            "coverage": float(np.mean(completed)),
            "accuracy": float(np.mean(correct)),
            "balanced_accuracy": float(np.mean(recalls)),
            "unstable_recall": recalls[0],
            "stable_recall": recalls[1],
        }
    return output


def endpoint_energetics(data, record_id):
    """Recompute MP2020 alignment from the archived relaxed geometry and energy."""
    import gzip
    import pickle

    from production_core import (
        fixed_mp_hull_distance,
        formation_energy_per_atom,
        load_elemental_reference_energies,
        mp2020_correction,
    )
    from pymatgen.core import Structure

    folder = Path(__file__).resolve().parent
    assets = json.loads((folder / "configs/reference_assets.json").read_text())
    paths = {}
    for name, info in assets.items():
        path = Path(data) / info["relative_to_data"]
        if checksum(path) != info["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {name}")
        paths[name] = path
    payload = read_structure(data, record_id, "relaxed")
    structure = Structure(
        payload["lattice_matrix_angstrom"],
        payload["atomic_numbers"],
        payload["fractional_coordinates"],
    )
    energy = (
        pl.scan_parquet(Path(data) / "labels/thermodynamics.parquet")
        .filter(pl.col("record_id") == record_id)
        .select("endpoint_total_energy_ev")
        .collect()
        .item()
    )
    correction, details = mp2020_correction(structure)
    references = load_elemental_reference_energies(paths["elemental_references"])
    formation = formation_energy_per_atom(
        total_energy_ev=energy,
        correction_ev=correction,
        composition=structure.composition,
        references=references,
    )
    # Only deserialize the bundled, checksum-pinned upstream reference asset.
    with gzip.open(paths["phase_diagram"], "rb") as handle:
        diagram = pickle.load(handle)
    hull, fallback = fixed_mp_hull_distance(diagram, structure.composition, formation, references)
    return {
        "record_id": record_id,
        "endpoint_total_energy_ev": energy,
        "mp2020_correction_ev": correction,
        "correction_details": details,
        "endpoint_formation_ev_atom": formation,
        "endpoint_signed_hull_mev_atom": 1000 * hull,
        "endpoint_hull_mev_atom": 1000 * max(0, hull),
        "phase_diagram_fallback": fallback,
        "energy_source": "archived EquFlashV2 total energy; no new force-model inference",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify")
    p.add_argument("candidate", type=Path)
    for name in ("stats", "select", "export", "calculate", "energetics"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, type=Path)
        if name in {"export", "calculate", "energetics"}:
            p.add_argument("--record-id", required=True)
            p.add_argument("--geometry", choices=("source", "relaxed"), default="relaxed")
            p.add_argument("--output", required=True, type=Path)
        if name == "select":
            p.add_argument("--max-hull-mev", type=float, default=100.0)
            p.add_argument("--min-frequency-thz", type=float, default=-0.10)
            p.add_argument("--ordinary-only", action="store_true")
            p.add_argument("--output", required=True, type=Path)
        if name == "calculate":
            p.add_argument("--checkpoint", required=True, type=Path)
            p.add_argument("--device", default="cpu")
            p.add_argument("--relax", action="store_true")
            p.add_argument(
                "--protocol", choices=("archived", "production", "replay"), default="archived"
            )
    p = sub.add_parser("metrics")
    p.add_argument("predictions", type=Path)
    p = sub.add_parser("invert")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("canonicalize")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.candidate)
    elif args.command == "stats":
        result = statistics(args.data)
    elif args.command == "metrics":
        result = metric_table(args.predictions)
    elif args.command == "select":
        frame = labels(args.data).filter(
            (pl.col("endpoint_hull_mev_atom") <= args.max_hull_mev)
            & (pl.col("minimum_frequency_thz") >= args.min_frequency_thz)
        )
        if args.ordinary_only:
            frame = frame.filter(pl.col("label_independently_evaluated"))
        frame.select("record_id").sink_parquet(args.output)
        result = {"selection": str(args.output)}
    elif args.command == "export":
        from ase.io import write

        write(args.output, ase_atoms(read_structure(args.data, args.record_id, args.geometry)))
        result = {"file": str(args.output)}
    elif args.command == "energetics":
        result = endpoint_energetics(args.data, args.record_id)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "invert":
        from production_core import invert_fractional_coordinates

        payload = json.loads(args.input.read_text())
        payload["fractional_coordinates"] = invert_fractional_coordinates(
            payload["fractional_coordinates"]
        ).tolist()
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        result = {"file": str(args.output), "parity_even_scalars_only": True}
    elif args.command == "canonicalize":
        from ase.io import read
        from production_core import canonicalize_structure
        from pymatgen.io.ase import AseAtomsAdaptor

        result = canonicalize_structure(
            AseAtomsAdaptor.get_structure(read(args.input)),
            symprec=0.01,
            angle_tolerance=5,
            decimals=6,
            minimum_distance=0.5,
            source_num_sites_parse_max=48,
            primitive_num_sites_max=48,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        import yaml
        from production_core import (
            PhononThresholds,
            _calculator_factory,
            _evaluate_endpoint,
            _phonon_from_relaxed,
            _relax,
            classify_phonon_stability,
        )

        folder = Path(__file__).resolve().parent
        model = json.loads((folder / "configs/model.json").read_text())
        if checksum(args.checkpoint) != model["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        protocol_name = args.protocol
        if protocol_name == "archived":
            tag = (
                pl.read_parquet(args.data / "labels/phonons.parquet")
                .filter(pl.col("record_id") == args.record_id)["phonon_protocol_id"]
                .item()
            )
            protocol_name = "replay" if tag.endswith("replay-v1") else "production"
        config_name = "phonon-replay.yaml" if protocol_name == "replay" else "phonon.yaml"
        protocol = yaml.safe_load((folder / "configs" / config_name).read_text())
        calculator = _calculator_factory(
            "equflashv2", {"checkpoint": str(args.checkpoint), "seed": 42}, args.device
        )()
        atoms = ase_atoms(read_structure(args.data, args.record_id, args.geometry))
        if args.relax:
            atoms, residual_force = _relax(atoms, calculator, fmax=0.03, steps=200)
        else:
            atoms.calc = calculator
            residual_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        result = (
            _evaluate_endpoint(atoms, calculator, protocol)
            if protocol_name == "replay"
            else _phonon_from_relaxed(atoms, calculator, protocol)
        )
        result = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
        result["calculation_protocol"] = protocol_name
        result["max_residual_force_ev_angstrom"] = residual_force
        result["phonon_class"] = classify_phonon_stability(
            result["min_frequency_thz"], PhononThresholds()
        )
        result["checkpoint_sha256"] = model["sha256"]
        result["total_energy_ev"] = float(atoms.get_potential_energy())
        result["relaxed_structure"] = {
            "atomic_numbers": atoms.numbers.tolist(),
            "lattice_matrix_angstrom": atoms.cell.array.tolist(),
            "fractional_coordinates": atoms.get_scaled_positions().tolist(),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
