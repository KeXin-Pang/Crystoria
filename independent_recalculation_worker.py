"""Recalculate a declared small cohort using only the distributed Crystoria methods."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from crystoria import ase_atoms, read_structure
from production_core import (
    PhononThresholds,
    _calculator_factory,
    _evaluate_endpoint,
    _phonon_from_relaxed,
    _relax,
    classify_phonon_stability,
    fixed_mp_hull_distance,
    formation_energy_per_atom,
    load_elemental_reference_energies,
    mp2020_correction,
)
from pymatgen.io.ase import AseAtomsAdaptor


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible GPU is required")
    model = json.loads((args.methods / "configs/model.json").read_text())
    if checksum(args.checkpoint) != model["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    reference_assets = json.loads((args.methods / "configs/reference_assets.json").read_text())
    paths = {}
    for name, item in reference_assets.items():
        path = args.data / item["relative_to_data"]
        if checksum(path) != item["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {name}")
        paths[name] = path
    references = load_elemental_reference_energies(paths["elemental_references"])
    with gzip.open(paths["phase_diagram"], "rb") as handle:
        diagram = pickle.load(handle)  # Checksum-pinned distributed reference asset.
    protocols = {
        name: yaml.safe_load((args.methods / "configs" / filename).read_text())
        for name, filename in (("production", "phonon.yaml"), ("replay", "phonon-replay.yaml"))
    }
    started = time.monotonic()
    calculator = _calculator_factory(
        "equflashv2", {"checkpoint": str(args.checkpoint), "seed": 42}, "cuda"
    )()
    import GGNN.common.calculator as ggnn_calculator
    import production_core

    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0),
        "model_initialization_seconds": time.monotonic() - started,
        "GGNN_file": ggnn_calculator.__file__,
        "public_core_file": production_core.__file__,
        "checkpoint_sha256": model["sha256"],
        "worker_sha256": checksum(__file__),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "ase",
                "phonopy",
                "pymatgen",
                "polars",
                "spglib",
                "fairchem-core",
                "cuequivariance",
            )
        },
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    cases = json.loads(args.cases.read_text())["cases"]
    results = []
    for case in cases:
        clock = time.monotonic()
        try:
            payload = read_structure(args.data, case["record_id"], case["geometry"])
            atoms = ase_atoms(payload)
            if case["relax"]:
                atoms, residual = _relax(atoms, calculator, fmax=0.03, steps=200)
            else:
                atoms.calc = calculator
                residual = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            total_energy = float(atoms.get_potential_energy())
            protocol = "replay" if case["archived_protocol"].endswith("replay-v1") else "production"
            response = (
                _evaluate_endpoint(atoms, calculator, protocols[protocol])
                if protocol == "replay"
                else _phonon_from_relaxed(atoms, calculator, protocols[protocol])
            )
            structure = AseAtomsAdaptor.get_structure(atoms)
            correction, _ = mp2020_correction(structure)
            formation = formation_energy_per_atom(
                total_energy_ev=total_energy,
                correction_ev=correction,
                composition=structure.composition,
                references=references,
            )
            hull, fallback = fixed_mp_hull_distance(
                diagram, structure.composition, formation, references
            )
            result = {
                "case_id": case["case_id"],
                "record_id": case["record_id"],
                "status": "complete",
                "protocol": protocol,
                "geometry": case["geometry"],
                "relax": case["relax"],
                "minimum_frequency_thz": float(response["min_frequency_thz"]),
                "phonon_class": classify_phonon_stability(
                    response["min_frequency_thz"], PhononThresholds()
                ),
                "max_residual_force_ev_angstrom": residual,
                "total_energy_ev": total_energy,
                "formation_energy_ev_atom": formation,
                "signed_hull_mev_atom": 1000 * hull,
                "phase_diagram_fallback": bool(fallback),
                "num_atoms": len(atoms),
                "num_displacements": int(response["num_displacements"]),
                "structure": {
                    "atomic_numbers": atoms.numbers.tolist(),
                    "lattice_matrix_angstrom": atoms.cell.array.tolist(),
                    "fractional_coordinates": atoms.get_scaled_positions().tolist(),
                },
            }
        except Exception as error:
            result = {
                "case_id": case["case_id"],
                "record_id": case["record_id"],
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        result["elapsed_seconds"] = time.monotonic() - clock
        (args.output / f"{case['case_id']}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        results.append(result)
        print(
            json.dumps({k: result[k] for k in ("case_id", "status", "elapsed_seconds")}), flush=True
        )
    (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if any(row["status"] != "complete" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
