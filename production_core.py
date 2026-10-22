"""Selected production functions; see production_source_manifest.json for origin."""
from __future__ import annotations
from collections.abc import Callable
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import importlib.machinery
import json
import math
from pathlib import Path
import sys
import types
from typing import Any
import warnings
import numpy as np
import polars as pl
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import spglib


@dataclass(frozen=True)
class PhononThresholds:
    stable_min_frequency_thz: float = -0.10
    unstable_min_frequency_thz: float = -0.30


def classify_phonon_stability(min_frequency_thz: float, thresholds: PhononThresholds) -> str:
    """Classify finite-displacement frequencies with a numerical-noise band."""
    if not np.isfinite(min_frequency_thz):
        return "failed"
    if min_frequency_thz >= thresholds.stable_min_frequency_thz:
        return "stable"
    if min_frequency_thz < thresholds.unstable_min_frequency_thz:
        return "unstable"
    return "borderline"


def _phonopy_atoms(atoms: Any) -> Any:
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )


def _relax(atoms: Any, calculator: Any, fmax: float, steps: int) -> tuple[Any, float]:
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    relaxed = atoms.copy()
    relaxed.calc = calculator
    filtered = FrechetCellFilter(relaxed)
    optimizer = FIRE(filtered, logfile=None)
    optimizer.run(fmax=fmax, steps=steps)
    forces = relaxed.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max())
    return relaxed, max_force


def _spacegroup_number(atoms: Any, *, symprec: float) -> int:
    """Return a fail-closed space-group number for a phonopy input geometry."""

    import spglib

    dataset = spglib.get_symmetry_dataset(
        (
            np.asarray(atoms.cell.array, dtype=float),
            np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float),
            np.asarray(atoms.numbers, dtype=int),
        ),
        symprec=symprec,
    )
    if dataset is None:
        raise ValueError("space-group detection failed during Niggli retry")
    return int(dataset.number)


def _build_phonopy(
    *,
    relaxed: Any,
    supercell_matrix: list[list[int]],
    primitive_matrix: Any,
    phonopy_symprec: float,
    niggli_retry_lattice_round_decimals: int | None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build Phonopy, optionally repairing only sub-microangstrom Niggli noise.

    The retry is deliberately opt-in. It preserves fractional coordinates and
    composition, requires the detected space group to remain unchanged, and
    exposes the exact perturbation in the candidate result.
    """

    from phonopy import Phonopy

    kwargs = {
        "supercell_matrix": np.asarray(supercell_matrix, dtype=int),
        "primitive_matrix": primitive_matrix,
        "symprec": phonopy_symprec,
    }
    base_metadata: dict[str, Any] = {
        "niggli_retry_used": False,
        "niggli_retry_lattice_round_decimals": niggli_retry_lattice_round_decimals,
    }
    try:
        return Phonopy(_phonopy_atoms(relaxed), **kwargs), relaxed, base_metadata
    except Exception as exc:
        if (
            "Niggli reduction failed" not in str(exc)
            or niggli_retry_lattice_round_decimals is None
        ):
            raise

    decimals = int(niggli_retry_lattice_round_decimals)
    if not 6 <= decimals <= 12:
        raise ValueError("Niggli retry lattice rounding must be between 6 and 12 decimals")
    original_cell = np.asarray(relaxed.cell.array, dtype=float)
    rounded_cell = np.round(original_cell, decimals=decimals)
    maximum_cell_change = float(np.max(np.abs(rounded_cell - original_cell)))
    # The tiny relative margin covers binary representation of decimal ties.
    allowed_change = 0.500001 * 10.0 ** (-decimals)
    if maximum_cell_change > allowed_change:
        raise ValueError("Niggli retry lattice perturbation exceeds the declared rounding bound")
    if not np.isfinite(np.linalg.det(rounded_cell)) or np.linalg.det(rounded_cell) <= 0.0:
        raise ValueError("Niggli retry produced a non-positive lattice volume")

    conditioned = relaxed.copy()
    original_positions = np.asarray(conditioned.positions, dtype=float).copy()
    conditioned.set_cell(rounded_cell, scale_atoms=True)
    maximum_position_change = float(
        np.max(np.linalg.norm(np.asarray(conditioned.positions) - original_positions, axis=1))
    )
    spacegroup_before = _spacegroup_number(relaxed, symprec=phonopy_symprec)
    spacegroup_after = _spacegroup_number(conditioned, symprec=phonopy_symprec)
    if spacegroup_after != spacegroup_before:
        raise ValueError(
            "Niggli retry changed the detected space group: "
            f"{spacegroup_before} -> {spacegroup_after}"
        )
    phonon = Phonopy(_phonopy_atoms(conditioned), **kwargs)
    return (
        phonon,
        conditioned,
        {
            "niggli_retry_used": True,
            "niggli_retry_lattice_round_decimals": decimals,
            "niggli_retry_max_cell_component_change_angstrom": maximum_cell_change,
            "niggli_retry_max_cartesian_position_change_angstrom": maximum_position_change,
            "niggli_retry_spacegroup_before": spacegroup_before,
            "niggli_retry_spacegroup_after": spacegroup_after,
        },
    )


def _install_torch_scatter_fallback() -> None:
    """Provide inference-only PyTorch scatter ops on old-glibc hosts.

    The published PyG wheels used by EquFlashV2 require a newer glibc than this
    server. Native ``Tensor.scatter_reduce_`` preserves GPU execution and is
    sufficient for the EquFlash inference and graph construction paths used by
    this benchmark.
    """
    import torch

    module = types.ModuleType("torch_scatter")
    module.__spec__ = importlib.machinery.ModuleSpec("torch_scatter", loader=None)
    module.__version__ = "fallback-pytorch-native"

    def _broadcast_index(index: Any, source: Any, dim: int) -> Any:
        if index.ndim == source.ndim:
            return index
        shape = [1] * source.ndim
        shape[dim] = index.numel()
        return index.reshape(shape).expand_as(source)

    def scatter(
        source: Any,
        index: Any,
        dim: int = -1,
        out: Any | None = None,
        dim_size: int | None = None,
        reduce: str = "sum",
    ) -> Any:
        dim %= source.ndim
        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() else 0
        output_shape = list(source.shape)
        output_shape[dim] = dim_size
        expanded_index = _broadcast_index(index, source, dim)
        normalized_reduce = {"add": "sum", "min": "amin", "max": "amax"}.get(reduce, reduce)
        if normalized_reduce in {"sum", "mean"}:
            result = source.new_zeros(output_shape) if out is None else out
            result.scatter_add_(dim, expanded_index, source)
            if normalized_reduce == "mean":
                counts = source.new_zeros(output_shape)
                counts.scatter_add_(dim, expanded_index, torch.ones_like(source))
                result = result / counts.clamp_min(1)
            return result
        if normalized_reduce in {"amin", "amax", "prod"}:
            if out is None:
                fill = {
                    "amin": float("inf"),
                    "amax": float("-inf"),
                    "prod": 1.0,
                }[normalized_reduce]
                result = source.new_full(output_shape, fill)
            else:
                result = out
            result.scatter_reduce_(
                dim, expanded_index, source, reduce=normalized_reduce, include_self=True
            )
            if normalized_reduce in {"amin", "amax"}:
                result = torch.where(torch.isinf(result), torch.zeros_like(result), result)
            return result
        raise ValueError(f"Unsupported scatter reduction: {reduce}")

    def segment_coo(
        source: Any,
        index: Any,
        out: Any | None = None,
        dim_size: int | None = None,
        reduce: str = "sum",
    ) -> Any:
        return scatter(source, index, dim=0, out=out, dim_size=dim_size, reduce=reduce)

    def segment_csr(source: Any, indptr: Any, out: Any | None = None, reduce: str = "sum") -> Any:
        counts = indptr[1:] - indptr[:-1]
        index = torch.repeat_interleave(torch.arange(len(counts), device=indptr.device), counts)
        return scatter(source, index, dim=0, out=out, dim_size=len(counts), reduce=reduce)

    module.scatter = scatter
    module.scatter_add = lambda source, index, dim=-1, out=None, dim_size=None: scatter(
        source, index, dim=dim, out=out, dim_size=dim_size, reduce="sum"
    )
    module.scatter_mean = lambda source, index, dim=-1, out=None, dim_size=None: scatter(
        source, index, dim=dim, out=out, dim_size=dim_size, reduce="mean"
    )
    module.scatter_max = lambda source, index, dim=-1, out=None, dim_size=None: (
        scatter(source, index, dim=dim, out=out, dim_size=dim_size, reduce="max"),
        None,
    )
    module.scatter_min = lambda source, index, dim=-1, out=None, dim_size=None: (
        scatter(source, index, dim=dim, out=out, dim_size=dim_size, reduce="min"),
        None,
    )
    module.segment_coo = segment_coo
    module.segment_csr = segment_csr
    sys.modules["torch_scatter"] = module


def _calculator_factory(model_name: str, model: dict[str, Any], device: str) -> Callable[[], Any]:
    checkpoint = str(model["checkpoint"])
    if model_name.startswith("mattersim"):
        from mattersim.forcefield import MatterSimCalculator

        return lambda: MatterSimCalculator(load_path=checkpoint, device=device)
    if model_name.startswith("equflash"):
        _install_torch_scatter_fallback()
        import GGNN.common.calculator as ggnn_calculator

        ggnn_calculator.setup_imports = lambda: None
        return lambda: ggnn_calculator.UCalculator(
            checkpoint_path=checkpoint,
            cpu=device == "cpu",
            seed=int(model.get("seed", 42)),
        )
    if model_name == "equiformerv3dens":
        _install_torch_scatter_fallback()
        import experimental.models.equiformer_v3.equiformer_v3  # noqa: F401
        import fairchem.core.common.relaxation.ase_utils as ase_utils
        import fairchem.core.trainers.ocp_trainer  # noqa: F401
        from fairchem.core import OCPCalculator

        # The generic importer loads every bundled architecture, including optional
        # torch-sparse models. This checkpoint's model is registered explicitly above.
        ase_utils.setup_imports = lambda: None
        return lambda: OCPCalculator(
            checkpoint_path=checkpoint,
            cpu=device == "cpu",
            seed=int(model.get("seed", 42)),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _equflash_batch_predictions(
    atoms_list: list[Any], calculator: Any, max_atoms_per_batch: int
) -> tuple[dict[str, tuple[float, np.ndarray]], dict[str, str]]:
    """Predict EquFlash energies and forces in bounded, failure-isolating batches."""
    from fairchem.core.datasets import data_list_collater

    predictions: dict[str, tuple[float, np.ndarray]] = {}
    failures: dict[str, str] = {}

    def predict_batch(batch_atoms: list[Any]) -> None:
        try:
            graphs = [calculator.a2g.convert(atoms) for atoms in batch_atoms]
            batch = data_list_collater(graphs, otf_graph=True)
            result = calculator.trainer.predict(batch, per_image=False, disable_tqdm=True)
            energies = np.asarray(result["energy"].detach().cpu(), dtype=float).reshape(-1)
            forces = np.asarray(result["forces"].detach().cpu(), dtype=float)
            if len(energies) != len(batch_atoms):
                raise ValueError(f"Expected {len(batch_atoms)} energies, received {len(energies)}")
            offset = 0
            for atoms, energy in zip(batch_atoms, energies, strict=True):
                stop = offset + len(atoms)
                predictions[str(atoms.info["sample_id"])] = (energy, forces[offset:stop])
                offset = stop
            if offset != len(forces):
                raise ValueError(f"Unassigned force rows: {len(forces) - offset}")
        except Exception as exc:
            if len(batch_atoms) > 1:
                midpoint = len(batch_atoms) // 2
                predict_batch(batch_atoms[:midpoint])
                predict_batch(batch_atoms[midpoint:])
            else:
                sample_id = str(batch_atoms[0].info["sample_id"])
                failures[sample_id] = f"{type(exc).__name__}:{exc}"

    batch: list[Any] = []
    atom_count = 0
    for atoms in atoms_list:
        if batch and atom_count + len(atoms) > max_atoms_per_batch:
            predict_batch(batch)
            batch = []
            atom_count = 0
        batch.append(atoms)
        atom_count += len(atoms)
    if batch:
        predict_batch(batch)
    return predictions, failures


class InfrastructureError(RuntimeError):
    """Error that invalidates the current GPU process rather than the structure."""


def _is_infrastructure_error(error: BaseException | str) -> bool:
    text = str(error).lower()
    markers = (
        "out of memory",
        "failed to allocate",
        "cuda error",
        "cuda runtime",
        "device-side assert",
        "illegal memory access",
        "cublas",
        "cudnn",
        "warp cuda",
        "driver shutting down",
    )
    return any(marker in text for marker in markers)


def _phonon_from_relaxed(
    relaxed: Any,
    calculator: Any,
    settings: dict[str, Any],
) -> dict[str, Any]:
    from ase import Atoms
    from phonopy import Phonopy

    primitive = settings.get("primitive_matrix")
    phonon = Phonopy(
        _phonopy_atoms(relaxed),
        supercell_matrix=np.asarray(settings["supercell_matrix"], dtype=int),
        primitive_matrix=primitive,
        symprec=float(settings["phonopy_symprec"]),
    )
    phonon.generate_displacements(
        distance=float(settings["displacement_angstrom"]), is_diagonal=False
    )
    displaced_structures = []
    sample_ids = []
    for index, supercell in enumerate(phonon.supercells_with_displacements):
        sample_id = f"displacement-{index:06d}"
        displaced = Atoms(
            symbols=supercell.symbols,
            cell=supercell.cell,
            scaled_positions=supercell.scaled_positions,
            pbc=True,
        )
        displaced.info["sample_id"] = sample_id
        displaced_structures.append(displaced)
        sample_ids.append(sample_id)
    predictions, failures = _equflash_batch_predictions(
        displaced_structures,
        calculator,
        max_atoms_per_batch=int(settings.get("phonon_max_atoms_per_batch", 1)),
    )
    if failures:
        error = "; ".join(f"{key}:{value}" for key, value in sorted(failures.items()))
        if _is_infrastructure_error(error):
            raise InfrastructureError(error)
        raise ValueError(f"finite-displacement force prediction failed: {error}")
    force_sets = []
    for sample_id in sample_ids:
        forces = np.asarray(predictions[sample_id][1], dtype=float)
        forces -= forces.mean(axis=0, keepdims=True)
        force_sets.append(forces)
    phonon.forces = force_sets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    phonon.run_mesh(
        [int(value) for value in settings["mesh"]],
        with_eigenvectors=False,
        is_mesh_symmetry=True,
    )
    frequencies = np.asarray(phonon.get_mesh_dict()["frequencies"], dtype=float)
    if not frequencies.size or not np.isfinite(frequencies).all():
        raise ValueError("phonon frequencies are empty or nonfinite")
    return {
        "min_frequency_thz": float(np.min(frequencies)),
        "imaginary_mode_fraction": float(np.mean(frequencies < 0.0)),
        "materially_imaginary_mode_fraction": float(np.mean(frequencies < -0.30)),
        "num_displacements": len(force_sets),
        "supercell_atoms": len(phonon.supercell),
    }


class ProperStructureMatcher(StructureMatcher):
    """StructureMatcher restricted to orientation-preserving isometries.

    Pymatgen's matcher intentionally permits improper rotations.  That is
    appropriate for many duplicate checks but collapses enantiomers.  This
    override retains only lattice alignments whose Cartesian orthogonal map
    has positive determinant.  It intentionally pins a small private pymatgen
    hook; the package version is recorded in the output manifest and covered by
    regression tests.
    """

    def _get_lattices(self, target_lattice, structure, supercell_size=1):
        mappings = structure.lattice.find_all_mappings(
            target_lattice,
            ltol=self.ltol,
            atol=self.angle_tol,
            skip_rotation_matrix=False,
        )
        for lattice, rotation_matrix, scale_matrix in mappings:
            if not math.isclose(
                abs(float(np.linalg.det(scale_matrix))),
                supercell_size,
                abs_tol=0.5,
                rel_tol=0,
            ):
                continue
            if rotation_matrix is None or float(np.linalg.det(rotation_matrix)) <= 0:
                continue
            yield lattice, scale_matrix

    def fit(
        self,
        struct1,
        struct2,
        symmetric: bool = False,
        skip_structure_reduction: bool = True,
    ) -> bool:
        # The Crystoria inputs are already primitive and canonically reduced.
        # Re-running Niggli/primitive reduction can itself choose an
        # orientation-reversing basis before the determinant gate, erasing the
        # distinction this matcher is designed to preserve.
        return super().fit(
            struct1,
            struct2,
            symmetric=symmetric,
            skip_structure_reduction=skip_structure_reduction,
        )


def structure_from_canonical_json(payload: str):
    from pymatgen.core import Structure

    document = json.loads(payload)
    return Structure(
        document["lattice_matrix_angstrom"],
        document["atomic_numbers"],
        document["fractional_coordinates"],
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def invert_fractional_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Apply spatial inversion while retaining the same Bravais lattice."""

    values = np.mod(-np.asarray(coordinates, dtype=np.float64), 1.0)
    values[np.isclose(values, 1.0, atol=1e-12, rtol=0.0)] = 0.0
    return values


def inverted_structure(structure):
    from pymatgen.core import Structure

    return Structure(
        structure.lattice,
        structure.species,
        invert_fractional_coordinates(structure.frac_coords),
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def _build_repaired_phonopy(
    relaxed: Any, protocol: dict[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    """Apply the frozen primary -> 7-digit -> 6-digit Phonopy route.

    The repair is limited to sub-microangstrom lattice-component rounding and
    requires the detected space group to remain unchanged.
    """

    decimals_sequence = [
        int(value) for value in protocol["niggli_retry_lattice_round_decimals_sequence"]
    ]
    require(decimals_sequence == [7, 6], "unexpected Niggli retry sequence")
    last_error: Exception | None = None
    original_cell = np.asarray(relaxed.cell.array, dtype=np.float64)
    original_positions = np.asarray(relaxed.positions, dtype=np.float64)
    spacegroup_before = _spacegroup_number(relaxed, symprec=float(protocol["phonopy_symprec"]))
    for decimals in (None, *decimals_sequence):
        conditioned = relaxed.copy()
        if decimals is None:
            maximum_cell_change = 0.0
            maximum_position_change = 0.0
            route = "primary"
        else:
            rounded_cell = np.round(original_cell, decimals=int(decimals))
            maximum_cell_change = float(np.max(np.abs(rounded_cell - original_cell)))
            allowed_change = 0.500001 * 10.0 ** (-int(decimals))
            require(
                maximum_cell_change <= allowed_change,
                "Niggli retry lattice perturbation exceeds the declared rounding bound",
            )
            require(
                np.isfinite(np.linalg.det(rounded_cell)) and np.linalg.det(rounded_cell) > 0.0,
                "Niggli retry produced a non-positive lattice volume",
            )
            conditioned.set_cell(rounded_cell, scale_atoms=True)
            maximum_position_change = float(
                np.max(
                    np.linalg.norm(
                        np.asarray(conditioned.positions, dtype=np.float64) - original_positions,
                        axis=1,
                    )
                )
            )
            route = f"round_{decimals}"
        spacegroup_after = _spacegroup_number(
            conditioned, symprec=float(protocol["phonopy_symprec"])
        )
        require(
            spacegroup_after == spacegroup_before,
            "Niggli retry changed the detected space group: "
            f"{spacegroup_before} -> {spacegroup_after}",
        )
        try:
            phonon, _, _ = _build_phonopy(
                relaxed=conditioned,
                supercell_matrix=protocol["supercell_matrix"],
                primitive_matrix=protocol["primitive_matrix"],
                phonopy_symprec=float(protocol["phonopy_symprec"]),
                niggli_retry_lattice_round_decimals=None,
            )
            return (
                phonon,
                conditioned,
                {
                    "niggli_retry_used": decimals is not None,
                    "niggli_retry_lattice_round_decimals": decimals,
                    "niggli_retry_max_cell_component_change_angstrom": (maximum_cell_change),
                    "niggli_retry_max_cartesian_position_change_angstrom": (
                        maximum_position_change
                    ),
                    "niggli_retry_spacegroup_before": spacegroup_before,
                    "niggli_retry_spacegroup_after": spacegroup_after,
                    "repair_route": route,
                },
            )
        except Exception as exc:
            if "Niggli reduction failed" not in str(exc):
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


def _ir_mesh_qpoints(phonon: Any, protocol: dict[str, Any]) -> np.ndarray:
    """Return the configured regular-mesh q-points without BZ relocation."""

    if (
        protocol["mesh_backend"]
        == "time_reversal_paired_regular_mesh_no_bz_relocation"
    ):
        return _time_reversal_mesh_qpoints(protocol)
    require(
        protocol["mesh_backend"] == "spglib_ir_reciprocal_mesh_no_bz_relocation",
        "unsupported reciprocal-mesh backend",
    )
    return _spglib_ir_mesh_qpoints(phonon, protocol)


def _spglib_ir_mesh_qpoints(phonon: Any, protocol: dict[str, Any]) -> np.ndarray:
    """Return point-group-reduced regular-mesh q-points without BZ relocation.

    Phonopy's BZ relocation performs another Niggli reduction after force
    constants are built.  The failed Crystoria cells sit on numerical Niggli
    boundaries, so this repair path obtains the same regular-grid irreducible
    representatives directly from spglib and evaluates them with
    ``run_qpoints``.  Reciprocal-lattice periodicity makes relocation into the
    first Brillouin zone unnecessary for the minimum-frequency statistic.
    """

    import spglib

    mesh = np.asarray(protocol["mesh"], dtype=np.int64)
    require(mesh.shape == (3,) and np.all(mesh > 0), "invalid phonon mesh")
    is_shift = np.asarray([int(value % 2 == 0) for value in mesh], dtype=np.int64)
    primitive = phonon.primitive
    mapping, addresses = spglib.get_ir_reciprocal_mesh(
        mesh,
        (
            np.asarray(primitive.cell, dtype=np.float64),
            np.asarray(primitive.scaled_positions, dtype=np.float64),
            np.asarray(primitive.numbers, dtype=np.int64),
        ),
        is_shift=is_shift,
        is_time_reversal=True,
        symprec=float(protocol["phonopy_symprec"]),
    )
    representatives = np.unique(np.asarray(mapping, dtype=np.int64))
    qpoints = (np.asarray(addresses[representatives], dtype=np.float64) + 0.5 * is_shift) / mesh
    require(
        0 < len(qpoints) <= int(np.prod(mesh)) and np.isfinite(qpoints).all(),
        "invalid irreducible reciprocal mesh",
    )
    return np.ascontiguousarray(qpoints, dtype=np.float64)


def _time_reversal_mesh_qpoints(protocol: dict[str, Any]) -> np.ndarray:
    """Return one representative from every q/-q pair on the MP mesh.

    This backend deliberately applies no point-group reduction.  It mirrors
    Phonopy's safe fallback when a half-grid shift is incompatible with the
    primitive-cell point group, while avoiding the first-Brillouin-zone
    relocation whose reciprocal-space Niggli reduction caused the frozen
    Crystoria failures.
    """

    mesh = np.asarray(protocol["mesh"], dtype=np.int64)
    require(mesh.shape == (3,) and np.all(mesh > 0), "invalid phonon mesh")
    is_shift = np.asarray([int(value % 2 == 0) for value in mesh], dtype=np.int64)
    addresses = np.indices(tuple(int(value) for value in mesh), dtype=np.int64)
    addresses = addresses.reshape(3, -1).T
    partner_addresses = np.mod(-addresses - is_shift, mesh)
    strides = np.asarray([mesh[1] * mesh[2], mesh[2], 1], dtype=np.int64)
    indices = addresses @ strides
    partner_indices = partner_addresses @ strides
    representatives = addresses[indices <= partner_indices]
    qpoints = (representatives.astype(np.float64) + 0.5 * is_shift) / mesh
    expected_pairs = (int(np.prod(mesh)) + int(np.sum(indices == partner_indices))) // 2
    require(len(qpoints) == expected_pairs, "time-reversal mesh pairing drift")
    require(np.isfinite(qpoints).all(), "nonfinite reciprocal mesh")
    return np.ascontiguousarray(qpoints, dtype=np.float64)


def _evaluate_endpoint(atoms: Any, calculator: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    from ase import Atoms

    phonon, conditioned, metadata = _build_repaired_phonopy(atoms, protocol)
    phonon.generate_displacements(
        distance=float(protocol["displacement_angstrom"]), is_diagonal=False
    )
    displaced_structures = []
    sample_ids = []
    for index, supercell in enumerate(phonon.supercells_with_displacements):
        sample_id = f"displacement-{index:06d}"
        displaced = Atoms(
            symbols=supercell.symbols,
            cell=supercell.cell,
            scaled_positions=supercell.scaled_positions,
            pbc=True,
        )
        displaced.info["sample_id"] = sample_id
        displaced_structures.append(displaced)
        sample_ids.append(sample_id)
    predictions, failures = _equflash_batch_predictions(
        displaced_structures,
        calculator,
        max_atoms_per_batch=int(protocol["phonon_max_atoms_per_batch"]),
    )
    if failures:
        error = "; ".join(f"{key}:{value}" for key, value in sorted(failures.items()))
        if _is_infrastructure_error(error):
            raise InfrastructureError(error)
        raise ValueError(f"finite-displacement force prediction failed: {error}")
    force_sets = []
    for sample_id in sample_ids:
        forces = np.asarray(predictions[sample_id][1], dtype=float)
        forces -= forces.mean(axis=0, keepdims=True)
        force_sets.append(forces)
    phonon.forces = force_sets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    qpoints = _ir_mesh_qpoints(phonon, protocol)
    phonon.run_qpoints(qpoints, with_eigenvectors=False)
    frequencies = np.asarray(phonon.get_qpoints_dict()["frequencies"], dtype=float)
    require(frequencies.size > 0 and np.isfinite(frequencies).all(), "nonfinite frequencies")
    thresholds = PhononThresholds(
        stable_min_frequency_thz=float(protocol["thresholds"]["stable_min_frequency_thz"]),
        unstable_min_frequency_thz=float(protocol["thresholds"]["unstable_min_frequency_thz"]),
    )
    minimum = float(np.min(frequencies))
    return {
        "min_frequency_thz": minimum,
        "imaginary_mode_fraction": float(np.mean(frequencies < 0.0)),
        "materially_imaginary_mode_fraction": float(np.mean(frequencies < -0.30)),
        "num_displacements": len(force_sets),
        "supercell_atoms": len(phonon.supercell),
        "mesh_ir_qpoints": len(qpoints),
        "mesh_backend": str(protocol["mesh_backend"]),
        "phonon_class": classify_phonon_stability(minimum, thresholds),
        "conditioned_cell": np.asarray(conditioned.cell.array, dtype=np.float64),
        **metadata,
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def geometry_features(raw, settings):
    """Continuous candidate descriptors; only the strict matcher decides identity."""
    document = json.loads(raw)
    lattice = np.asarray(document["lattice_matrix_angstrom"], dtype=float)
    coordinates = np.asarray(document["fractional_coordinates"], dtype=float)
    numbers = np.asarray(document["atomic_numbers"], dtype=int)
    reduced = spglib.niggli_reduce(lattice, eps=1e-5)
    require(reduced is not None, "Niggli reduction failed")
    gram = reduced @ reduced.T
    lengths = np.sqrt(np.diag(gram))
    cosines = gram / np.outer(lengths, lengths)
    volume = abs(float(np.linalg.det(lattice)))
    distances = Lattice(lattice).get_all_distances(coordinates, coordinates)
    upper = distances[np.triu_indices(len(numbers), 1)]
    quantiles = np.quantile(upper, [0, .1, .25, .5, .75, .9, 1]) if len(upper) else np.zeros(7)
    free_length = (volume / len(numbers)) ** (1 / 3)
    features = np.r_[
        np.sort(np.log(lengths)) / settings["log_length_window"],
        np.sort(np.abs(cosines[np.triu_indices(3, 1)])) / settings["cosine_window"],
        np.log(volume) / settings["log_volume_window"],
        quantiles / free_length / settings["normalized_distance_window"],
    ]
    require(np.isfinite(features).all(), "nonfinite geometry descriptor")
    elements, counts = np.unique(numbers, return_counts=True)
    composition = "|".join(f"{z}:{count}" for z, count in zip(elements, counts, strict=True))
    return composition, features.tolist()


def continuous_candidates(features, maximum_pairs):
    """Exact Chebyshev radius search, without quantization-bin boundaries."""
    grouped = features.with_columns(pl.len().over("composition").alias("_size")).filter(pl.col("_size") > 1)
    frames, total = [], 0
    for group in grouped.partition_by("composition", maintain_order=False):
        values = np.asarray(group["features"].to_list(), dtype=float)
        pairs = cKDTree(values).query_pairs(1.0, p=np.inf, eps=0, output_type="ndarray")
        if not len(pairs):
            continue
        index = group["record_index"].to_numpy()
        left, right = index[pairs[:, 0]], index[pairs[:, 1]]
        frames.append(pl.DataFrame({"left_index": np.minimum(left, right),
                                    "right_index": np.maximum(left, right)}))
        total += len(pairs)
        require(total <= maximum_pairs, "candidate safety limit exceeded; no candidate pairs were silently discarded")
    if not frames:
        return pl.DataFrame(schema={"left_index": pl.UInt32, "right_index": pl.UInt32})
    return pl.concat(frames).cast({"left_index": pl.UInt32, "right_index": pl.UInt32}).unique().sort("left_index", "right_index")


def component_roots(size, edges):
    if not len(edges):
        return np.arange(size, dtype=np.int64)
    matrix = coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(size, size)).tocsr()
    count, labels = connected_components(matrix, directed=False)
    roots = np.full(count, size, dtype=np.int64)
    np.minimum.at(roots, labels, np.arange(size))
    return roots[labels]


def direct_representatives(size, edges, derived):
    """Greedy direct-match representatives avoid merging a nontransitive chain."""
    representative = np.arange(size, dtype=np.int64)
    observed_edges = edges[~derived[edges[:, 0]] & ~derived[edges[:, 1]]]
    if not len(observed_edges):
        return representative
    both = np.concatenate([observed_edges, observed_edges[:, ::-1]])
    graph = coo_matrix((np.ones(len(both), dtype=np.int8), (both[:, 0], both[:, 1])), shape=(size, size)).tocsr()
    for node in np.flatnonzero(np.diff(graph.indptr)):
        if representative[node] != node:
            continue
        for other in graph.indices[graph.indptr[node]:graph.indptr[node + 1]]:
            if representative[other] == other:
                require(other > node, "representative order is not lexicographic")
                representative[other] = node
    return representative


def grouped_split(root_id, seed):
    token = f"crystoria-geometric-v1.2:{seed}:{root_id}"
    bucket = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16) / 16**16
    return "train" if bucket < .90 else "validation" if bucket < .95 else "test"


def _canonical_payload(structure: Structure, decimals: int) -> tuple[dict[str, Any], str]:
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    coords = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    numbers = np.asarray([site.specie.Z for site in structure], dtype=int)

    # Remove arbitrary origin choices by selecting the lexicographically smallest
    # site-anchored representation. Niggli reduction upstream removes cell-basis noise.
    candidates: list[tuple[str, list[int], list[list[float]]]] = []
    for anchor in range(len(structure)):
        shifted = np.mod(coords - coords[anchor], 1.0)
        shifted[np.isclose(shifted, 1.0, atol=10 ** (-decimals))] = 0.0
        rounded = np.round(shifted, decimals)
        order = sorted(
            range(len(structure)),
            key=lambda i: (int(numbers[i]), *rounded[i].tolist()),
        )
        candidate_numbers = [int(numbers[i]) for i in order]
        candidate_coords = [rounded[i].tolist() for i in order]
        candidate_key = json.dumps([candidate_numbers, candidate_coords], separators=(",", ":"))
        candidates.append((candidate_key, candidate_numbers, candidate_coords))
    _, canonical_numbers, canonical_coords = min(candidates, key=lambda item: item[0])
    canonical = {
        "atomic_numbers": canonical_numbers,
        "lattice_matrix_angstrom": np.round(lattice, decimals).tolist(),
        "fractional_coordinates": canonical_coords,
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return canonical, hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_structure(
    structure: Structure,
    *,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
    minimum_distance: float,
    source_num_sites_parse_max: int,
    primitive_num_sites_max: int,
) -> dict[str, Any]:
    if not structure.is_ordered:
        raise ValueError("disordered_structure")
    if len(structure) > source_num_sites_parse_max:
        raise ValueError("source_num_sites_exceeds_parse_limit")
    if not np.isfinite(structure.volume) or structure.volume <= 0:
        raise ValueError("nonpositive_or_nonfinite_volume")

    analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    primitive = analyzer.find_primitive()
    if primitive is None:
        primitive = structure.get_primitive_structure(tolerance=symprec)
    primitive = primitive.get_reduced_structure(reduction_algo="niggli")
    if len(primitive) > primitive_num_sites_max:
        raise ValueError("primitive_num_sites_exceeds_limit")

    distance_matrix = np.asarray(primitive.distance_matrix)
    if len(primitive) > 1:
        distance_matrix[distance_matrix < 1.0e-12] = np.inf
        min_distance = float(np.min(distance_matrix))
    else:
        min_distance = float(min(primitive.lattice.abc))
    if not math.isfinite(min_distance) or min_distance < minimum_distance:
        raise ValueError("minimum_distance_below_limit")

    canonical, signature = _canonical_payload(primitive, decimals)
    refined = SpacegroupAnalyzer(primitive, symprec=symprec, angle_tolerance=angle_tolerance)
    elements = sorted({str(site.specie.symbol) for site in primitive})
    return {
        "canonical_structure_id": f"canonical:{signature}",
        "canonical_signature_sha256": signature,
        "canonical_structure_json": json.dumps(canonical, sort_keys=True, separators=(",", ":")),
        "governed_leakage_group": None,
        "source_structure_locator": None,
        "formula": primitive.composition.formula,
        "reduced_formula": primitive.composition.reduced_formula,
        "chemical_system": "-".join(elements),
        "elements_json": json.dumps(elements, separators=(",", ":")),
        "source_num_sites": int(len(structure)),
        "primitive_num_sites": int(len(primitive)),
        "volume_per_atom_angstrom3": float(primitive.volume / len(primitive)),
        "minimum_distance_angstrom": min_distance,
        "spacegroup_number": int(refined.get_space_group_number()),
        "spacegroup_symbol": str(refined.get_space_group_symbol()),
        "crystal_system": str(refined.get_crystal_system()),
        "point_group": str(refined.get_point_group_symbol()),
        "structure_valid": True,
        "canonicalization_status": "passed_primitive_niggli",
        "canonicalization_error": None,
    }


def load_elemental_reference_energies(path: Path) -> dict[str, float]:
    """Load the frozen minimum-energy MP elemental entries."""
    from pymatgen.entries.computed_entries import ComputedEntry

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        entries = json.load(handle)
    return {
        symbol: float(ComputedEntry.from_dict(entry).energy_per_atom)
        for symbol, entry in entries.items()
    }


def elemental_reference_energy(composition: Any, references: dict[str, float]) -> float:
    """Return the total elemental reference energy for a composition."""
    missing = {str(element) for element in composition.elements} - set(references)
    if missing:
        raise ValueError(f"Missing MP elemental references for {sorted(missing)}")
    return float(sum(references[str(element)] * amount for element, amount in composition.items()))


def formation_energy_per_atom(
    *, total_energy_ev: float, correction_ev: float, composition: Any, references: dict[str, float]
) -> float:
    """Convert an absolute energy to MP2020-corrected formation energy."""
    return float(
        (total_energy_ev + correction_ev - elemental_reference_energy(composition, references))
        / composition.num_atoms
    )


def mp2020_correction(structure: Any) -> tuple[float, dict[str, Any]]:
    """Apply the Matbench MP2020 correction protocol to a relaxed structure."""
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from pymatgen.io.vasp.sets import MPRelaxSet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_set = MPRelaxSet(structure)
        incar = input_set.incar
    symbols = list(input_set.poscar.site_symbols)
    values = [float(value) for value in incar.get("LDAUU", [0.0] * len(symbols))]
    hubbards = dict(zip(symbols, values, strict=True))
    is_hubbard = bool(incar.get("LDAU", False) and any(value > 0 for value in values))
    parameters = {
        "run_type": "GGA+U" if is_hubbard else "GGA",
        "is_hubbard": is_hubbard,
        "hubbards": hubbards,
    }
    entry = ComputedStructureEntry(structure, 0.0, parameters=parameters)
    compatibility = MaterialsProject2020Compatibility(check_potcar=False, strict_anions="no_check")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        processed = compatibility.process_entry(entry, clean=True)
    if processed is None:
        raise ValueError(f"MP2020 compatibility rejected {structure.composition}")
    details = {
        "run_type": parameters["run_type"],
        "hubbards": hubbards,
        "adjustments": [str(adjustment) for adjustment in processed.energy_adjustments],
    }
    return float(processed.correction), details


def entry_from_formation_energy(
    composition: Any, formation_energy_ev_atom: float, references: dict[str, float]
) -> Any:
    """Construct an absolute-energy entry on the frozen MP elemental scale."""
    from pymatgen.analysis.phase_diagram import PDEntry

    energy = formation_energy_ev_atom * composition.num_atoms + elemental_reference_energy(
        composition, references
    )
    return PDEntry(composition, energy)


def fixed_mp_hull_distance(
    phase_diagram: Any,
    composition: Any,
    formation_energy_ev_atom: float,
    references: dict[str, float],
) -> tuple[float, bool]:
    """Return signed distance to the fixed MP DFT hull and fallback status."""
    entry = entry_from_formation_energy(composition, formation_energy_ev_atom, references)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hull = phase_diagram.get_e_above_hull(entry, allow_negative=True)
    fallback = any("No suitable PhaseDiagrams" in str(item.message) for item in caught)
    return float(hull), fallback


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Return stable-positive binary metrics with explicit confusion counts."""
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be aligned one-dimensional arrays")
    true_stable = int(np.sum(labels & predictions))
    false_unstable = int(np.sum(labels & ~predictions))
    true_unstable = int(np.sum(~labels & ~predictions))
    false_stable = int(np.sum(~labels & predictions))
    stable_total = true_stable + false_unstable
    unstable_total = true_unstable + false_stable
    if not stable_total or not unstable_total:
        raise ValueError("both truth classes are required")
    stable_recall = true_stable / stable_total
    unstable_recall = true_unstable / unstable_total
    denominator = math.sqrt(
        (true_stable + false_stable)
        * stable_total
        * unstable_total
        * (true_unstable + false_unstable)
    )
    return {
        "n": int(labels.size),
        "true_stable": true_stable,
        "false_unstable": false_unstable,
        "true_unstable": true_unstable,
        "false_stable": false_stable,
        "accuracy": (true_stable + true_unstable) / labels.size,
        "balanced_accuracy": 0.5 * (stable_recall + unstable_recall),
        "stable_recall": stable_recall,
        "unstable_recall": unstable_recall,
        "false_unstable_rate": false_unstable / stable_total,
        "false_stable_rate": false_stable / unstable_total,
        "precision_stable": true_stable / (true_stable + false_stable),
        "matthews_correlation_coefficient": (
            (true_stable * true_unstable - false_stable * false_unstable) / denominator
            if denominator
            else None
        ),
    }


def _classification_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    return {
        "true_stable": int(np.sum(labels & predictions)),
        "false_unstable": int(np.sum(labels & ~predictions)),
        "true_unstable": int(np.sum(~labels & ~predictions)),
        "false_stable": int(np.sum(~labels & predictions)),
    }


def _bootstrap_balanced_accuracy(
    labels: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    clusters: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        group_indices[str(cluster)].append(index)
    groups = list(group_indices.values())
    contributions: dict[str, np.ndarray] = {}
    for model_name, predictions in predictions_by_model.items():
        rows = []
        for indices in groups:
            counts = _classification_counts(labels[indices], predictions[indices])
            rows.append(
                [
                    counts["true_stable"],
                    counts["false_unstable"],
                    counts["true_unstable"],
                    counts["false_stable"],
                ]
            )
        contributions[model_name] = np.asarray(rows, dtype=np.int64)
    generator = np.random.default_rng(seed)
    estimates = {name: np.empty(replicates, dtype=float) for name in predictions_by_model}
    names = list(predictions_by_model)
    for start in range(0, replicates, 64):
        stop = min(start + 64, replicates)
        sampled = generator.integers(0, len(groups), size=(stop - start, len(groups)))
        for name in names:
            totals = contributions[name][sampled].sum(axis=1)
            stable_recall = totals[:, 0] / np.maximum(totals[:, 0] + totals[:, 1], 1)
            unstable_recall = totals[:, 2] / np.maximum(totals[:, 2] + totals[:, 3], 1)
            estimates[name][start:stop] = 0.5 * (stable_recall + unstable_recall)
    payload: dict[str, Any] = {}
    for name, values in estimates.items():
        payload[name] = {
            "balanced_accuracy_95_ci": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
        }
    if len(names) == 2:
        difference = estimates[names[0]] - estimates[names[1]]
        payload["paired_difference"] = {
            "order": [names[0], names[1]],
            "balanced_accuracy_difference_95_ci": [
                float(np.quantile(difference, 0.025)),
                float(np.quantile(difference, 0.975)),
            ],
        }
    return payload


def _mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    from scipy.stats import binomtest

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    pvalue = float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant": discordant,
        "two_sided_exact_pvalue": pvalue,
    }
