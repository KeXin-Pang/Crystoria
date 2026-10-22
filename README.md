# Crystoria

Data loading, crystal curation, lattice-dynamical calculation and analysis methods accompanying **Crystoria: A Large-Scale Crystal Dataset Linking Thermodynamic and Dynamical Stability for Materials AI**.

Version 1.0.0 packages the scientific data and calculation functions of the 1.0.0-rc4 submission candidate. The eight scientific Python entrypoints are unchanged. The complete dataset contains 2,649,640 structures; it is distributed separately from this code repository.

Dataset and code archive DOI: https://doi.org/10.5281/zenodo.23572554

## Installation and data access

Use Python 3.12 and a dedicated environment. Extract the accompanying data archive as `Crystoria-1.0.0`; its `02-methods` directory contains this code. Run from that directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-data.txt
python crystoria.py verify ..
python crystoria.py stats --data ../01-data
python -m unittest discover -s tests -v
```

For a separate checkout, replace `..`, `../01-data` and `../03-evidence` with the corresponding paths in the extracted dataset. Consult `CITATION.cff` and the repository record for version identifiers.

## Reuse and numerical reproduction

```sh
python assemble_dataset.py --data ../01-data --output joined-labels.parquet
python reproduce_figures.py --data ../01-data --evidence ../03-evidence --output summaries
python reproduce_symmetry.py --results ../03-evidence/figure-source-data/symmetry --output symmetry-summary
python benchmark_metrics.py ../03-evidence/validation/pbe_full_reference_predictions.parquet --output benchmark.json
```

`assemble_dataset.py` joins tables on `record_id`; `--geometry source` or `--geometry both` includes coordinates. Optional grouped splits are reuse aids. Replace `RECORD_ID` below with an ID from `examples/record.json`:

```sh
python crystoria.py export --data ../01-data --record-id RECORD_ID --geometry relaxed --output structure.cif
python crystoria.py canonicalize structure.cif standardized.json
python crystoria.py invert examples/structure.json inverted.json
python crystoria.py select --data ../01-data --ordinary-only --output stable-selection.parquet
python crystoria.py energetics --data ../01-data --record-id RECORD_ID --output energetics.json
```

`energetics` recomputes MP2020 alignment and the frozen-reference hull from archived energy and coordinates. It does not run the force model. The bundled phase-diagram reference is an upstream pickle; the program checks its pinned SHA256 before deserialization. Use only the checksum-verified reference.

## Force-model calculations

Use a separate model environment. `requirements-model-observed.txt` and `configs/` record the tested worker settings; they are not a universal CUDA installation lock. Install the pinned GGNN revision and dependencies, apply the separately licensed compatibility patch, download the official checkpoint identified in `configs/model.json`, and verify its SHA256.

```sh
python crystoria.py calculate --data ../01-data --record-id RECORD_ID --checkpoint /path/to/equflashv2-45M-oam.pt --device cuda --output recalculation.json
```

The default uses the record's archived phonon protocol. `--relax` starts a new relaxation. `--protocol production` and `--protocol replay` select specific protocols. Generated inversion partners inherit scalar quantities from their actual calculation parents.

`Methods.md` describes the physical methods. `REPRODUCTION.md` documents twelve archived representative source-isolated model recalculations in the recorded worker environment. These do not establish a full-population rerun, a fresh model-environment installation, or per-record DFT validation. `FIGURE_REPRODUCTION.md` maps the released figure inputs to numerical reproduction.

## Licenses and citation

Original Crystoria code is licensed under MIT; see `LICENSE`. The GGNN compatibility patch and upstream notice retain **CC BY-NC-SA 4.0** and are excluded from the MIT grant. The complete calculation stack therefore has a noncommercial third-party component. The official EquFlashV2 checkpoint is separately declared CC BY 4.0 in Figshare record 10.6084/m9.figshare.32630910.v4; checkpoint weights and upstream GGNN source are not bundled.

Data and example coordinates retain the terms and attribution in the data archive's `LICENSES/`; the code license does not replace them. See `THIRD_PARTY_NOTICES.md`. Cite the exact data and code versions used.

Code repository: https://github.com/KeXin-Pang/Crystoria
