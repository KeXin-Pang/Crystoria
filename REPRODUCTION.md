# Independent public-method recalculation

Twelve declared representative calculations passed: nine fixed relaxed geometries, one Si source workflow, and two further source workflows with nonzero lattice relaxation (Te and SmZnSn). Stable, borderline and unstable cases cover both archived phonon protocols. A Te inversion pair was recalculated independently rather than assigning inherited scalar results. A separate Si test also executed the unchanged public crystoria.py calculate command.

The distributed independent_recalculation_worker.py imports the same public production functions and reuses one calculator instance; it does not read archived labels to manufacture predictions. Predictions are compared with the predeclared expected values only in the external audit.

Use the pinned model runtime and GGNN patch described in README.md. Then run:

```sh
python independent_recalculation_worker.py --data ../01-data --methods . --cases ../03-evidence/validation/public-method-recalculation/cohort.json --checkpoint /path/to/equflashv2-45M-oam.pt --output recalculated
```

For the two nontrivial source relaxations, use crystoria.py calculate with each record ID from nontrivial_relaxation_cohort.json, --geometry source --relax and the same checkpoint. The source cell must actually change, residual forces must be <=0.03 eV/angstrom, and the class must agree. A 0.01 THz tolerance was declared for these fresh relaxations; fixed-geometry frequency tolerance was 0.001 THz. Fixed-geometry energy and force tolerances were 0.0001 eV/atom and 0.0001 eV/angstrom, respectively.

Runtime versions, exact record IDs, predictions, differences and all pass/fail checks are in the validation directory. These tests validate source-isolated method reproducibility in the recorded existing environment, not a fresh installation, a new DFT validation or a rerun of all 2,649,640 structures. Stored scientific values have not been replaced.
