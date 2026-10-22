# Structure-rendering extension

`publication-style.mjs` records the MatterViz 0.4.1 material and bond-style extension used for the paper's structure illustrations. `package.json` and `pnpm-lock.yaml` record the associated dependencies. This directory is an extension reference, not a standalone application.

The `StructureScene.svelte` and `Bond.svelte` transforms require upstream source files with the SHA256 values in `UPSTREAM_HASHES`. Host-specific application integration, integration tests and application build commands are omitted.

Complete native renders, POSCAR coordinates, neighbor bonds and camera/material settings are supplied in `03-evidence/figure-source-data/Fig1-assets` in the data archive. Numerical figure reproduction uses the Python programs described in `FIGURE_REPRODUCTION.md`.
