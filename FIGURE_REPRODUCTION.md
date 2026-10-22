# Figure evidence and reproduction

The scientific population is fixed at 2,649,640. Separate original calculation parents and illustrative figure geometries are not added to that denominator.

| Figure | Supplied inputs and reusable calculation |
|---|---|
| Fig. 1a,b | Exact POSCARs, native PNGs and neighbor bonds in figure-source-data/Fig1-assets; rendering.json records camera, radii and materials. Equivalent source cells and a displaced supercell are figure examples, not new dataset rows. |
| Fig. 1c | Eight stable examples, Te inversion pair, SnO2 primitive cell and colors in the same directory. Source coordinates and scalar labels remain in 01-data. |
| Fig. 2 | reproduce_figures.py rebuilds element, chemical-order, space-group and primitive-size counts from source coordinates. Archived CSVs are in figure-source-data/coverage. |
| Fig. 3 | reproduce_figures.py rebuilds numeric grids, quantiles and display counts using screening_source_hull_mev_atom and the relaxed model hull. Both differ in purpose from minimum/maximum source-admission values. |
| Fig. S1 | identity/mirrors tables, chirality config and independent_inversion_pairs.parquet; generated scalar results are inherited. |
| Fig. S2 | source_thermodynamics.parquet, excluding generated mirrors and phase_diagram_fallback cases; hull_*_statistics.csv are the supplied summaries. |
| Fig. S3 | Both full and balanced prediction Parquets; benchmark_metrics.py reproduces grouped bootstrap intervals and exact paired tests. No original DFT outputs are redistributed. |

The portable diagnostic plots do not recreate publication typography or layout pixel-for-pixel. The MatterViz 0.4.1 material extension and dependency lock are supplied in figure-rendering; this is an extension, not a standalone web application. It uses glossy atoms, solid midpoint-split bond colors, the closed displayed cell and internal source-anchored neighbor segments. Geometry is not changed by styling.

## Added source-to-relaxed symmetry evidence (Fig. S4)

figure-source-data/symmetry/ contains all 2,649,640 paired assignments, the complete 230-by-230 transition matrix, crystal-system and group summaries. reproduce_symmetry.py independently aggregates these assignments and can reassign spglib symmetry for a deterministic coordinate sample. Common tolerances are 0.01 angstrom and 5 degrees. The symmetry assignments and numerical inputs are unchanged from the submission candidate.
