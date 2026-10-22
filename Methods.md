# Complete physical methods

 ## Supplementary Methods S1. Crystal Structures and Lattice Dynamics

### S1.1 Crystal collection and physical domain

Crystoria contains 2,649,640 ordered, three-dimensional crystal structures, with 31,997,598 atoms in their primitive cells. The selected cells contain 1–48 atoms, have supported atomic species with $Z\leq100$, and have minimum interatomic separations of at least 0.50 Å. Source attribution and the conditions of the original thermodynamic calculations accompany each structure.

The contributing collections are Alexandria, Materials Project, JARVIS, NIMS HeuslerDB, and OQMD. Some Alexandria and Materials Project data are obtained through the collections distributed with MatterGen, including its DFT relaxation and energy processing [@Zeni2025MatterGen]. A crystal appearing in several collections remains associated with its original sources.

Energy above the convex hull measures competition with other phases at the same overall composition. The source-energy selection requires every retained admissible estimate to be no greater than 100 meV atom$^{-1}$. The minimum and maximum admissible source estimates are retained separately, and the maximum controls the initial selection. This cutoff is applied to source estimates before relaxation. The source-to-relaxed comparison in Fig. 3 uses the source hull energy associated with each structure's original calculation.

### S1.2 Equivalent cells and independent comparisons

Different lattice bases, origins, or atomic orderings can describe the same periodic crystal. Primitive-cell reduction and symmetry analysis use pymatgen and spglib [@Ong2013Pymatgen; @togo2024spglib]. Initial symmetry processing uses a 0.01 Å positional tolerance and a $5^\circ$ angular tolerance, with fractional coordinates rounded to six decimal places.

Only structures with the same composition are compared. Candidate pairs are selected using reduced lattice lengths, angles, volume, and periodic interatomic distances. The allowed differences are 0.001 for sorted logarithmic lattice lengths and sorted absolute angle cosines, and 0.003 for logarithmic volume. Distance percentiles at 0, 10, 25, 50, 75, 90, and 100% are normalized by $(V/N)^{1/3}$, where $V$ is cell volume and $N$ is atom count; their comparison window is 0.0025.

Equivalence requires a direct, species-preserving correspondence between periodic atomic positions. Relative lattice and normalized atomic-position tolerances are $10^{-4}$, with a $0.01^\circ$ angle tolerance. The comparison permits proper rotations but not reflection, lattice scaling, additional primitive reduction, or supercell matching. Representative selection does not depend on the phonon class or other property values.

### S1.3 Relaxed structures and thermodynamic quantities

Both the cell and atomic positions are relaxed with the same EquFlashV2 interatomic potential, using a maximum-force criterion of 0.03 eV Å$^{-1}$ and at most 200 steps. The model is the 45-million-parameter version compared with density-functional theory (DFT) in Figure S3.

For the relaxed structures, EquFlashV2 single-point energies are corrected consistently with the Materials Project 2020 (MP2020) scheme and compared with a fixed reference phase diagram [@wang2021corrections]. Both signed and nonnegative distances from this reference hull are retained as outputs of the common model calculation. Source and relaxed-structure values remain attached to the atomic arrangements from which they were obtained. Their differences combine relaxation with changes in the interatomic potential, energy corrections, and reference phases. Figure S2 shows how these differences vary with composition, symmetry, and cell size.

### S1.4 Phonon calculations and stability margins

The harmonic response is calculated with Phonopy using atomic displacements of 0.01 Å in $2\times2\times2$ supercells [@Togo2015Phonons]. The mean force is removed from each displaced supercell, and the reconstructed force constants are symmetrized. Force changes are evaluated without separately subtracting the forces of the undisplaced supercell.

Frequencies are evaluated on a non-$\Gamma$-centred $12\times12\times12$ reciprocal-space mesh. We retain the uncorrected minimum over this mesh as a continuous quantity, with imaginary modes represented by negative values. A negative harmonic mode identifies an atomic displacement along which the calculated energy decreases near the relaxed structure.

The reported classes are stable for $\omega_{\min}\geq-0.10$ THz, borderline for $-0.30\leq\omega_{\min}<-0.10$ THz, and unstable for $\omega_{\min}<-0.30$ THz. These margins separate pronounced harmonic instabilities from nearly soft modes.

All structures included in the dataset have finite phonon results and converged parent relaxations.

### S1.5 Inversion partners and chirality

Spatial inversion generates the missing handed partner through $\mathbf{x}\mapsto-\mathbf{x}\pmod 1$, with the lattice and species order retained. Existing partners are connected rather than added again. The chirality assessment uses consistent source-cell symmetry assignments within the 65 Sohncke space-group types.

The dataset contains 133,195 explicitly generated inversion partners. They inherit the parent energy and phonon results as parity-even scalar quantities, which are unchanged by spatial inversion. Handedness-dependent quantities, such as optical-rotation sign or topological charge, are not inherited. Chirality counts refer to the source-cell symmetry, while phonon properties refer to the relaxed arrangements. Independently calculated inversion pairs provide a separate consistency test in Supplementary Methods S2.2 and Figure S1.

## Supplementary Methods S2. Reliability and Physical Interpretation

### S2.1 Crystal symmetry before and after relaxation

Space-group types are determined independently from the source and relaxed atomic coordinates using spglib, with a common positional tolerance of 0.01 Å and an angular tolerance of $5^\circ$. All 2,649,640 structures enter this comparison, including generated inversion partners.

The space-group type is unchanged for 2,624,127 structures (99.04%) and changes for 25,513 (0.96%). Agreement ranges from 95.82% for triclinic source structures to 99.95% for cubic source structures. Figure S4 gives the complete correspondence over space groups 1–230 and the population and agreement within each crystal system. Among structures retaining their space-group type, 710,222 (27.07%) are dynamically unstable. Symmetry retention and harmonic stability therefore describe different aspects of the relaxed lattice.

### S2.2 Inversion symmetry

Crystal energies and harmonic phonon frequencies are unchanged by spatial inversion. Among 209 independently calculated inversion pairs, 208 have the same phonon class. The largest minimum-frequency difference is 0.05444 THz. Figure S1 shows the corresponding symmetry populations and independent pair comparisons.

### S2.3 Comparison with DFT phonons

The external reference comprises 9,958 nonmagnetic semiconductors with DFT phonon calculations using the Perdew–Burke–Ernzerhof (PBE) approximation: 8,189 stable and 1,769 unstable [@Loew2025Phonons]. EquFlashV2 and MatterSim forces are evaluated on identical 0.01 Å displacements of the fixed DFT structures. The original supercell matrices and commensurate reciprocal-space points are retained. Mean-force removal and force-constant reconstruction are common to both models, and no model relaxation is applied.

The reference uses a numerical tolerance of $-35$ K for acoustic modes at $\Gamma$. This reference convention differs from the frequency margins and fixed reciprocal-space mesh used for Crystoria.

EquFlashV2 gives 95.11% accuracy and 91.22% balanced accuracy on the full reference set. Balanced accuracy is the mean recall of the stable and unstable classes, avoiding dominance by the more numerous stable structures. Stable recall is 97.25%, unstable recall is 85.19%, and stable precision is 96.81%.

The 95% confidence interval for balanced accuracy is 90.36–92.06%, obtained by resampling groups of identical crystal geometries. Differences between the two potentials are assessed with a paired exact McNemar test. Figure S3 also includes a balanced set of 2,688 structures with agreeing PBE and PBEsol classifications, and reports the model differences and their uncertainty. Overlap with the training data of either potential remains undetermined.

## Supplementary Methods S3. Physical Quantities and Use of the Data

### S3.1 Structural and energetic information

Each structure is described by its lattice vectors, atomic species, and fractional coordinates, together with composition, primitive-cell atom count, and source symmetry. The machine-readable tables use Parquet, with atomic structures represented in JavaScript Object Notation (JSON).

Thermodynamic information distinguishes source hull energies from the estimate for the relaxed structure. Dynamical quantities comprise the sampled minimum frequency and its stability class. Energies are reported per atom and frequencies in THz, with source and calculation conditions accompanying the values.

The chemical, symmetry-resolved, and energy-conditioned proportions in the main text use the complete 2,649,640-structure population. Chemical and symmetry-resolved populations use the source compositions and space groups shown in Figure 2; energy windows use hull estimates for the relaxed structures. Generated inversion partners remain included in these population statistics.

