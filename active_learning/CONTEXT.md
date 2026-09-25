# Active Learning for IR spectra (ALIR)

On-the-fly active learning that trains a committee of machine-learned interatomic potentials predicting energy,
forces and dipole (cMLIP-EFD) for infrared spectra. The ALIR paper and its supplement define the reference workflows.

## Language

### Workflows (AL strategies)

**AL strategy**:
A named, fully specified active-learning workflow from the ALIR paper, reproduced by exactly one YAML file.
_Avoid_: experiment, config, attempt

**AL run**:
One launch of an **AL strategy** YAML; the paper's run #1/#2/#3 are repeated launches of the same file.
_Avoid_: replica, trial

**Baseline**:
The initial committee trained on bootstrap frames from 1000 K MD and GO with normal-mode sampling; the start
engine for every other **AL strategy**.
_Avoid_: initial model, alir1

**Bootstrap frames**:
The 240 **Baseline** frames: 6 per task (MD at 1000 K, GO + normal-mode sampling at 1000 K) per amino acid, of which
exactly 1 per (task, molecule) is validation and 5 are training.
_Avoid_: initial dataset, seed data

**als-ST**:
Single-task **AL strategy** whose journey contains only the **GO-task**.

**als-MT**:
Multi-task **AL strategy** whose journey contains the **GO-task**, **MD-task** and **CF-task**.

**als-MT-filter**:
**als-MT** plus a duplicate-removal pre-filter on candidate frames.

**als-MT-filter-stop**:
**als-MT-filter** plus a stop rule that ends the loop as soon as convergence leaves fewer than two active systems;
that iteration is not trained, so the final committee is the previous iteration's and its labelled frames are unused.

**als-MT-AA|DP**:
**als-MT** with the 10 dipeptides as the only active systems, fine-tuned from the final **als-MT** run #1 committee
weights without importing its training data (the dataset starts empty).

**Validation group**:
The set of newly labelled frames sharing (task, frame getter, molecule); each group of two or more frames sends at
least one frame to validation (10%, minimum one).
_Avoid_: stratum, category

### Levels of theory

**Reference engine**:
The engine that labels frames with energy, forces and dipole; GFNFF in the reproduction (ADF XLYP/DZP in the paper).
_Avoid_: DFT (in the reproduction), labeller engine (as a domain term)

**Bootstrap generator**:
The engine that runs the **Baseline** sampling tasks; GFN1-xTB, deliberately different from the **Reference engine**.
_Avoid_: start engine (for the Baseline)

### Validity checks

**Committee agreement**:
The mean, over committee members, of CosIR between each member's harmonic IR spectrum and the spectrum from the
committee-averaged Hessian; a **GO-task** passes when it is at least 0.9.
_Avoid_: IR uncertainty, R2/Pearson agreement

**CosIR**:
Cosine similarity between two IR spectra Lorentzian-broadened (30 cm⁻¹) on a common frequency grid.
_Avoid_: match score, spectral similarity (unqualified)

### Tasks

**GO-task**:
Geometry optimization followed by a harmonic frequency (IR) calculation with the committee.

**MD-task**:
NVT molecular dynamics at 300 K stopped early when committee uncertainty exceeds its limits.

**CF-task**:
RDKit conformer generation optimized with the committee.

### Frame getters

**MD-getter**:
Selects, from an **MD-task** trajectory sampled every 2 steps, the top 25% frames by committee energy uncertainty and
then up to 5 by farthest-point sampling on time; an early-stopped trajectory with fewer than 5 frames yields 1 frame.
_Avoid_: MDLinear, MDUnc

**GO-getter**:
Collects the middle and last frame of a **GO-task** trajectory and, only if the optimization converged, 5
normal-mode samples at 300 K from the committee's modes.
_Avoid_: GOTrajectoryFractions, NMS (as the whole getter)

**CF-getter**:
Collects every conformer of a **CF-task**, only if its optimizations converged without errors; no trajectory frames.
_Avoid_: ConfTrajGetter

### Filters

**Pre-filter**:
A geometry-only rule applied to candidate frames before labelling; every **AL strategy** drops frames with any atom
pair closer than 0.7 Å.

**Duplicate removal**:
The **als-MT-filter** pre-filter that clusters candidate frames of the same molecule by Kabsch RMSD (aligned to the
first candidate, complete linkage, 0.05 Å) and keeps one frame per cluster.
_Avoid_: dedup, conformer filter

**Post-filter**:
A rule on labelled frames applied before splitting; every **AL strategy** drops frames with a maximum force above
10 eV/Å.

### Journey

**Active system**:
A molecule with at least one task that has not yet passed its validity and accuracy checks.
_Avoid_: running molecule, pending molecule

**Task retirement**:
A task that passes all its checks for a molecule is never re-run for that molecule; the molecule becomes inactive once
every task has passed, possibly in different iterations with different committees.

### Evaluation

**Loop-side metrics**:
Numbers produced by an **AL run** itself: iterations, final training-set size, wall time, **Committee agreement** and
active-system count per iteration, and accuracy-check MAEs per iteration.

**Test set**:
A fixed collection of reference IR spectra (test-20AA, test-20AAC, test-10DP, test-7AAMD) used only after the loop to
score a final committee; never seen by the loop.
_Avoid_: validation set (that is the loop's split)

## Relationships

- The **Baseline** is the start engine of **als-ST**, **als-MT**, **als-MT-filter** and **als-MT-filter-stop**; they
  are fine-tuned from its weights, but its **Bootstrap frames** are never imported, so their datasets start empty
- **als-MT-AA|DP** starts from the final committee of an **als-MT** **AL run**, not from the **Baseline**, and likewise
  imports none of that run's training data
- An **AL strategy** has one YAML and one or more **AL runs**
- The **Bootstrap generator** produces **Baseline** frames; the **Reference engine** labels every frame in every
  **AL strategy**

## Open items

- **Test sets** recomputed with the GFNFF **Reference engine**, and the spectrum-scoring pipeline (CosIR, Hungarian
  assignment, Duschinsky, PVDOS), are out of scope for the YAML reproduction and remain a follow-up.
- The dipeptide geometries for **als-MT-AA|DP** still have to be located.

## Example dialogue

> **Dev:** "Do I need a separate YAML for als-MT run #2?"
> **Domain expert:** "No — run #2 is just another **AL run** of the als-MT **AL strategy**; the seed stays the same."

## Flagged ambiguities

- The existing `alir1_*` / `alir2_*` YAML names mix strategies and tuning attempts — resolved: new YAMLs are named
  after the paper's **AL strategy** labels; the old files are kept as history.
- "DFT" in the paper means the **Reference engine**; in the reproduction that is GFNFF, so the **Baseline** stays
  off-reference because its frames come from GFN1-xTB.
- Reproduction fidelity: every **AL strategy** uses the paper's settings verbatim (MACE Table S3, accuracy targets
  Table I, task settings Tables S1/S2), even where the GFNFF **Reference engine** makes them easy to satisfy;
  deviations are new, separately named strategies.
- "In addition" in the paper's GO-getter means a union: trajectory frames are always collected and normal-mode
  samples are added on top for a converged GO (`type: UnionCheckerGetter`); the older `type: ConcatCheckerGetter` is a cascade that keeps only
  one getter's frames and must not be used for the **GO-getter**.
- The paper's "grouping by origin, as determined by the frame getter" is implemented as a **Validation group** that
  also includes the molecule, so every molecule keeps validation coverage; this raises the validation share above
  10% for small groups.
