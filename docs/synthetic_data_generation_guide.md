# Synthetic Hemodynamic Dataset Generation Guide

This document summarizes the synthetic simulator used for the project. The
simulator is designed as a transparent in silico benchmark: it is not a full
cardiovascular digital twin, but it provides controlled ground truth for
resistance, compliance, intervention input, artifacts, and short-horizon ABP
forecasting.

## Codebase Architecture

The synthetic generation pipeline uses:

- `src/physics/windkessel.py`: 2-element and 3-element Windkessel solvers,
  pulsatile inflow generation, PPG transfer, and baseline noise injection.
- `src/physics/generate_synthetic_cohort.py`: cohort-level parameter sampling,
  clinical scenario generation, delayed intervention-response functions,
  artifact injection, and reproducibility metadata.
- `src/data/preprocess_synthetic.py`: conversion to the shared model window
  schema.
- `src/evaluation/validate_simulator.py`: distribution comparison against
  processed MIMIC-BP and VitalDB summaries.

## Governing Equations

The default generator uses the 3-element Windkessel model:

```text
C(t) dPc(t)/dt = Q_in(t) - (Pc(t) - Pv) / Rp(t)
P_aortic(t) = Pc(t) + Rc(t) Q_in(t)
```

where:

- `P_aortic(t)` is arterial blood pressure in mmHg.
- `Q_in(t)` is pulsatile aortic inflow in mL/s.
- `R(t)` is total systemic vascular resistance in mmHg*s/mL.
- `C(t)` is arterial compliance in mL/mmHg.
- `Pv` is downstream venous pressure, fixed at 4 mmHg.
- `Rc` is characteristic impedance, set by `R_c_ratio` in the simulator.

The 2-element model remains available as an ablation with
`--simulator-model 2element`.

## Physiological Parameter Ranges

The generator records the exact sampled configuration in
`data/synthetic/simulator_config.json`. Current adult perioperative/ICU
plausible ranges are:

| Parameter | Range |
|---|---:|
| Heart rate | 60-110 bpm |
| Stroke volume | 50-95 mL |
| Systemic vascular resistance | 0.75-1.45 mmHg*s/mL |
| Arterial compliance | 0.65-1.75 mL/mmHg |
| PPG transit time | 140-260 ms |
| ABP SNR | 22-32 dB |
| PPG/ECG SNR | 16-26 dB |

For publication, cite literature supporting these ranges in the manuscript and
report the actual generated cohort distribution from `cohort_summary.csv`.

## Clinical Scenarios

Patients cycle deterministically through seven scenario classes so every cohort
contains broad coverage:

- `stable`
- `septic_shock`
- `vasopressor_rescue`
- `low_compliance`
- `fluid_challenge`
- `mixed_shock_stiff`
- `hemorrhage_resuscitation`

Each scenario records its audit parameters in `cohort_summary.csv`, including
resistance drop, rescue fraction, compliance change, vasopressor onset, fluid
onset, and hemorrhage onset where applicable.

## Intervention Modeling

Interventions are represented in two layers:

- `u_vasopressor` and `u_fluid`: observable administration-style input signals.
- delayed latent effects on `R(t)` and `C(t)`: onset and response are modeled
  with plateau or gamma-like pharmacodynamic curves.

This prevents the simulator from unrealistically changing physiology at the
exact same instant that an intervention input appears.

## Artifacts and Noise

The generator adds Gaussian noise and baseline wander, then optionally injects
audited artifacts with `--artifact-prob`:

- burst noise
- dropped samples
- flatline segments
- clipping
- ABP flush-like spikes

Artifact fractions and counts are saved per patient in `cohort_summary.csv`.
Use `--artifact-prob 0` to create a clean ablation cohort.

## Output Archive Structure

Each `patient_XXX_<scenario>.npz` file includes:

| Key | Description |
|---|---|
| `t` | Time vector |
| `fs` | Sample rate |
| `simulator_model` | `3element` or `2element` |
| `abp_clean`, `abp_noisy` | Clean and observed ABP |
| `ppg_clean`, `ppg_noisy` | Clean and observed PPG |
| `ecg_clean`, `ecg_noisy` | Clean and observed ECG |
| `q_in` | Pulsatile inflow |
| `R_ground_truth` | True resistance trajectory |
| `C_ground_truth` | True compliance trajectory |
| `u_vasopressor` | Vasopressor input |
| `u_fluid` | Fluid input |
| `u_intervention` | Combined intervention input |
| `scenario` | Scenario label |
| `artifact_prob` | Artifact generation probability |

Global files:

- `data/synthetic/cohort_summary.csv`
- `data/synthetic/simulator_config.json`

## Validation Workflow

Generate and preprocess synthetic data:

```bash
python -m src.physics.generate_synthetic_cohort --num-patients 300 --duration-sec 120 --simulator-model 3element --artifact-prob 0.12 --output-dir data/synthetic
python -m src.data.preprocess_synthetic --input-dir data/synthetic --output data/processed/synthetic_windows.npz
```

After MIMIC-BP and VitalDB are also preprocessed, run:

```bash
python -m src.evaluation.validate_simulator --output-dir data/figures/simulator_validation
```

The validation output includes:

- synthetic scenario boxplots
- synthetic vs real distribution overlays
- summary statistics by dataset
- standardized mean difference and quantile-overlap tables

## Publication Positioning

The simulator should be described as a reduced-order, physiology-constrained
benchmark. It supports identifiability and robustness testing but does not
replace real-world validation. Report limitations clearly: simplified PPG
generation, approximate pharmacodynamics, no full baroreflex model, and no
patient-specific vascular anatomy.
