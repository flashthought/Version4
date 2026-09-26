# Physics-Informed Estimation of Latent Hemodynamic States from Multimodal Physiological Signals

An MPhil / BME Class Project developing a physics-informed computational framework for continuous latent cardiovascular state estimation and short-horizon trajectory forecasting from non-invasive physiological signals.

## Accelerated 15-Day Milestone (Sept 16 - Sept 30, 2026)
- **Phase 1 (Sept 16-18):** Scaffolding, MIMIC-BP / VitalDB ingestion, and preprocessing pipeline (filters, R-peaks, systolic foot, PAT, SQI).
- **Phase 2 (Sept 19-22):** Controllable reduced-order Windkessel simulator generating synthetic ground truth [R, C, u(t)] under noise.
- **Phase 3 (Sept 23-26):** 4-regime identifiability experiments and physics-informed deep learning architecture.
- **Phase 4 (Sept 27-30):** Multi-tier explainability, SaMD regulatory dossier (IEC 62304 / ISO 14971), and final report.

## Directory Layout
- src/physics/: Reduced-order Windkessel ODE solver, inflow generator, perturbation dynamics.
- src/preprocessing/: Butterworth filtering, fiducial detection, PAT extraction, SQI modules.
- src/data/: Dataset-specific ingestion and preprocessing into a common model-ready window schema.
- src/models/: Dual 1D-CNN temporal encoders, cross-modal attention, latent state z(t), physics loss.
- src/evaluation/: Identifiability metrics, parameter recovery error, trajectory forecasting metrics.
- data/: Raw, processed, and synthetic data storage.
- configs/: YAML experiment configurations.
- docs/samd/: Pre-market SaMD regulatory mapping, risk logs, and intended use statements.
- 	ests/: Automated unit tests for numerical physics and signal processing.

## Dataset Strategy
- **Synthetic simulator:** controlled ground truth for R, C, intervention input, identifiability, and short-horizon ABP forecasting. The generator now uses a 3-element Windkessel model by default, audited physiological parameter ranges, delayed intervention-response functions, and optional measurement artifacts.
- **MIMIC-BP:** public ICU waveform characterization for ECG, PPG, ABP quality, timing, morphology, missingness, and feature distributions.
- **VitalDB:** high-frequency ECG, PPG, ABP plus targeted intervention/context tracks for intervention-response exploration.

Preprocessing scripts write a shared archive format under `data/processed/`.
The default model target is now a 15-second future ABP/MAP trajectory sampled
once per second from a 10-second observation window:
- `python -m src.physics.generate_synthetic_cohort --num-patients 300 --duration-sec 120 --simulator-model 3element --artifact-prob 0.12 --output-dir data/synthetic`
- `python -m src.data.preprocess_synthetic --input-dir data/synthetic --output data/processed/synthetic_windows.npz --window-sec 10 --stride-sec 2 --forecast-horizon-sec 15 --forecast-step-sec 1`
- `python -m src.data.preprocess_mimicbp --input-dir "data/raw/Mimic BP" --output data/processed/mimicbp_windows.npz --window-sec 10 --stride-sec 2 --forecast-horizon-sec 15 --forecast-step-sec 1`
- `python -m src.data.download_vitaldb --include-interventions --include-context`
- `python -m src.data.preprocess_vitaldb --input-dir data/raw/vitaldb --output data/processed/vitaldb_windows.npz --window-sec 10 --stride-sec 2 --forecast-horizon-sec 15 --forecast-step-sec 1`

For a quick VitalDB smoke test, add `--max-cases 5`. Omitting `--max-cases` downloads all eligible cases.

The model now supports the four narrowed-document observation regimes:
- `ppg_only`
- `ppg_ecg`
- `ppg_ecg_abp`
- `ppg_ecg_abp_intervention`

Training uses patient-level synthetic train/validation/test splits and modality dropout by default to improve incomplete-observability robustness:
PPG is kept as the minimum channel, while ECG, ABP, and intervention inputs are randomly masked during training only.

Training hyperparameters live in `configs/training.yaml`. The training loop uses `AdamW` with weight decay, gradient clipping, validation early stopping, best-checkpoint restore, pulse-pressure loss for compliance learning, and automatic CPU/GPU device selection.

Generate report plots after preprocessing/training with:
- `python -m src.evaluation.plot_results --output-dir data/figures`

Validate simulator realism against processed MIMIC-BP/VitalDB summaries with:
- `python -m src.evaluation.validate_simulator --output-dir data/figures/simulator_validation`

For publication stress tests, set `holdout_scenarios` in `configs/training.yaml`, for example:
- `holdout_scenarios: [mixed_shock_stiff, hemorrhage_resuscitation]`

Cluster setup files are provided:
- `requirements-cluster.txt` for non-PyTorch dependencies
- `requirements-cu121.txt`, `requirements-cu124.txt`, `requirements-cu128.txt` for CUDA PyTorch environments
- `docs/cluster_environment.md` for setup/run instructions
