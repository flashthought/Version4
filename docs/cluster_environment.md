# Cluster Environment Setup

This project should be run in a clean Python environment on the cluster. Use a
GPU node for training.

## Recommended Setup

Create an environment:

```bash
conda create -n hemo python=3.10 -y
conda activate hemo
```

Install PyTorch for the CUDA version supported by the cluster. Ask the cluster
docs/admins which CUDA module is available.

For CUDA 12.1:

```bash
pip install -r requirements-cu121.txt
```

For CUDA 12.4:

```bash
pip install -r requirements-cu124.txt
```

For CUDA 12.8:

```bash
pip install -r requirements-cu128.txt
```

If the cluster provides PyTorch as a module, load that module first and install
only the non-PyTorch packages:

```bash
pip install -r requirements-cluster.txt
```

## Verify GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

The output should include `True`.

## Standard Run Order

```bash
python -m src.physics.generate_synthetic_cohort --num-patients 300 --duration-sec 120 --simulator-model 3element --artifact-prob 0.12 --output-dir data/synthetic
python -m src.data.preprocess_synthetic --input-dir data/synthetic --output data/processed/synthetic_windows.npz --window-sec 10 --stride-sec 2 --forecast-horizon-sec 15 --forecast-step-sec 1
python -m src.models.train
python -m src.evaluation.plot_results --output-dir data/figures
```

If running MIMIC-BP/VitalDB on the cluster, copy the raw datasets into
`data/raw/` first, then run:

```bash
python -m src.data.preprocess_mimicbp --output data/processed/mimicbp_windows.npz --window-sec 10 --stride-sec 2 --forecast-horizon-sec 15 --forecast-step-sec 1
python -m src.data.preprocess_vitaldb --output data/processed/vitaldb_windows.npz --window-sec 10 --stride-sec 2 --forecast-horizon-sec 15 --forecast-step-sec 1
python -m src.evaluation.validate_simulator --output-dir data/figures/simulator_validation
```

For scenario-level generalization tests, edit `configs/training.yaml` and set
`holdout_scenarios`, for example `[mixed_shock_stiff, hemorrhage_resuscitation]`.
