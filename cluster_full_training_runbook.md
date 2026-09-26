# Full Cluster Training and Evaluation Runbook

This runbook is for a robust, publishable full run on a GPU cluster. It starts
from environment setup and ends with synthetic, MIMIC-BP, VitalDB, model
training, evaluation, and plots.

Use normal underscores in commands. Do not copy commands with escaped
underscores like `src.models.train\_something`.

## 0. Assumptions

- You have copied this project folder to the cluster.
- You have access to a GPU node.
- Your full MIMIC-BP dataset is available or will be copied into:

```bash
data/raw/Mimic\ BP/
```

- VitalDB will either be downloaded on the cluster or copied from your local
machine into:

```bash
data/raw/vitaldb/
```

## 1. Log In and Use Scratch

Log in to the cluster:

```bash
ssh your_username@your_cluster_address
```

Use `/scratch`, not `/home`, for the full run. Your home space is nearly full,
while scratch has enough space for generated windows, checkpoints, and logs.

```bash
mkdir -p /scratch/bayeboah/ClassProject
rsync -av --exclude data --exclude logs "/home/bayeboah/Class Project/" /scratch/bayeboah/ClassProject/
cd /scratch/bayeboah/ClassProject
```

For a short interactive GPU check on this cluster, request an H100 like this:

```bash
salloc --account=def-uanazodo-ab --gpus-per-node=h100:1 --cpus-per-task=8 --mem=64G --time=0-01:00
```

For the full run, submit the batch script instead of using an interactive
session:

```bash
sbatch scripts/run_hemo_full.slurm
```

By default, this script runs the heavy configuration:

```text
5000 synthetic patients
3500 VitalDB cases
full MIMIC-BP preprocessing, if data/raw/Mimic BP is present
80 training epochs
both data-only and physics-informed models
MIMIC-BP test-window evaluation
VitalDB real-window evaluation
simulator realism validation
explainability attribution output
publication output archive
```

To change the VitalDB target, submit with an environment override:

```bash
VITALDB_MAX_CASES=4000 sbatch scripts/run_hemo_full.slurm
```

If you need a synthetic-only emergency run, explicitly disable the real-data
steps:

```bash
RUN_VITALDB=0 RUN_MIMICBP=0 sbatch scripts/run_hemo_full.slurm
```

Check the queue and logs:

```bash
squeue -u "$USER"
tail -f logs/Hemo_Full-<JOBID>.out
```

## 2. Create and Activate the Python Environment

```bash
module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.6
python -m venv /scratch/bayeboah/hemo_env
source /scratch/bayeboah/hemo_env/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch With CUDA

Choose the command that matches your cluster CUDA version.

For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 12.4:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If the cluster already provides PyTorch as a module, use the cluster module
instead:

```bash
module load cuda
module load pytorch
```

## 4. Install Project Dependencies

If `requirements-cluster.txt` exists:

```bash
pip install -r requirements-cluster.txt
```

If it does not exist, install the required packages directly:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm pyyaml vitaldb requests
```

Optional, useful for notebooks:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name hemo --display-name "Python (hemo)"
```

## 5. Verify GPU Access

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Expected:

```text
True
<GPU name>
```

## 6. Create Required Directories

```bash
mkdir -p data/raw data/processed data/synthetic data/models data/figures logs
```

## 7. Copy Raw Datasets to the Cluster

### 7.1. MIMIC-BP

Expected layout:

```text
data/raw/Mimic BP/
  abp/
  ecg/
  ppg/
  labels/
  train_subjects.txt
  val_subjects.txt
  test_subjects.txt
```

Example copy command from your local machine:

```bash
scp -r "C:/Users/sam/OneDrive/Desktop/Class Project/data/raw/Mimic BP" your_username@your_cluster_address:/path/to/Class\ Project/data/raw/
```

On Linux/macOS local machines, use:

```bash
scp -r "/path/to/Mimic BP" your_username@your_cluster_address:/path/to/Class\ Project/data/raw/
```

### 7.2. VitalDB

If you already downloaded VitalDB locally:

```bash
scp -r "C:/Users/sam/OneDrive/Desktop/Class Project/data/raw/vitaldb" your_username@your_cluster_address:/path/to/Class\ Project/data/raw/
```

If downloading VitalDB on the cluster, use Section 10.

## 8. Generate a Large Synthetic Cohort

For a robust publication-oriented run:

```bash
python -m src.physics.generate_synthetic_cohort \
  --num-patients 1000 \
  --duration-sec 180 \
  --simulator-model 3element \
  --artifact-prob 0.12 \
  --output-dir data/synthetic
```

If cluster time is limited, use a medium run:

```bash
python -m src.physics.generate_synthetic_cohort \
  --num-patients 300 \
  --duration-sec 120 \
  --simulator-model 3element \
  --artifact-prob 0.12 \
  --output-dir data/synthetic
```

## 9. Preprocess Synthetic Data

```bash
python -m src.data.preprocess_synthetic \
  --input-dir data/synthetic \
  --output data/processed/synthetic_windows.npz \
  --window-sec 10 \
  --stride-sec 2 \
  --forecast-horizon-sec 15 \
  --forecast-step-sec 1
```

Expected output:

```text
data/processed/synthetic_windows.npz
data/processed/synthetic_windows_summary.csv
```

## 10. Download More VitalDB Cases

Skip this section if you already copied VitalDB raw files to the cluster.

For a medium VitalDB run:

```bash
python -m src.data.download_vitaldb \
  --include-interventions \
  --include-context \
  --max-cases 100 \
  --output-dir data/raw/vitaldb
```

For a larger run:

```bash
python -m src.data.download_vitaldb \
  --include-interventions \
  --include-context \
  --max-cases 500 \
  --output-dir data/raw/vitaldb
```

For the full eligible VitalDB download, omit `--max-cases`:

```bash
python -m src.data.download_vitaldb \
  --include-interventions \
  --include-context \
  --output-dir data/raw/vitaldb
```

If the VitalDB track table endpoint is blocked, use your locally downloaded
VitalDB folder instead.

## 11. Preprocess Full MIMIC-BP

```bash
python -m src.data.preprocess_mimicbp \
  --input-dir "data/raw/Mimic BP" \
  --output data/processed/mimicbp_windows.npz \
  --window-sec 10 \
  --stride-sec 2 \
  --forecast-horizon-sec 15 \
  --forecast-step-sec 1
```

Expected output:

```text
data/processed/mimicbp_windows.npz
data/processed/mimicbp_windows_summary.csv
```

## 12. Preprocess VitalDB

```bash
python -m src.data.preprocess_vitaldb \
  --input-dir data/raw/vitaldb \
  --output data/processed/vitaldb_windows.npz \
  --window-sec 10 \
  --stride-sec 2 \
  --forecast-horizon-sec 15 \
  --forecast-step-sec 1
```

Expected output:

```text
data/processed/vitaldb_windows.npz
data/processed/vitaldb_windows_summary.csv
```

## 13. Quick Sanity Training Before Full Training

Run this before committing many GPU hours:

```bash
python -m src.models.train \
  --train-models physics_informed \
  --num-epochs 5 \
  --batch-size 64 \
  --early-stopping-patience 2 \
  --log-every 1
```

Check the result:

```bash
cat data/synthetic/model_comparison_results.csv
```

If the model does not beat a simple baseline after the quick run, that is not
fatal. The purpose here is to catch code, CUDA, and data-shape errors.

## 14. Full Main Training

Run both data-only and physics-informed models:

```bash
python -m src.models.train \
  --train-models both \
  --num-epochs 80 \
  --batch-size 64 \
  --early-stopping-patience 10 \
  --log-every 1 \
  2>&1 | tee logs/full_training.log
```

If GPU memory is limited:

```bash
python -m src.models.train \
  --train-models both \
  --num-epochs 80 \
  --batch-size 32 \
  --early-stopping-patience 10 \
  --log-every 1 \
  2>&1 | tee logs/full_training_bs32.log
```

Expected outputs:

```text
data/models/data-only_baseline.pth
data/models/physics-informed_model.pth
data/synthetic/model_comparison_results.csv
data/synthetic/test_window_predictions.csv
data/synthetic/forecast_examples.npz
```

## 15. Publication Stress Test: Scenario Holdout

Edit `configs/training.yaml`:

```yaml
training:
  holdout_scenarios:
    - mixed_shock_stiff
    - hemorrhage_resuscitation
```

Then run:

```bash
python -m src.models.train \
  --train-models physics_informed \
  --num-epochs 80 \
  --batch-size 64 \
  --early-stopping-patience 10 \
  --log-every 1 \
  2>&1 | tee logs/scenario_holdout_training.log
```

After this run, save the outputs separately:

```bash
mkdir -p data/publication_runs/scenario_holdout
cp data/synthetic/model_comparison_results.csv data/publication_runs/scenario_holdout/
cp data/synthetic/test_window_predictions.csv data/publication_runs/scenario_holdout/
cp data/models/physics-informed_model.pth data/publication_runs/scenario_holdout/
```

Set `holdout_scenarios: []` again before the main final training run.

## 16. Evaluate on Processed MIMIC-BP

```bash
python -m src.evaluation.evaluate_mimic \
  --model data/models/physics-informed_model.pth \
  --archive data/processed/mimicbp_windows.npz \
  --split test \
  --batch-size 128 \
  --output data/processed/mimicbp_evaluation.csv
```

## 17. Evaluate on Processed VitalDB

```bash
python -m src.evaluation.evaluate_vitaldb \
  --model data/models/physics-informed_model.pth \
  --archive data/processed/vitaldb_windows.npz \
  --batch-size 128 \
  --output data/processed/vitaldb_evaluation.csv
```

## 18. Generate All Plots

```bash
python -m src.evaluation.plot_results \
  --output-dir data/figures
```

## 19. Validate Simulator Realism Against Real Data

```bash
python -m src.evaluation.validate_simulator \
  --synthetic-summary data/processed/synthetic_windows_summary.csv \
  --mimic-summary data/processed/mimicbp_windows_summary.csv \
  --vitaldb-summary data/processed/vitaldb_windows_summary.csv \
  --cohort-summary data/synthetic/cohort_summary.csv \
  --output-dir data/figures/simulator_validation
```

Expected outputs:

```text
data/figures/simulator_validation/simulator_distribution_summary.csv
data/figures/simulator_validation/simulator_real_overlap_summary.csv
data/figures/simulator_validation/*.png
```

## 20. Summarize Results in the Terminal

Main synthetic comparison:

```bash
cat data/synthetic/model_comparison_results.csv
```

Compact prediction summary:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/synthetic/test_window_predictions.csv")
df["r_abs_err"] = (df["r_pred"] - df["r_true"]).abs()
df["c_abs_err"] = (df["c_pred"] - df["c_true"]).abs()
print(df.groupby(["Model", "Regime"])[[
    "forecast_mae_mmHg",
    "physics_residual",
    "r_abs_err",
    "c_abs_err"
]].mean().round(3))
PY
```

MIMIC-BP evaluation:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/processed/mimicbp_evaluation.csv")
print(df[["mae_mmHg", "rmse_mmHg"]].mean().round(3))
PY
```

VitalDB evaluation:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/processed/vitaldb_evaluation.csv")
print(df[["mae_mmHg", "rmse_mmHg"]].mean().round(3))
print("Intervention windows:", int(df.get("has_intervention", False).sum()))
PY
```

## 21. Archive Outputs for Publication

```bash
mkdir -p data/publication_runs/final_full_run
cp configs/training.yaml data/publication_runs/final_full_run/
cp data/synthetic/cohort_summary.csv data/publication_runs/final_full_run/
cp data/synthetic/model_comparison_results.csv data/publication_runs/final_full_run/
cp data/synthetic/test_window_predictions.csv data/publication_runs/final_full_run/
cp data/processed/*_summary.csv data/publication_runs/final_full_run/
cp data/processed/*_evaluation.csv data/publication_runs/final_full_run/ 2>/dev/null || true
cp -r data/figures data/publication_runs/final_full_run/
```

Compress the archive:

```bash
tar -czf publication_full_run_outputs.tar.gz data/publication_runs/final_full_run
```

## 22. Recommended Final Run Order

For this cluster, the simplest full run is:

```bash
cd /scratch/bayeboah/ClassProject
sbatch scripts/run_hemo_full.slurm
```

For manual control, execute these sections in order:

```text
1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 or copy VitalDB -> 11 -> 12 -> 13 -> 14 -> 16 -> 17 -> 18 -> 19 -> 20 -> 21
```

Run Section 15 separately as a publication stress test.
