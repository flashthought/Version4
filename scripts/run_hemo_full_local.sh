#!/bin/bash
# Local (non-Slurm) equivalent of scripts/run_hemo_full.slurm.
#
# dipperlab-internal has a GPU but no Slurm scheduler and no environment
# modules, so the #SBATCH directives, `module load`, and the /scratch paths do
# not apply. Every pipeline stage, argument and default below is otherwise
# identical to the cluster script, with one exception: the VitalDB download is
# skipped because data/raw/vitaldb already holds all 3458 eligible cases
# (download_vitaldb.py is byte-identical to the version that fetched them, so
# re-downloading would take ~3h to produce the same files).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kyei/Document/Version 4}"
VENV_DIR="${VENV_DIR:-/home/kyei/Document/Class-Project/.venv}"

SYN_PATIENTS="${SYN_PATIENTS:-5000}"
SYN_DURATION_SEC="${SYN_DURATION_SEC:-180}"
ARTIFACT_PROB="${ARTIFACT_PROB:-0.12}"
RUN_VITALDB="${RUN_VITALDB:-1}"
RUN_MIMICBP="${RUN_MIMICBP:-1}"
WINDOW_SEC="${WINDOW_SEC:-10}"
STRIDE_SEC="${STRIDE_SEC:-2}"
FORECAST_HORIZON_SEC="${FORECAST_HORIZON_SEC:-15}"
FORECAST_STEP_SEC="${FORECAST_STEP_SEC:-1}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_REAL_FINETUNE="${RUN_REAL_FINETUNE:-1}"
REAL_FINETUNE_EPOCHS="${REAL_FINETUNE_EPOCHS:-8}"
REAL_FINETUNE_MAX_WINDOWS="${REAL_FINETUNE_MAX_WINDOWS:-60000}"
SYN_MODEL_PATH="data/models/physics-informed_model.pth"
REAL_MODEL_PATH="data/models/physics-informed_real_finetuned.pth"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

mkdir -p data/raw data/processed data/synthetic data/models data/figures logs

echo "Project directory: ${PROJECT_DIR}"
echo "Host: $(hostname)"
echo "Synthetic patients: ${SYN_PATIENTS}"
echo "Synthetic duration: ${SYN_DURATION_SEC} sec"
echo "Epochs: ${EPOCHS} | Batch size: ${BATCH_SIZE}"
echo "Real fine-tuning: ${RUN_REAL_FINETUNE}"
echo "Started: $(date)"

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo "[$(date +%H:%M:%S)] Generating synthetic cohort..."
python -u -m src.physics.generate_synthetic_cohort \
  --num-patients "${SYN_PATIENTS}" \
  --duration-sec "${SYN_DURATION_SEC}" \
  --simulator-model 3element \
  --artifact-prob "${ARTIFACT_PROB}" \
  --output-dir data/synthetic

echo "[$(date +%H:%M:%S)] Preprocessing synthetic cohort..."
python -u -m src.data.preprocess_synthetic \
  --input-dir data/synthetic \
  --output data/processed/synthetic_windows.npz \
  --window-sec "${WINDOW_SEC}" \
  --stride-sec "${STRIDE_SEC}" \
  --forecast-horizon-sec "${FORECAST_HORIZON_SEC}" \
  --forecast-step-sec "${FORECAST_STEP_SEC}"

if [ "${RUN_VITALDB}" = "1" ]; then
  echo "[$(date +%H:%M:%S)] Preprocessing VitalDB (reusing $(ls data/raw/vitaldb/case_*.npz | wc -l) downloaded cases)..."
  python -u -m src.data.preprocess_vitaldb \
    --input-dir data/raw/vitaldb \
    --output data/processed/vitaldb_windows.npz \
    --window-sec "${WINDOW_SEC}" \
    --stride-sec "${STRIDE_SEC}" \
    --forecast-horizon-sec "${FORECAST_HORIZON_SEC}" \
    --forecast-step-sec "${FORECAST_STEP_SEC}"
fi

if [ "${RUN_MIMICBP}" = "1" ]; then
  echo "[$(date +%H:%M:%S)] Preprocessing MIMIC-BP..."
  python -u -m src.data.preprocess_mimicbp \
    --input-dir "data/raw/Mimic BP" \
    --output data/processed/mimicbp_windows.npz \
    --window-sec "${WINDOW_SEC}" \
    --stride-sec "${STRIDE_SEC}" \
    --forecast-horizon-sec "${FORECAST_HORIZON_SEC}" \
    --forecast-step-sec "${FORECAST_STEP_SEC}"
fi

echo "[$(date +%H:%M:%S)] Training full synthetic model comparison..."
python -u -m src.models.train \
  --processed-archive data/processed/synthetic_windows.npz \
  --train-models both \
  --num-epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --early-stopping-patience 10 \
  --log-every 1

EVAL_MODEL_PATH="${SYN_MODEL_PATH}"
if [ "${RUN_REAL_FINETUNE}" = "1" ] && [ "${RUN_MIMICBP}" = "1" ]; then
  echo "[$(date +%H:%M:%S)] Fine-tuning physics-informed model on real processed windows..."
  FINETUNE_ARGS=(
    --base-model "${SYN_MODEL_PATH}"
    --output-model "${REAL_MODEL_PATH}"
    --mimic-archive data/processed/mimicbp_windows.npz
    --batch-size 128
    --num-epochs "${REAL_FINETUNE_EPOCHS}"
    --max-windows-per-source "${REAL_FINETUNE_MAX_WINDOWS}"
  )
  if [ "${RUN_VITALDB}" = "1" ]; then
    FINETUNE_ARGS+=(--vitaldb-archive data/processed/vitaldb_windows.npz --include-vitaldb)
  fi
  python -u -m src.models.fine_tune_real "${FINETUNE_ARGS[@]}"
  EVAL_MODEL_PATH="${REAL_MODEL_PATH}"
fi

echo "[$(date +%H:%M:%S)] Exporting deployment model metadata..."
python -u -m src.inference.export_model_metadata \
  --output data/models/model_metadata.json \
  --model-version "physics-informed-hemo-${RUN_TAG:-local}" \
  --model-path "${EVAL_MODEL_PATH}" \
  --target-fs 125 \
  --window-sec "${WINDOW_SEC}" \
  --forecast-horizon-steps "${FORECAST_HORIZON_SEC}" \
  --forecast-step-sec "${FORECAST_STEP_SEC}"

echo "[$(date +%H:%M:%S)] Exporting app-importable deployment model..."
python -u -m src.inference.export_deployment_model \
  --checkpoint "${EVAL_MODEL_PATH}" \
  --output-dir data/models/deployment \
  --metadata data/models/model_metadata.json \
  --formats torchscript,onnx \
  --target-fs 125 \
  --window-sec "${WINDOW_SEC}" \
  --forecast-horizon "${FORECAST_HORIZON_SEC}" \
  --device cpu

if [ "${RUN_MIMICBP}" = "1" ]; then
  echo "[$(date +%H:%M:%S)] Evaluating on MIMIC-BP test windows..."
  python -u -m src.evaluation.evaluate_mimic \
    --model "${EVAL_MODEL_PATH}" \
    --archive data/processed/mimicbp_windows.npz \
    --split test \
    --batch-size 128 \
    --output data/processed/mimicbp_evaluation.csv
fi

if [ "${RUN_VITALDB}" = "1" ]; then
  echo "[$(date +%H:%M:%S)] Evaluating on VitalDB windows..."
  VITAL_EVAL_ARGS=(
    --model "${EVAL_MODEL_PATH}"
    --archive data/processed/vitaldb_windows.npz
    --batch-size 128
    --output data/processed/vitaldb_evaluation.csv
  )
  if [ -f data/processed/vitaldb_finetune_test_cases.txt ]; then
    VITAL_EVAL_ARGS+=(--include-cases-file data/processed/vitaldb_finetune_test_cases.txt)
  fi
  python -u -m src.evaluation.evaluate_vitaldb "${VITAL_EVAL_ARGS[@]}"
fi

if [ "${RUN_MIMICBP}" = "1" ] && [ "${RUN_VITALDB}" = "1" ]; then
  echo "[$(date +%H:%M:%S)] Validating simulator realism..."
  python -u -m src.evaluation.validate_simulator \
    --synthetic-summary data/processed/synthetic_windows_summary.csv \
    --mimic-summary data/processed/mimicbp_windows_summary.csv \
    --vitaldb-summary data/processed/vitaldb_windows_summary.csv \
    --cohort-summary data/synthetic/cohort_summary.csv \
    --output-dir data/figures/simulator_validation
fi

echo "[$(date +%H:%M:%S)] Running explainability analysis..."
python -u -m src.evaluation.explainability \
  --model "${EVAL_MODEL_PATH}" \
  --fallback-model "${SYN_MODEL_PATH}" \
  --archive data/processed/synthetic_windows.npz \
  --output-dir data/synthetic/explainability

echo "[$(date +%H:%M:%S)] Generating result plots..."
python -u -m src.evaluation.plot_results --output-dir data/figures

RUN_DIR="data/publication_runs/full_${RUN_TAG:-local}"
mkdir -p "${RUN_DIR}"
cp configs/training.yaml "${RUN_DIR}/" 2>/dev/null || true
cp data/models/model_metadata.json "${RUN_DIR}/" 2>/dev/null || true
cp -r data/models/deployment "${RUN_DIR}/" 2>/dev/null || true
cp data/synthetic/cohort_summary.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/synthetic/model_comparison_results.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/synthetic/test_window_predictions.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/processed/*_summary.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/processed/*_evaluation.csv "${RUN_DIR}/" 2>/dev/null || true
cp -r data/figures "${RUN_DIR}/" 2>/dev/null || true
tar -czf "${RUN_DIR}.tar.gz" "${RUN_DIR}"

echo "Done at $(date). Main result table:"
cat data/synthetic/model_comparison_results.csv
echo "Archived run outputs to ${RUN_DIR}.tar.gz"
