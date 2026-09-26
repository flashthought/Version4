#!/bin/bash
#SBATCH --account=def-uanazodo-ab
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-05:00
#SBATCH --job-name=Hemo_Full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/bayeboah/ClassProject}"
VENV_DIR="${VENV_DIR:-/scratch/bayeboah/hemo_env}"

SYN_PATIENTS="${SYN_PATIENTS:-5000}"
SYN_DURATION_SEC="${SYN_DURATION_SEC:-180}"
ARTIFACT_PROB="${ARTIFACT_PROB:-0.12}"
VITALDB_MAX_CASES="${VITALDB_MAX_CASES:-3500}"
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

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.6

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

mkdir -p data/raw data/processed data/synthetic data/models data/figures logs

echo "Project directory: ${PROJECT_DIR}"
echo "Virtual environment: ${VENV_DIR}"
echo "SLURM job id: ${SLURM_JOB_ID:-manual}"
echo "Synthetic patients: ${SYN_PATIENTS}"
echo "Synthetic duration: ${SYN_DURATION_SEC} sec"
echo "VitalDB target cases: ${VITALDB_MAX_CASES}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Real fine-tuning: ${RUN_REAL_FINETUNE}"

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo "Checking Python modules..."
CORE_MODULES=(
  src/physics/generate_synthetic_cohort.py \
  src/data/preprocess_synthetic.py \
  src/models/train.py \
  src/models/fine_tune_real.py \
  src/evaluation/plot_results.py \
  src/evaluation/explainability.py \
  src/evaluation/validate_simulator.py \
  src/inference/export_model_metadata.py \
  src/inference/export_deployment_model.py \
  src/inference/predictor.py \
  src/inference/preprocessing.py \
  src/inference/quality_gate.py \
  src/inference/schemas.py \
  src/inference/streaming_buffer.py
)

OPTIONAL_MODULES=()
if [ "${RUN_VITALDB}" = "1" ]; then
  if [ ! -f src/data/download_vitaldb.py ] || [ ! -f src/data/preprocess_vitaldb.py ]; then
    echo "ERROR: VitalDB full run requested, but VitalDB download/preprocess scripts are missing."
    echo "Expected: src/data/download_vitaldb.py and src/data/preprocess_vitaldb.py"
    exit 1
  fi
  OPTIONAL_MODULES+=(src/data/download_vitaldb.py src/data/preprocess_vitaldb.py)
  OPTIONAL_MODULES+=(src/evaluation/evaluate_vitaldb.py)
fi
if [ "${RUN_MIMICBP}" = "1" ]; then
  if [ ! -f src/data/preprocess_mimicbp.py ]; then
    echo "ERROR: MIMIC-BP full run requested, but src/data/preprocess_mimicbp.py is missing."
    exit 1
  fi
  if [ ! -d "data/raw/Mimic BP" ]; then
    echo "ERROR: MIMIC-BP full run requested, but data/raw/Mimic BP was not found."
    exit 1
  fi
  OPTIONAL_MODULES+=(src/data/preprocess_mimicbp.py)
  OPTIONAL_MODULES+=(src/evaluation/evaluate_mimic.py)
fi

python -m py_compile "${CORE_MODULES[@]}" "${OPTIONAL_MODULES[@]}"

echo "Generating synthetic cohort..."
python -m src.physics.generate_synthetic_cohort \
  --num-patients "${SYN_PATIENTS}" \
  --duration-sec "${SYN_DURATION_SEC}" \
  --simulator-model 3element \
  --artifact-prob "${ARTIFACT_PROB}" \
  --output-dir data/synthetic

echo "Preprocessing synthetic cohort..."
python -m src.data.preprocess_synthetic \
  --input-dir data/synthetic \
  --output data/processed/synthetic_windows.npz \
  --window-sec "${WINDOW_SEC}" \
  --stride-sec "${STRIDE_SEC}" \
  --forecast-horizon-sec "${FORECAST_HORIZON_SEC}" \
  --forecast-step-sec "${FORECAST_STEP_SEC}"

if [ "${RUN_VITALDB}" = "1" ]; then
  echo "Downloading VitalDB cases: ${VITALDB_MAX_CASES}"
  python -m src.data.download_vitaldb \
    --include-interventions \
    --include-context \
    --max-cases "${VITALDB_MAX_CASES}" \
    --output-dir data/raw/vitaldb
else
  echo "Skipping VitalDB because RUN_VITALDB=${RUN_VITALDB}."
fi

if [ "${RUN_VITALDB}" = "1" ]; then
  echo "Preprocessing VitalDB..."
  python -m src.data.preprocess_vitaldb \
    --input-dir data/raw/vitaldb \
    --output data/processed/vitaldb_windows.npz \
    --window-sec "${WINDOW_SEC}" \
    --stride-sec "${STRIDE_SEC}" \
    --forecast-horizon-sec "${FORECAST_HORIZON_SEC}" \
    --forecast-step-sec "${FORECAST_STEP_SEC}"
fi

if [ "${RUN_MIMICBP}" = "1" ]; then
  echo "Preprocessing MIMIC-BP..."
  python -m src.data.preprocess_mimicbp \
    --input-dir "data/raw/Mimic BP" \
    --output data/processed/mimicbp_windows.npz \
    --window-sec "${WINDOW_SEC}" \
    --stride-sec "${STRIDE_SEC}" \
    --forecast-horizon-sec "${FORECAST_HORIZON_SEC}" \
    --forecast-step-sec "${FORECAST_STEP_SEC}"
else
  echo "Skipping MIMIC-BP because RUN_MIMICBP=${RUN_MIMICBP}."
fi

echo "Training full synthetic model comparison..."
python -m src.models.train \
  --processed-archive data/processed/synthetic_windows.npz \
  --train-models both \
  --num-epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --early-stopping-patience 10 \
  --log-every 1

EVAL_MODEL_PATH="${SYN_MODEL_PATH}"
if [ "${RUN_REAL_FINETUNE}" = "1" ] && [ "${RUN_MIMICBP}" = "1" ]; then
  echo "Fine-tuning physics-informed model on real processed windows..."
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
  python -m src.models.fine_tune_real "${FINETUNE_ARGS[@]}"
  EVAL_MODEL_PATH="${REAL_MODEL_PATH}"
else
  echo "Skipping real-data fine-tuning because RUN_REAL_FINETUNE=${RUN_REAL_FINETUNE} or MIMIC-BP is disabled."
fi

echo "Exporting deployment model metadata..."
python -m src.inference.export_model_metadata \
  --output data/models/model_metadata.json \
  --model-version "physics-informed-hemo-${SLURM_JOB_ID:-local}" \
  --model-path "${EVAL_MODEL_PATH}" \
  --target-fs 125 \
  --window-sec "${WINDOW_SEC}" \
  --forecast-horizon-steps "${FORECAST_HORIZON_SEC}" \
  --forecast-step-sec "${FORECAST_STEP_SEC}"

echo "Exporting app-importable deployment model..."
python -m src.inference.export_deployment_model \
  --checkpoint "${EVAL_MODEL_PATH}" \
  --output-dir data/models/deployment \
  --metadata data/models/model_metadata.json \
  --formats torchscript,onnx \
  --target-fs 125 \
  --window-sec "${WINDOW_SEC}" \
  --forecast-horizon "${FORECAST_HORIZON_SEC}" \
  --device cpu

if [ "${RUN_MIMICBP}" = "1" ]; then
  echo "Evaluating trained model on MIMIC-BP test windows..."
  python -m src.evaluation.evaluate_mimic \
    --model "${EVAL_MODEL_PATH}" \
    --archive data/processed/mimicbp_windows.npz \
    --split test \
    --batch-size 128 \
    --output data/processed/mimicbp_evaluation.csv
fi

if [ "${RUN_VITALDB}" = "1" ]; then
  echo "Evaluating trained model on VitalDB windows..."
  VITAL_EVAL_ARGS=(
    --model "${EVAL_MODEL_PATH}"
    --archive data/processed/vitaldb_windows.npz
    --batch-size 128
    --output data/processed/vitaldb_evaluation.csv
  )
  if [ -f data/processed/vitaldb_finetune_test_cases.txt ]; then
    VITAL_EVAL_ARGS+=(--include-cases-file data/processed/vitaldb_finetune_test_cases.txt)
  fi
  python -m src.evaluation.evaluate_vitaldb "${VITAL_EVAL_ARGS[@]}"
fi

if [ "${RUN_MIMICBP}" = "1" ] && [ "${RUN_VITALDB}" = "1" ]; then
  echo "Validating simulator realism against MIMIC-BP and VitalDB summaries..."
  python -m src.evaluation.validate_simulator \
    --synthetic-summary data/processed/synthetic_windows_summary.csv \
    --mimic-summary data/processed/mimicbp_windows_summary.csv \
    --vitaldb-summary data/processed/vitaldb_windows_summary.csv \
    --cohort-summary data/synthetic/cohort_summary.csv \
    --output-dir data/figures/simulator_validation
else
  echo "Skipping simulator validation because it needs both MIMIC-BP and VitalDB summaries."
fi

echo "Running explainability analysis..."
python -m src.evaluation.explainability \
  --model "${EVAL_MODEL_PATH}" \
  --fallback-model "${SYN_MODEL_PATH}" \
  --archive data/processed/synthetic_windows.npz \
  --output-dir data/synthetic/explainability

echo "Generating result plots..."
python -m src.evaluation.plot_results --output-dir data/figures

RUN_DIR="data/publication_runs/full_${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_DIR}"
cp configs/training.yaml "${RUN_DIR}/" 2>/dev/null || true
cp data/models/model_metadata.json "${RUN_DIR}/" 2>/dev/null || true
cp -r data/models/deployment "${RUN_DIR}/" 2>/dev/null || true
cp data/synthetic/cohort_summary.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/synthetic/model_comparison_results.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/synthetic/test_window_predictions.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/processed/*_summary.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/processed/*_evaluation.csv "${RUN_DIR}/" 2>/dev/null || true
cp data/processed/*_finetune_*_cases.txt "${RUN_DIR}/" 2>/dev/null || true
cp -r data/figures "${RUN_DIR}/" 2>/dev/null || true
cp -r data/synthetic/explainability "${RUN_DIR}/" 2>/dev/null || true
tar -czf "${RUN_DIR}.tar.gz" "${RUN_DIR}"

echo "Done. Main result table:"
cat data/synthetic/model_comparison_results.csv
echo "Archived run outputs to ${RUN_DIR}.tar.gz"
