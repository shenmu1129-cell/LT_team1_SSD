#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
DATA_ROOT="${DATA_ROOT:-/home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/CCTSDB2021}"
EXP_NAME="${EXP_NAME:-cctsdb_ssd}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-logs}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.005}"
EVAL_MAP_EVERY="${EVAL_MAP_EVERY:-10}"
QUICK_EVAL_SAMPLES="${QUICK_EVAL_SAMPLES:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${OUTPUT_DIR}/train.pid"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python train_ssd.py \
  --config configs/cctsdb_ssd.yaml \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --eval-map-every "${EVAL_MAP_EVERY}" \
  --quick-eval-samples "${QUICK_EVAL_SAMPLES}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started CCTSDB SSD training: PID=$(cat "${PID_FILE}") log=${LOG_FILE}"
