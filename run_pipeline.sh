#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-./runs}"
DATA_ROOT="${DATA_ROOT:-/nas/data_EMG/data_DT}"
TRIAL_ID="${TRIAL_ID:-5}"
DATA_TYPE="${DATA_TYPE:-unslice_features_half}"
TEST_IDS="${TEST_IDS:-1}"
SESSION_ID="${SESSION_ID:-1}"
DEVICE_IDS="${DEVICE_IDS:-0}"
ACCELERATOR="${ACCELERATOR:-gpu}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${ROOT_DIR}/outputs/terminal"

python step0_main_code.py \
  --root_dir "${ROOT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --trail_id "${TRIAL_ID}" \
  --test_id ${TEST_IDS} \
  --session_id "${SESSION_ID}" \
  --data_type "${DATA_TYPE}" \
  --purpose train \
  --accelerator "${ACCELERATOR}" \
  --devices ${DEVICE_IDS} \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

python step0_main_code.py \
  --root_dir "${ROOT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --trail_id "${TRIAL_ID}" \
  --test_id ${TEST_IDS} \
  --session_id "${SESSION_ID}" \
  --data_type "${DATA_TYPE}" \
  --purpose test \
  --accelerator "${ACCELERATOR}" \
  --devices ${DEVICE_IDS} \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

python step1_test_classifier.py \
  --root_dir "${ROOT_DIR}" \
  --trial_id "${TRIAL_ID}" \
  --test_id ${TEST_IDS} \
  --session_id "${SESSION_ID}" \
  --purpose p \
  --methods knn
