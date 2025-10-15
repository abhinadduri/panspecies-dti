#!/bin/bash
#SBATCH --job-name=LitPCBA_grid
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --array=0-59%30

set -euo pipefail

mkdir -p logs
mkdir -p results

echo "=============================================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID  Array ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "=============================================================="

LOG_FILE="logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

# --- Environment setup --- #
module load cuda
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate sprint2

cd ~/panspecies-dti

# --- Load target list --- #
TARGET_FILE="${TARGET_FILE:-targets.txt}"
if [[ ! -f "$TARGET_FILE" ]]; then
  echo "Error: Targets file not found at $TARGET_FILE"
  exit 1
fi

mapfile -t TARGETS < <(grep -Ev '^\s*($|#)' "$TARGET_FILE")
THRESHOLDS=(0.3 0.5 0.7 0.9)

NUM_T=${#TARGETS[@]}
NUM_H=${#THRESHOLDS[@]}
TOTAL=$(( NUM_T * NUM_H ))

echo "Targets: $NUM_T  Thresholds: $NUM_H  Total combinations: $TOTAL"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "Array index $SLURM_ARRAY_TASK_ID out of range 0..$((TOTAL-1))"
  exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
T_IDX=$(( IDX / NUM_H ))
H_IDX=$(( IDX % NUM_H ))

TARGET="${TARGETS[$T_IDX]}"
THR="${THRESHOLDS[$H_IDX]}"

echo "--------------------------------------------------------------"
echo "Selected Target:      $TARGET"
echo "Similarity Threshold: $THR"
echo "Log file:             $LOG_FILE"
echo "--------------------------------------------------------------"

# --- Directory and W&B --- #
PCBA_DIR="data/lit_pcba"
RESULTS_FILE="results/results_grid.csv"

if [[ ! -f "$RESULTS_FILE" ]]; then
  echo "Target,Threshold,AUROC,BEDROC_85,EF_0.005,EF_0.01,EF_0.05" > "$RESULTS_FILE"
fi

export WANDB_RUN_GROUP="LitPCBA_fullgrid_${SLURM_ARRAY_JOB_ID}"

nvidia-smi || true

# --- Run training + evaluation --- #
ultrafast-train \
  --exp-id "LitPCBA_${TARGET}_thr${THR}" \
  --config configs/saprot_agg_config.yaml \
  --task merged \
  --epochs 15 \
  --ship-model 1 \
  --pcba-target "${TARGET}" \
  --ship-sim-threshold "${THR}" \
  --pcba-dir "$PCBA_DIR" \
  --eval-pcba \
  --wandb-proj "panspecies-litpcba"

# --- Extract metrics from the SLURM log file --- #
AUROC=$(grep -m1 "Average AUROC:" "$LOG_FILE" | awk '{print $3}' | tr -d '\r' || true)
BEDROC=$(grep -m1 "Average BEDROC_85:" "$LOG_FILE" | awk '{print $3}' | tr -d '\r' || true)
EF005=$(grep -m1 "Average EF:" "$LOG_FILE" | sed 's/.*0.005: *\([^,}]*\).*/\1/' | tr -d '\r' || true)
EF01=$( grep -m1 "Average EF:" "$LOG_FILE" | sed 's/.*0.01: *\([^,}]*\).*/\1/'  | tr -d '\r' || true)
EF05=$( grep -m1 "Average EF:" "$LOG_FILE" | sed 's/.*0.05: *\([^,}]*\).*/\1/'  | tr -d '\r' || true)

AUROC=${AUROC:-"NA"}
BEDROC=${BEDROC:-"NA"}
EF005=${EF005:-"NA"}
EF01=${EF01:-"NA"}
EF05=${EF05:-"NA"}

echo "${TARGET},${THR},${AUROC},${BEDROC},${EF005},${EF01},${EF05}" >> "$RESULTS_FILE"

echo "--------------------------------------------------------------"
echo "Saved results for ${TARGET} @ ${THR} → ${RESULTS_FILE}"
echo "--------------------------------------------------------------"
echo "Job completed successfully at: $(date)"
echo "=============================================================="
