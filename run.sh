#!/bin/bash -l

#SBATCH --job-name=pyccempi
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:0
#SBATCH --output=temp_slurm_out/run_%j.out
#SBATCH --error=temp_slurm_out/run_%j.err

# First argument is the run_id (without .yaml)
RUN_ID="$1"
if [ -z "$RUN_ID" ]; then
  echo "Usage: sbatch run.sh <run_id>"
  echo "This will use config ./config/<run_id>.yaml"
  exit 1
fi
YAML_PATH="./config/${RUN_ID}.yaml"
RUN_DIR="./runs/${RUN_ID}"

cd "$SLURM_SUBMIT_DIR"
module load mpi/mpich-x86_64
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate pycce-env

echo "Running on host: $(hostname)"
echo "SLURM_CPUS_PER_TASK = $SLURM_NTASKS"
echo "Working dir: $(pwd)"
echo "Run ID: $RUN_ID"
echo "Config file: $YAML_PATH"
echo "Timestamp: $(date -Iseconds)"

# Send all Python __pycache__ into temp_slurm_out first
export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/temp_slurm_out/pycache_${RUN_ID}_${SLURM_JOB_ID}"
PYCACHE_SRC="temp_slurm_out/pycache_${RUN_ID}_${SLURM_JOB_ID}"

# run the python scripy run.py
mpirun -n "$SLURM_NTASKS" python -u run.py "$YAML_PATH"

##################### IF JOB ABORTED / FAILED #####################
MPI_STATUS=$?
unset PYTHONPYCACHEPREFIX   # <- IMPORTANT: stop later Python (conda) from using our prefix
if [ "$MPI_STATUS" -ne 0 ]; then
  # move pycache into aborted_pycache
  ABORTED_DIR="temp_slurm_out/aborted_pycache"
  mkdir -p "$ABORTED_DIR"   # safe even if it already exists
  if [ -d "$PYCACHE_SRC" ]; then
    mv "$PYCACHE_SRC" "${ABORTED_DIR}/pycache_${RUN_ID}_${SLURM_JOB_ID}"
    echo "Job aborted (exit $MPI_STATUS); moved pycache to ${ABORTED_DIR}/pycache_${RUN_ID}_${SLURM_JOB_ID}"
  else
    echo "Job aborted (exit $MPI_STATUS); no pycache dir $PYCACHE_SRC found."
  fi

  conda deactivate
  exit "$MPI_STATUS"
fi
##################################################################

# assuming job succeeds -->
# ===================== Post-run: copy scripts ===================== #
if [ -d "$RUN_DIR" ]; then
  COPIES_DIR="${RUN_DIR}/copies_of_scripts"
  mkdir -p "$COPIES_DIR"

  # script name without path / extension
  SCRIPT_BASE="$(basename "$0")"
  SCRIPT_STEM="${SCRIPT_BASE%.sh}"

  # Copy with new names
  cp ./sim.py   "${COPIES_DIR}/sim-copy.py"
  cp ./model.py "${COPIES_DIR}/model-copy.py"
  cp ./preprocessing1.py   "${COPIES_DIR}/preprocessing1-copy.py"
  cp ./run.py "${COPIES_DIR}/run-copy.py"
  cp "$0"       "${COPIES_DIR}/${SCRIPT_STEM}-copy.sh"

  echo "Copied scripts → {script_name}-copy.py"
  echo "Copied $0 → ${SCRIPT_STEM}-copy.sh in $COPIES_DIR"
else
  echo "WARNING: run directory '$RUN_DIR' does not exist; skipping script copy."
fi
# ================================================================= #

# ===================== Post-run: move Slurm .out/.err ===================== #
if [ -d "$RUN_DIR" ]; then
  SLURM_DIR="${RUN_DIR}/slurm_output"
  mkdir -p "$SLURM_DIR"

  OUT_FILE="temp_slurm_out/run_${SLURM_JOB_ID}.out"
  ERR_FILE="temp_slurm_out/run_${SLURM_JOB_ID}.err"

  if [ -f "$OUT_FILE" ]; then
    mv "$OUT_FILE" "$SLURM_DIR/"
    echo "Moved $OUT_FILE to $SLURM_DIR/"
  else
    echo "WARNING: $OUT_FILE not found."
  fi

  if [ -f "$ERR_FILE" ]; then
    mv "$ERR_FILE" "$SLURM_DIR/"
    echo "Moved $ERR_FILE to $SLURM_DIR/"
  else
    echo "WARNING: $ERR_FILE not found."
  fi

  # Move pycache tree if it exists
  if [ -d "$PYCACHE_SRC" ]; then
    mv "$PYCACHE_SRC" "${SLURM_DIR}/pycache"
    echo "Moved Python pycache dir $PYCACHE_SRC to ${SLURM_DIR}/pycache"
  else
    echo "No pycache dir $PYCACHE_SRC found."
  fi
else
  echo "WARNING: run directory '$RUN_DIR' does not exist; skipping Slurm output move."
fi
# ========================================================================== #

conda deactivate
