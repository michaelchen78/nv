#!/bin/bash -l

#SBATCH --account=physics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:0
#SBATCH --output=temp_slurm_out/run_%j.out
#SBATCH --error=temp_slurm_out/run_%j.err

usage() {
  echo "Usage: ./run.sh -p <p|s> -n <ntasks> [-m <mem_GB>] [-d <slurm_job_id>] <run_id>"
  echo
  echo "Required:"
  echo "  -p p        Use preemptable partition"
  echo "  -p s        Use standard partition"
  echo "  -n N        Number of MPI tasks"
  echo
  echo "Optional:"
  echo "  -m GB       Memory in GB, e.g. -m 36 becomes --mem=36G"
  echo "  -d JOBID    Do not start until Slurm job JOBID has finished"
  echo
  echo "Example:"
  echo "  ./run.sh -p p -n 30 -m 36 -d 1234567 2026.5.23_some-identifier"
}

PARTITION_CODE=""
NTASKS=""
MEM_RAW=""
DEPENDENCY_JOB_ID=""
ORIGINAL_ARGS=("$@")

while getopts ":p:n:m:d:h" opt; do
  case "$opt" in
    p)
      PARTITION_CODE="$OPTARG"
      ;;
    n)
      NTASKS="$OPTARG"
      ;;
    m)
      MEM_RAW="$OPTARG"
      ;;
    d)
      DEPENDENCY_JOB_ID="$OPTARG"
      ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "ERROR: unknown option -$OPTARG"
      usage
      exit 1
      ;;
    :)
      echo "ERROR: option -$OPTARG requires an argument"
      usage
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

RUN_ID="$1"

if [ -z "$PARTITION_CODE" ]; then
  echo "ERROR: partition is required."
  usage
  exit 1
fi

if [ "$PARTITION_CODE" = "p" ]; then
  PARTITION_NAME="preemptable"
elif [ "$PARTITION_CODE" = "s" ]; then
  PARTITION_NAME="standard"
else
  echo "ERROR: partition must be either 'p' for preemptable or 's' for standard."
  usage
  exit 1
fi

if [ -z "$NTASKS" ]; then
  echo "ERROR: ntasks is required."
  usage
  exit 1
fi

if ! [[ "$NTASKS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ntasks must be a positive integer."
  exit 1
fi

if [ "$NTASKS" -lt 1 ]; then
  echo "ERROR: ntasks must be at least 1."
  exit 1
fi

if [ -z "$RUN_ID" ]; then
  echo "ERROR: run_id is required."
  usage
  exit 1
fi

MEM_ARG=""
if [ -n "$MEM_RAW" ]; then
  if [[ "$MEM_RAW" =~ ^[0-9]+$ ]]; then
    MEM_ARG="${MEM_RAW}G"
  elif [[ "$MEM_RAW" =~ ^[0-9]+[KkMmGgTt]$ ]]; then
    MEM_ARG="$MEM_RAW"
  else
    echo "ERROR: memory must look like 36, 36G, 36000M, etc."
    exit 1
  fi
fi

if [ -n "$DEPENDENCY_JOB_ID" ]; then
  if ! [[ "$DEPENDENCY_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: dependency job ID must be a Slurm job number."
    exit 1
  fi
fi

# ===================== Self-submit with dynamic Slurm options ===================== #
# Slurm reads #SBATCH directives before this script runs, so dynamic job names,
# partition, ntasks, memory, and dependencies must be passed through sbatch.
if [ -z "$SLURM_JOB_ID" ]; then
  mkdir -p temp_slurm_out

  # Extract everything after leading date pattern like 2026.5.23_
  JOB_NAME="$(echo "$RUN_ID" | sed -E 's/^[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}_//')"

  # Keep only reasonable Slurm-safe characters
  JOB_NAME="$(echo "$JOB_NAME" | tr -cd '[:alnum:]_.-')"

  # Keep first 10 characters
  JOB_NAME="${JOB_NAME:0:10}"

  # Fallback if something weird happens
  if [ -z "$JOB_NAME" ]; then
    JOB_NAME="pycce"
  fi

  SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --partition="$PARTITION_NAME"
    --ntasks="$NTASKS"
    --export=ALL,RUN_SH_SELF_SUBMITTED=1
    --no-requeue
  )

  if [ -n "$MEM_ARG" ]; then
    SBATCH_ARGS+=(--mem="$MEM_ARG")
  fi

  if [ -n "$DEPENDENCY_JOB_ID" ]; then
    SBATCH_ARGS+=(--dependency="afterany:${DEPENDENCY_JOB_ID}")
  fi

  RUN_ARGS=(
    -p "$PARTITION_CODE"
    -n "$NTASKS"
  )

  if [ -n "$MEM_RAW" ]; then
    RUN_ARGS+=(-m "$MEM_RAW")
  fi

  if [ -n "$DEPENDENCY_JOB_ID" ]; then
    RUN_ARGS+=(-d "$DEPENDENCY_JOB_ID")
  fi

  RUN_ARGS+=("$RUN_ID")

  exec sbatch "${SBATCH_ARGS[@]}" "$0" "${RUN_ARGS[@]}"
fi

if [ -z "$RUN_SH_SELF_SUBMITTED" ]; then
  echo "ERROR: do not submit this script with sbatch directly."
  echo "Use:"
  echo "  ./run.sh -p p -n 30 [-m 36] [-d 1234567] <run_id>"
  exit 1
fi
# ================================================================================ #

cd "$SLURM_SUBMIT_DIR"

YAML_PATH="./config/${RUN_ID}.yaml"
RUN_DIR="./runs/${RUN_ID}"

mkdir -p temp_slurm_out

module load mpi/mpich-x86_64
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate pycce-env

SCRIPT_START_TIMESTAMP="$(date -Iseconds)"
SCRIPT_START_EPOCH="$(date +%s)"

echo "Running on host: $(hostname)"
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_PARTITION = $SLURM_JOB_PARTITION"
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "Requested memory = ${MEM_ARG:-unspecified}"
echo "Dependency job ID = ${DEPENDENCY_JOB_ID:-none}"
echo "Working dir: $(pwd)"
echo "Run ID: $RUN_ID"
echo "Config file: $YAML_PATH"
echo "Start timestamp: $SCRIPT_START_TIMESTAMP"
echo
echo "Arguments passed to run.sh:"
printf '  %q' "${ORIGINAL_ARGS[@]}"
echo
echo

echo "Parsed arguments:"
echo "  PARTITION_CODE = $PARTITION_CODE"
echo "  PARTITION_NAME = $PARTITION_NAME"
echo "  NTASKS = $NTASKS"
echo "  MEM_RAW = ${MEM_RAW:-unset}"
echo "  MEM_ARG = ${MEM_ARG:-unset}"
echo "  DEPENDENCY_JOB_ID = ${DEPENDENCY_JOB_ID:-unset}"
echo "  RUN_ID = $RUN_ID"
echo "  YAML_PATH = $YAML_PATH"
echo "  RUN_DIR = $RUN_DIR"
echo

# Send all Python __pycache__ into temp_slurm_out first
export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/temp_slurm_out/pycache_${RUN_ID}_${SLURM_JOB_ID}"
PYCACHE_SRC="temp_slurm_out/pycache_${RUN_ID}_${SLURM_JOB_ID}"

# ===================== Pre-run: freeze submitted code + YAML ===================== #
# This snapshot is made BEFORE Python starts.
# Python is then run FROM this snapshot, so these are the actual files used.
SNAPSHOT_DIR="temp_slurm_out/submitted_code_${RUN_ID}_${SLURM_JOB_ID}"

rm -rf "$SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"

# Copy all top-level Python files so local imports use frozen versions.
find . -maxdepth 1 -type f -name "*.py" -exec cp -p {} "$SNAPSHOT_DIR/" \;

# Copy this run script.
SCRIPT_BASE="$(basename "$0")"
SCRIPT_STEM="${SCRIPT_BASE%.sh}"
cp -p "$0" "${SNAPSHOT_DIR}/${SCRIPT_STEM}.sh"

# Copy YAML config used for this run.
if [ -f "$YAML_PATH" ]; then
  cp -p "$YAML_PATH" "${SNAPSHOT_DIR}/config.yaml"
  echo "Frozen YAML config $YAML_PATH to ${SNAPSHOT_DIR}/config.yaml"
else
  echo "ERROR: YAML config '$YAML_PATH' not found; cannot run."
  conda deactivate
  exit 1
fi

# Force Python imports to prefer the frozen snapshot.
export PYTHONPATH="$SNAPSHOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Frozen submitted code in $SNAPSHOT_DIR"
echo

# Run the frozen Python script, not the mutable working-tree version.
mpirun -n "$SLURM_NTASKS" python -u "${SNAPSHOT_DIR}/run.py" "${SNAPSHOT_DIR}/config.yaml"

##################### IF JOB ABORTED / FAILED #####################
MPI_STATUS=$?
MPI_FINISH_TIMESTAMP="$(date -Iseconds)"
MPI_FINISH_EPOCH="$(date +%s)"
MPI_ELAPSED_SECONDS=$((MPI_FINISH_EPOCH - SCRIPT_START_EPOCH))

echo "MPI finish timestamp: $MPI_FINISH_TIMESTAMP"
echo "MPI elapsed seconds: $MPI_ELAPSED_SECONDS"
echo "MPI exit status: $MPI_STATUS"

unset PYTHONPYCACHEPREFIX   # IMPORTANT: stop later Python from using our prefix

if [ "$MPI_STATUS" -ne 0 ]; then
  ABORTED_DIR="temp_slurm_out/aborted_pycache"
  mkdir -p "$ABORTED_DIR"

  if [ -d "$PYCACHE_SRC" ]; then
    mv "$PYCACHE_SRC" "${ABORTED_DIR}/pycache_${RUN_ID}_${SLURM_JOB_ID}"
    echo "Job aborted exit $MPI_STATUS; moved pycache to ${ABORTED_DIR}/pycache_${RUN_ID}_${SLURM_JOB_ID}"
  else
    echo "Job aborted exit $MPI_STATUS; no pycache dir $PYCACHE_SRC found."
  fi

  if [ -d "$SNAPSHOT_DIR" ]; then
    mv "$SNAPSHOT_DIR" "${ABORTED_DIR}/submitted_code_${RUN_ID}_${SLURM_JOB_ID}"
    echo "Moved frozen submitted code to ${ABORTED_DIR}/submitted_code_${RUN_ID}_${SLURM_JOB_ID}"
  fi

  SCRIPT_FINISH_TIMESTAMP="$(date -Iseconds)"
  SCRIPT_FINISH_EPOCH="$(date +%s)"
  SCRIPT_ELAPSED_SECONDS=$((SCRIPT_FINISH_EPOCH - SCRIPT_START_EPOCH))

  echo "Script finish timestamp: $SCRIPT_FINISH_TIMESTAMP"
  echo "Script elapsed seconds: $SCRIPT_ELAPSED_SECONDS"
  echo "Final status: FAILED"

  conda deactivate
  exit "$MPI_STATUS"
fi
##################################################################

# ===================== Post-run: store frozen submitted code ===================== #
# At this point, SNAPSHOT_DIR contains the exact code/YAML that was run.
# Now archive it using the old copy-file names.
if [ -d "$RUN_DIR" ]; then
  COPIES_DIR="${RUN_DIR}/copies_of_scripts"

  rm -rf "$COPIES_DIR"
  mkdir -p "$COPIES_DIR"

  # Copy the main files using the old archive names.
  if [ -f "${SNAPSHOT_DIR}/sim.py" ]; then
    cp -p "${SNAPSHOT_DIR}/sim.py" "${COPIES_DIR}/sim-copy.py"
  else
    echo "WARNING: ${SNAPSHOT_DIR}/sim.py not found; skipping sim-copy.py"
  fi

  if [ -f "${SNAPSHOT_DIR}/model.py" ]; then
    cp -p "${SNAPSHOT_DIR}/model.py" "${COPIES_DIR}/model-copy.py"
  else
    echo "WARNING: ${SNAPSHOT_DIR}/model.py not found; skipping model-copy.py"
  fi

  if [ -f "${SNAPSHOT_DIR}/pp1.py" ]; then
    cp -p "${SNAPSHOT_DIR}/pp1.py" "${COPIES_DIR}/preprocessing1-copy.py"
  else
    echo "WARNING: ${SNAPSHOT_DIR}/pp1.py not found; skipping preprocessing1-copy.py"
  fi

  if [ -f "${SNAPSHOT_DIR}/run.py" ]; then
    cp -p "${SNAPSHOT_DIR}/run.py" "${COPIES_DIR}/run-copy.py"
  else
    echo "WARNING: ${SNAPSHOT_DIR}/run.py not found; skipping run-copy.py"
  fi

  if [ -f "${SNAPSHOT_DIR}/${SCRIPT_STEM}.sh" ]; then
    cp -p "${SNAPSHOT_DIR}/${SCRIPT_STEM}.sh" "${COPIES_DIR}/${SCRIPT_STEM}-copy.sh"
  else
    echo "WARNING: ${SNAPSHOT_DIR}/${SCRIPT_STEM}.sh not found; skipping ${SCRIPT_STEM}-copy.sh"
  fi

  if [ -f "${SNAPSHOT_DIR}/config.yaml" ]; then
    cp -p "${SNAPSHOT_DIR}/config.yaml" "${COPIES_DIR}/config-copy.yaml"
  else
    echo "WARNING: ${SNAPSHOT_DIR}/config.yaml not found; skipping config-copy.yaml"
  fi

  # Preserve any other top-level Python files that were part of the frozen snapshot.
  # These are not renamed because the old script did not have special names for them.
  for PY_FILE in "${SNAPSHOT_DIR}"/*.py; do
    [ -e "$PY_FILE" ] || continue

    PY_BASE="$(basename "$PY_FILE")"

    if [ "$PY_BASE" = "sim.py" ] || \
       [ "$PY_BASE" = "model.py" ] || \
       [ "$PY_BASE" = "pp1.py" ] || \
       [ "$PY_BASE" = "run.py" ]; then
      continue
    fi

    cp -p "$PY_FILE" "${COPIES_DIR}/${PY_BASE}"
  done

  rm -rf "$SNAPSHOT_DIR"

  echo "Stored actual submitted code/YAML in $COPIES_DIR"
else
  echo "WARNING: run directory '$RUN_DIR' does not exist; frozen code remains in $SNAPSHOT_DIR"
fi
# ================================================================================ #

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

SCRIPT_FINISH_TIMESTAMP="$(date -Iseconds)"
SCRIPT_FINISH_EPOCH="$(date +%s)"
SCRIPT_ELAPSED_SECONDS=$((SCRIPT_FINISH_EPOCH - SCRIPT_START_EPOCH))

echo "Script finish timestamp: $SCRIPT_FINISH_TIMESTAMP"
echo "Script elapsed seconds: $SCRIPT_ELAPSED_SECONDS"
echo "Final status: SUCCESS"

conda deactivate
