#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  ./copy.sh [run_id]"
    echo "  ./copy.sh -m [run_id]"
    exit 1
}

multi=false

if [ "$#" -eq 1 ]; then
    run_id="$1"
elif [ "$#" -eq 2 ] && [ "$1" = "-m" ]; then
    multi=true
    run_id="$2"
else
    usage
fi

remote_user="f006vh6"
remote_host="discovery.dartmouth.edu"
remote="${remote_user}@${remote_host}"

remote_dir="pycce_runs/data/${run_id}/"
local_dir="./data/${run_id}/"

mkdir -p "$local_dir"

if [ "$multi" = false ]; then
    # normal run: csvs are directly in data/[run_id]/
    rsync -av \
        --include='*.csv' \
        --exclude='*' \
        "${remote}:${remote_dir}" \
        "$local_dir"
else
    # multi run: csvs are in data/[run_id]/subdir/*.csv
    # preserves subdirectory names, copies only csvs, skips unfinished empty/no-csv dirs
    rsync -av \
        --prune-empty-dirs \
        --include='*/' \
        --include='*.csv' \
        --exclude='*' \
        "${remote}:${remote_dir}" \
        "$local_dir"
fi