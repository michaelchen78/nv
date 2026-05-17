#!/bin/bash

if [ "$1" = "--once" ]; then
    jobs=$(squeue -h -u "$USER" -t RUNNING,PENDING -o "%A" | sort -u)

    if [ -z "$jobs" ]; then
        echo "No currently running or pending jobs for $USER"
        exit 0
    fi

    echo "SQUEUE"
    squeue -u "$USER" -t RUNNING,PENDING \
        -o "%.18i %.24j %.12T %.12M %.12l %.8C %.8D %.12m %.8c %.30R"

    exit 0
fi

watch -t -n 1 "$0 --once"