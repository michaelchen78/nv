#!/bin/bash

if [ "$1" = "--once" ]; then
    jobs=$(squeue -h -u "$USER" -t RUNNING,PENDING -o "%A" | sort -u)

    if [ -z "$jobs" ]; then
        echo "No currently running or pending jobs for $USER"
        exit 0
    fi

    echo "SQUEUE"
    squeue -u "$USER" -t RUNNING,PENDING \
        -o "%.18i %.14P %.24j %.12T %.12M %.12l %.8C %.8D %.12m %.8c %.30R"

    echo
    echo "MEMORY"
    printf "%-10s %-24s %-14s %-14s\n" "JOBID" "NAME" "AveRSS_GB" "MaxRSS_GB"

    for jobid in $jobs; do
        name=$(squeue -h -j "$jobid" -o "%.24j" | head -n 1)
        state=$(squeue -h -j "$jobid" -o "%T" | head -n 1)

        if [ "$state" != "RUNNING" ]; then
            printf "%-10s %-24s %-14s %-14s\n" "$jobid" "$name" "PENDING" "PENDING"
            continue
        fi

        mem=$(
            sstat -j "$jobid" --parsable2 \
                --format=JobID,NTasks,AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize,MaxDiskRead,MaxDiskWrite 2>/dev/null |
            awk -F'|' '
                NR == 1 { next }

                function bad(x) {
                    return x == "" || x == "N/A" || x == "0" || x == "0K" || x == "0M" || x == "0G" || x == "0T"
                }

                function to_gb(x, n, u) {
                    if (bad(x)) return -1

                    n = x + 0
                    u = substr(x, length(x), 1)

                    if (u == "K") return n / 1024 / 1024
                    if (u == "M") return n / 1024
                    if (u == "G") return n
                    if (u == "T") return n * 1024

                    return n / 1024 / 1024
                }

                BEGIN {
                    best = -1
                }

                {
                    ave_gb = to_gb($4)
                    max_gb = to_gb($5)

                    if (max_gb > best) {
                        best = max_gb
                        best_ave = ave_gb
                        best_max = max_gb
                    }
                }

                END {
                    if (best > 0) {
                        printf "%.3f %.3f\n", best_ave, best_max
                    }
                }
            '
        )

        if [ -z "$mem" ]; then
            printf "%-10s %-24s %-14s %-14s\n" "$jobid" "$name" "N/A" "N/A"
        else
            ave=$(echo "$mem" | awk '{print $1}')
            max=$(echo "$mem" | awk '{print $2}')
            printf "%-10s %-24s %-14s %-14s\n" "$jobid" "$name" "$ave" "$max"
        fi
    done

    exit 0
fi

watch -t -n 1.0 "$0 --once"