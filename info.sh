#!/bin/bash

if [ "$1" = "--once" ]; then
    jobs=$(squeue -h -u "$USER" -t RUNNING -o "%A" | sort -u)

    if [ -z "$jobs" ]; then
        echo "No currently running jobs for $USER"
        exit 0
    fi

    for jobid in $jobs; do
        echo "============================================================"
        echo "JOB $jobid"
        echo "============================================================"

        echo
        echo "SQUEUE"
        squeue -j "$jobid" -o "%.18i %.24j %.12T %.12M %.12l %.8C %.8D %.20R"

        echo
        echo "LIVE SSTAT"
        sstat -j "$jobid" --parsable2 --format=JobID,NTasks,AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize,MaxDiskRead,MaxDiskWrite | awk -F'|' '
        function gb(x,n,u){
            n=x+0
            u=substr(x,length(x),1)
            if(u=="K") return n/1024/1024
            if(u=="M") return n/1024
            if(u=="G") return n
            if(u=="T") return n*1024
            return n
        }
        NR==1 {
            printf "%-18s %8s %12s %12s %12s %12s %12s %12s %14s %14s\n", "JobID", "NTasks", "AveCPU", "AveRSS_G", "MaxRSS_G", "TotRSS_G", "AveVM_G", "MaxVM_G", "DiskRead", "DiskWrite"
            next
        }
        {
            ave_rss=gb($4)
            max_rss=gb($5)
            ave_vm=gb($6)
            max_vm=gb($7)
            total_rss=ave_rss*$2
            printf "%-18s %8s %12s %12.3f %12.3f %12.3f %12.3f %12.3f %14s %14s\n", $1, $2, $3, ave_rss, max_rss, total_rss, ave_vm, max_vm, $8, $9
        }'

        echo
        echo "SACCT"
        sacct -j "$jobid" --format=JobID%18,JobName%24,State%18,ExitCode%10,Elapsed%12,ReqCPUS%8,AllocCPUS%10,ReqMem%12,MaxRSS%12,AveRSS%12,MaxVMSize%12

        echo
    done

    exit 0
fi

watch -t -n 1 "$0 --once"