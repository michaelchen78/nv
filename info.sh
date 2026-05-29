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
        squeue -j "$jobid" -o "%.18i %.14P %.24j %.12T %.12M %.12l %.8C %.8D %.20R"

        echo
        echo "LIVE SSTAT"
        sstat -j "$jobid" --parsable2 --format=JobID,NTasks,AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize,MaxDiskRead,MaxDiskWrite | awk -F'|' '
        function gb(x,n,u){
            n=x+0
            u=substr(x,length(x),1)
            if(u=="K") return n/1024/1024
            if(u=="M") return n/1024
            if(u=="G") return n*1024
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
        sacct -j "$jobid" --format=JobID%18,Partition%14,JobName%24,State%18,ExitCode%10,Elapsed%12,ReqCPUS%8,AllocCPUS%10,ReqMem%12,MaxRSS%12,AveRSS%12,MaxVMSize%12

        echo
    done

    exit 0
fi

if [ "$1" = "--watch" ]; then
    watch -t -n 1 "$0 --once"
    exit 0
fi

viewer_py=$(mktemp /tmp/slurm_info_viewer.XXXXXX.py)

cleanup() {
    rm -f "$viewer_py"
}
trap cleanup EXIT INT TERM

cat > "$viewer_py" <<'PY'
import curses
import subprocess
import sys
import time

SCRIPT = sys.argv[1]
REFRESH_SECONDS = 1.0


def get_snapshot():
    try:
        result = subprocess.run(
            ["/bin/bash", SCRIPT, "--once"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        return result.stdout.splitlines()
    except Exception as e:
        return ["ERROR running {} --once:".format(SCRIPT), str(e)]


def wrap_lines(raw_lines, width):
    usable_width = max(1, width - 1)
    wrapped = []

    for line in raw_lines:
        if line == "":
            wrapped.append("")
            continue

        start = 0
        while start < len(line):
            wrapped.append(line[start:start + usable_width])
            start += usable_width

    if not wrapped:
        wrapped = [""]

    return wrapped


def clamp_top(top, lines, body_height):
    max_top = max(0, len(lines) - body_height)
    return max(0, min(top, max_top))


def draw(stdscr, lines, top, last_update, paused):
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    body_height = max(1, height - 2)
    top = clamp_top(top, lines, body_height)

    for row in range(body_height):
        idx = top + row
        if idx >= len(lines):
            break

        try:
            stdscr.addstr(row, 0, lines[idx][:max(1, width - 1)])
        except curses.error:
            pass

    first = top + 1 if lines else 0
    last = min(top + body_height, len(lines))
    scrollable = "YES" if len(lines) > body_height else "NO"

    status = (
        "updated {} | rows {}-{}/{} | scrollable={} | "
        "j/k scroll | u/d page | g top | G bottom | p pause={} | r refresh | q quit"
    ).format(
        last_update,
        first,
        last,
        len(lines),
        scrollable,
        "on" if paused else "off",
    )

    if len(status) > width - 1:
        status = status[:width - 1]

    try:
        stdscr.addstr(height - 1, 0, status, curses.A_REVERSE)
    except curses.error:
        pass

    stdscr.refresh()
    return top


def main(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(100)

    raw_lines = []
    lines = []
    top = 0
    paused = False
    last_refresh = 0
    last_update = "never"
    last_width = None

    while True:
        height, width = stdscr.getmaxyx()
        body_height = max(1, height - 2)
        max_top = max(0, len(lines) - body_height)

        now = time.time()

        if last_width != width:
            last_width = width
            lines = wrap_lines(raw_lines, width)
            top = clamp_top(top, lines, body_height)

        if (not paused) and (now - last_refresh >= REFRESH_SECONDS):
            raw_lines = get_snapshot()
            lines = wrap_lines(raw_lines, width)
            top = clamp_top(top, lines, body_height)
            last_update = time.strftime("%Y-%m-%d %H:%M:%S")
            last_refresh = now

        top = draw(stdscr, lines, top, last_update, paused)

        height, width = stdscr.getmaxyx()
        body_height = max(1, height - 2)
        max_top = max(0, len(lines) - body_height)

        key = stdscr.getch()

        if key == -1:
            continue

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("r"), ord("R")):
            raw_lines = get_snapshot()
            lines = wrap_lines(raw_lines, width)
            top = clamp_top(top, lines, body_height)
            last_update = time.strftime("%Y-%m-%d %H:%M:%S")
            last_refresh = time.time()

        elif key in (ord("p"), ord("P")):
            paused = not paused

        elif key in (curses.KEY_UP, ord("k"), ord("K")):
            paused = True
            top = max(0, top - 1)

        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            paused = True
            top = min(max_top, top + 1)

        elif key in (curses.KEY_PPAGE, ord("u"), ord("U"), ord("b"), ord("B")):
            paused = True
            top = max(0, top - body_height)

        elif key in (curses.KEY_NPAGE, ord("d"), ord("D"), ord(" ")):
            paused = True
            top = min(max_top, top + body_height)

        elif key == ord("g"):
            paused = True
            top = 0

        elif key == ord("G"):
            paused = True
            top = max_top


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
PY

python3 "$viewer_py" "$0"