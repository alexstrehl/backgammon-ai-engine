#!/usr/bin/env bash
# CUDA MPS lifecycle helper for the 0-ply matchplay collector.
#
# WHY: 0-ply collection is latency-bound (~8 rows per slot-yield), so W
# driver contexts time-slice instead of overlapping. One shared MPS
# scheduler removes that: measured 230 -> 356 ep/s (+56%) at W=32 on a
# 9980X + RTX 5090, with batch loss and pool size unchanged. The gain is
# contention between contexts, so it scales with worker count (+8% only
# at W=16) and is ~0% at 1-ply.
#
# WHY THIS IS A SCRIPT AND NOT AUTOMATIC: the daemon is HOST-WIDE. It
# changes GPU behaviour for every CUDA client on the box and weakens
# error isolation -- one client's fault can poison the shared context.
# Training code detects and reports it (trainer.py); starting it is the
# box operator's call. The guard below refuses on a box where somebody
# else is already on the GPU.
#
# Usage: ./mps.sh start|stop|status
set -u
DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
LOG="${CUDA_MPS_LOG_DIRECTORY:-${DIR%/*}/mps_log}"
export CUDA_MPS_PIPE_DIRECTORY="$DIR" CUDA_MPS_LOG_DIRECTORY="$LOG"

alive() {  # a pipe dir outlives its daemon; verify the pid too
  [ -S "$DIR/control" ] || return 1
  local p; p=$(cat "$DIR/nvidia-cuda-mps-control.pid" 2>/dev/null) || return 1
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null
}

foreign_gpu_users() {
  # PIDs on the GPU that are not ours. nvidia-smi reports compute apps
  # across all users; owner comes from ps. Empty output == box is ours.
  local me; me=$(id -un)
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    local o; o=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$o" ] && [ "$o" != "$me" ] && echo "$p($o)"
  done
}

case "${1:-status}" in
  start)
    if alive; then echo "MPS already running (pipe dir $DIR)"; exit 0; fi
    f=$(foreign_gpu_users)
    if [ -n "$f" ]; then
      echo "REFUSING to start MPS: other users are on this GPU: $f" >&2
      echo "MPS is host-wide and would change GPU behaviour for their jobs." >&2
      exit 1
    fi
    mkdir -p "$DIR" "$LOG" || exit 1
    nvidia-cuda-mps-control -d || { echo "failed to start MPS" >&2; exit 1; }
    sleep 1
    alive && echo "MPS started (pipe dir $DIR, log $LOG)" || { echo "MPS did not come up" >&2; exit 1; }
    echo "Export in every client shell:  export CUDA_MPS_PIPE_DIRECTORY=$DIR"
    ;;
  stop)
    alive || { echo "MPS not running (pipe dir $DIR)"; exit 0; }
    echo quit | nvidia-cuda-mps-control
    sleep 1
    alive && { echo "MPS still up after quit" >&2; exit 1; } || echo "MPS stopped"
    ;;
  status)
    if alive; then
      echo "MPS: ON   pipe dir $DIR"
      # -f, not -x: the kernel truncates comm to 15 chars so the full
      # server name can never match by name. Bracket guards self-match.
      echo "servers: $(pgrep -f -c "[n]vidia-cuda-mps-server" 2>/dev/null; true)"
    else
      echo "MPS: OFF  (pipe dir $DIR)"
    fi
    ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
