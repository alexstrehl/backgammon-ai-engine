#!/usr/bin/env bash
# Training-throughput harness for the collector measurements (SPEC.md).
#
# Runs ONE train_batch.py (prob5-*: train_prob5.py) workload for ROUNDS
# rounds with frozen weights (--lr 1e-9) and prints the inter-round rate
# over rounds 2..N plus the collect/train split from the BG_ROUND_TIMING
# round lines.
#
#   WORKLOAD=cubeful-0ply ROUNDS=30 WORKERS=48 bash experiments/c_collect/bench.sh
#   WORKLOAD=cubeful-0ply-c bash experiments/c_collect/bench.sh
#   WORKLOAD=dmp-0ply POOL=0 bash experiments/c_collect/bench.sh --seed 7
#
# Workloads (all --device cuda; the model is the 4L prod net of the mode).
# The stock workloads run without the trainers' collector defaults: they
# pass --no-c-collect, and BG_TRAIN_CUDA_GRAPH and BG_COLLECT_SHM default
# to 0 (a value the caller exports wins).
#   cubeful-0ply      cubeful-money, --pipeline-collect
#   cubeful-1ply-gpu  cubeful-money, --oneply (auto-routed vectorized GPU
#                     collector) --vectorized-workers $WORKERS,
#                     BG_EVAL_SLOT_DTYPE=fp16. No --pipeline-collect: the
#                     vectorized collector disables it.
#   cubeful-1ply-cpu  cubeful-money, --oneply --no-vectorized-collect
#                     --pipeline-collect (Python CPU collector)
#   cubeless-1ply     cubeless-money, --oneply (vectorized)
#                     --vectorized-workers $WORKERS, BG_VEC_CUBELESS_FAST=1
#   dmp-0ply          dmp, --pipeline-collect
#   prob5-1ply        train_prob5.py, prod prob5 net, --oneply (auto-routed
#                     vectorized GPU collector) --vectorized-workers
#                     $WORKERS, BG_EVAL_SLOT_DTYPE=fp16 (no pipeline)
#   prob5-0ply        train_prob5.py --pipeline-collect (Python CPU collector)
# The -c workloads are the default arms: the C collector (--c-collect
# explicit, so a library that cannot be built is an error, not a Python
# run) with BG_TRAIN_CUDA_GRAPH and BG_COLLECT_SHM from the caller,
# unset = on (p4_sweep.py pins both to 0).
#   cubeful-0ply-c    cubeful-0ply with --c-collect
#   cubeful-1ply-c    cubeful-money, --oneply --c-collect --pipeline-collect
#   cubeless-1ply-c   cubeless-money, --oneply --c-collect --pipeline-collect
#   dmp-0ply-c        dmp-0ply with --c-collect
#   prob5-1ply-c      train_prob5.py --oneply --c-collect --pipeline-collect
#   prob5-0ply-c      train_prob5.py --c-collect --pipeline-collect
#                     (BG_POOL_ON_DEVICE has no effect in train_prob5.py,
#                     whose pool is on --device; BG_COLLECT_SHM applies)
#   (the C backend is --c-collect-backend's default, simd; EXTRA overrides)
#
# Env (defaults in brackets):
#   WORKLOAD   one of the names above (required)
#   ROUNDS     rounds to run [30]; rounds 2..ROUNDS are measured
#   WORKERS    --workers, and --vectorized-workers on the GPU 1-ply paths [48]
#   EPR        --episodes-per-round [4000]
#   BATCH      --batch-size [256]
#   SEED       --seed and --torch-seed [1]
#   MODEL      override the workload's --resume checkpoint
#   EXTRA      extra train_batch.py / train_prob5.py flags (appended; "$@" too)
#   POOL       BG_POOL_ON_DEVICE [1]; 0 keeps the pool on the host
#              (BG_TRAIN_CUDA_GRAPH implies the device pool)
#   PIPELINE   --pipeline-collect [1 on the CPU-collector workloads, else 0];
#              0 gives the true collect time and an uncontended train phase
#   SHIM       1 = LD_PRELOAD c_engine/libfakeintel.so (opt-in) [0]
#   OUT_DIR    run dirs go here [$REPO/runs/c_collect_bench]
#   TAG        run-dir name [<workload>_W<workers>_<utc time>]
#   PYTHON     interpreter [$REPO/.venv/bin/python, else python3]
#   MAX_LOAD   refuse to start above this 1-min load [8]; FORCE=1 skips it
# BG_TRAIN_CUDA_GRAPH, BG_COLLECT_SHM and BG_ROUND_PHASES pass through from
# the caller and are logged; the summary averages the BG_ROUND_PHASES lines.
#
# Rate: episodes of rounds 2..N over the wall time between the round-1 and
# round-N log lines (each line is timestamped as it arrives; "Done in"
# includes worker startup and is not used). Under --pipeline-collect,
# "collect" is only the wait for the round submitted before the previous
# train step, so collect ~0 means the round is train-bound, and 2..N reads
# high by up to one round's tail (round 2's collection starts during round
# 1); the 3..N rate is printed as a check.
#
# The 0-ply rates move by 10-30% with CPU temperature (Tctl, logged) and
# with a few cores of unrelated load (results/baseline.md). Compare
# arms on a quiet box, interleaved (A B A B), never one run each.
#
# The last line is machine-readable:
#   SUMMARY workload=... W=... epr=... rounds=2..N eps=... collect=... train=...
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1

WORKLOAD="${WORKLOAD:-}"
ROUNDS="${ROUNDS:-30}"
WORKERS="${WORKERS:-48}"
EPR="${EPR:-4000}"
BATCH="${BATCH:-256}"
SEED="${SEED:-1}"
EXTRA="${EXTRA:-}"
POOL="${POOL:-1}"
MAX_LOAD="${MAX_LOAD:-8}"
FORCE="${FORCE:-0}"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$REPO/.venv/bin/python" ]; then PYTHON="$REPO/.venv/bin/python"; fi
    PYTHON="${PYTHON:-python3}"
fi

SHIM="${SHIM:-0}"

CF4=best_models/cubeful_money_512_512_256_256.pt
P5=best_models/cubeless_prob5_512_512_256_256.pt
WL_PIPE=0
SCRIPT=train_batch.py
# Stock workloads (WL_STOCK=1): no collector defaults (see the header).
STOCK_FLAGS=(--no-c-collect)
WL_STOCK=0
case "$WORKLOAD" in
    prob5-1ply)
        SCRIPT=train_prob5.py
        WL_MODEL=$P5
        WL_FLAGS=("${STOCK_FLAGS[@]}" --oneply --vectorized-workers "$WORKERS")
        WL_STOCK=1
        WL_ENV=(BG_EVAL_SLOT_DTYPE=fp16) ;;
    prob5-1ply-c)
        SCRIPT=train_prob5.py
        WL_MODEL=$P5
        WL_FLAGS=(--oneply --c-collect)
        WL_PIPE=1
        WL_ENV=() ;;
    prob5-0ply)
        SCRIPT=train_prob5.py
        WL_MODEL=$P5
        WL_FLAGS=("${STOCK_FLAGS[@]}")
        WL_PIPE=1
        WL_STOCK=1
        WL_ENV=() ;;
    prob5-0ply-c)
        SCRIPT=train_prob5.py
        WL_MODEL=$P5
        WL_FLAGS=(--c-collect)
        WL_PIPE=1
        WL_ENV=() ;;
    cubeful-0ply)
        WL_MODEL=$CF4
        WL_FLAGS=("${STOCK_FLAGS[@]}" --game-mode cubeful-money)
        WL_PIPE=1
        WL_STOCK=1
        WL_ENV=() ;;
    cubeful-1ply-gpu)
        WL_MODEL=$CF4
        WL_FLAGS=("${STOCK_FLAGS[@]}" --game-mode cubeful-money --oneply
                  --vectorized-workers "$WORKERS")
        WL_STOCK=1
        WL_ENV=(BG_EVAL_SLOT_DTYPE=fp16) ;;
    cubeful-1ply-cpu)
        WL_MODEL=$CF4
        WL_FLAGS=("${STOCK_FLAGS[@]}" --game-mode cubeful-money --oneply
                  --no-vectorized-collect)
        WL_PIPE=1
        WL_STOCK=1
        WL_ENV=() ;;
    cubeless-1ply)
        WL_MODEL=best_models/cubeless_money_512_512_256_256.pt
        WL_FLAGS=("${STOCK_FLAGS[@]}" --game-mode cubeless-money --oneply
                  --vectorized-workers "$WORKERS")
        WL_STOCK=1
        WL_ENV=(BG_VEC_CUBELESS_FAST=1) ;;
    dmp-0ply)
        WL_MODEL=best_models/dmp_512_512_256_256.pt
        WL_FLAGS=("${STOCK_FLAGS[@]}" --game-mode dmp)
        WL_PIPE=1
        WL_STOCK=1
        WL_ENV=() ;;
    cubeful-0ply-c)
        WL_MODEL=$CF4
        WL_FLAGS=(--game-mode cubeful-money --c-collect)
        WL_PIPE=1
        WL_ENV=() ;;
    cubeful-1ply-c)
        WL_MODEL=$CF4
        WL_FLAGS=(--game-mode cubeful-money --oneply --c-collect)
        WL_PIPE=1
        WL_ENV=() ;;
    cubeless-1ply-c)
        WL_MODEL=best_models/cubeless_money_512_512_256_256.pt
        WL_FLAGS=(--game-mode cubeless-money --oneply --c-collect)
        WL_PIPE=1
        WL_ENV=() ;;
    dmp-0ply-c)
        WL_MODEL=best_models/dmp_512_512_256_256.pt
        WL_FLAGS=(--game-mode dmp --c-collect)
        WL_PIPE=1
        WL_ENV=() ;;
    *)
        echo "ERROR: WORKLOAD must be one of cubeful-0ply cubeful-1ply-gpu" \
             "cubeful-1ply-cpu cubeless-1ply dmp-0ply cubeful-0ply-c" \
             "cubeful-1ply-c cubeless-1ply-c dmp-0ply-c prob5-1ply prob5-1ply-c" \
             "prob5-0ply prob5-0ply-c (got '${WORKLOAD}')" >&2
        exit 2 ;;
esac
PIPELINE="${PIPELINE:-$WL_PIPE}"
if [ "$PIPELINE" = 1 ]; then WL_FLAGS+=(--pipeline-collect); fi
MODEL="${MODEL:-$WL_MODEL}"
[ -f "$MODEL" ] || { echo "ERROR: no model $MODEL" >&2; exit 2; }
[ "$ROUNDS" -ge 3 ] || { echo "ERROR: ROUNDS must be >= 3" >&2; exit 2; }

if [ "$SHIM" = 1 ]; then
    if [ ! -f c_engine/libfakeintel.so ]; then
        bash c_engine/build_fakeintel.sh >/dev/null || exit 1
    fi
    # Training command only: some tools (e.g. hostname) refuse a preload
    # from a home path.
    PRELOAD=(env "LD_PRELOAD=$REPO/c_engine/libfakeintel.so${LD_PRELOAD:+:$LD_PRELOAD}")
    SHIM_TAG="1(md5 $(md5sum c_engine/libfakeintel.so | cut -d' ' -f1))"
else
    PRELOAD=()
    SHIM_TAG=0
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BG_ROUND_TIMING=1 PYTHONUNBUFFERED=1
if [ "$POOL" = 1 ]; then export BG_POOL_ON_DEVICE=1; else unset BG_POOL_ON_DEVICE; fi
for kv in "${WL_ENV[@]}"; do export "${kv?}"; done
if [ "$WL_STOCK" = 1 ]; then
    export BG_TRAIN_CUDA_GRAPH="${BG_TRAIN_CUDA_GRAPH:-0}" BG_COLLECT_SHM="${BG_COLLECT_SHM:-0}"
fi

load1=$(cut -d' ' -f1 /proc/loadavg)
if [ "$FORCE" != 1 ] && awk -v l="$load1" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
    echo "ERROR: 1-min load $load1 > $MAX_LOAD (another measurement running?);" \
         "FORCE=1 to run anyway" >&2
    exit 3
fi

OUT_DIR="${OUT_DIR:-$REPO/runs/c_collect_bench}"
TAG="${TAG:-${WORKLOAD}_W${WORKERS}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUT_DIR/$TAG"
mkdir -p "$RUN_DIR" || exit 1
LOG="$RUN_DIR/train.log"

NUM_EPISODES=$((ROUNDS * EPR))
# shellcheck disable=SC2206
EXTRA_ARR=($EXTRA)
CMD=("${PRELOAD[@]}" "$PYTHON" -u "$SCRIPT" "${WL_FLAGS[@]}"
     --resume "$MODEL" --device cuda
     --workers "$WORKERS" --episodes-per-round "$EPR" --batch-size "$BATCH"
     --num-episodes "$NUM_EPISODES" --lr 1e-9 --seed "$SEED" --torch-seed "$SEED"
     --log-every 1 --save "$RUN_DIR/"
     "${EXTRA_ARR[@]}" "$@")

commit=$(git rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
tctl=$(sensors 2>/dev/null | awk '/^Tctl:/{print $2; exit}')
tctl="${tctl:-n/a}"
dirty=$(git status --porcelain --untracked-files=no 2>/dev/null | head -c1)
{
    echo "# bench.sh workload=$WORKLOAD commit=$commit${dirty:+-dirty}"
    echo "# date=$(date -u +%FT%TZ) load1=$load1 tctl=$tctl rounds=$ROUNDS W=$WORKERS epr=$EPR batch=$BATCH pipeline=$PIPELINE"
    echo "# env: OMP_NUM_THREADS=1 BG_ROUND_TIMING=1" \
         "BG_POOL_ON_DEVICE=${BG_POOL_ON_DEVICE:-unset}" \
         "BG_TRAIN_CUDA_GRAPH=${BG_TRAIN_CUDA_GRAPH:-unset(on)}" \
         "BG_COLLECT_SHM=${BG_COLLECT_SHM:-unset(on)}" \
         "BG_ROUND_PHASES=${BG_ROUND_PHASES:-unset}" \
         "${WL_ENV[*]:-}" "shim=$SHIM_TAG"
    echo "# cmd: ${CMD[*]}"
} | tee "$LOG"

# Timestamp every line as it arrives (python -u: no buffering).
"${CMD[@]}" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
    printf '%s %s\n' "$EPOCHREALTIME" "$line"
done | tee -a "$LOG" | cut -d' ' -f2-
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "ERROR: $SCRIPT exited $rc; log: $LOG" >&2
    exit "$rc"
fi

"$PYTHON" - "$LOG" "$WORKLOAD" "$WORKERS" "$EPR" "$BATCH" > "$RUN_DIR/summary.txt" <<'PY'
import re
import statistics as st
import sys

log, workload = sys.argv[1], sys.argv[2]
workers, epr, batch = (int(v) for v in sys.argv[3:6])
# train_batch.py: "| episodes E/N | pool P ... | collect Cs train Ts";
# train_prob5.py: "| eps E/N | pool P ... | collect Cs train Ts | G g/s".
pat = re.compile(
    r"^(\d+\.\d+) Round\s+(\d+)/(\d+) \| (?:episodes|eps) (\d+)/\d+ \| pool\s+(\d+) .*"
    r"\| collect\s+([\d.]+)s train\s+([\d.]+)s(?: \| \d+ g/s)?\s*$")
rows = []
notes = []
phases = {}                                      # round -> {phase: s}
ppat = re.compile(r"^\d+\.\d+ +phases round (\d+): (.*?) \|")
for ln in open(log):
    m = pat.match(ln)
    if m:
        t, r, _n, eps, pool, c, tr = m.groups()
        rows.append((float(t), int(r), int(eps), int(pool), float(c), float(tr)))
        continue
    m = ppat.match(ln)
    if m:
        kv = m.group(2).split()
        phases[int(m.group(1))] = {k: float(v) for k, v in zip(kv[::2], kv[1::2])}
        continue
    body = ln.split(" ", 1)[-1]
    if re.match(r"\s*(AUTO:|NOTE:|WARN:.*(ectoriz|VRAM|collect)|Vectorized GPU collection|C collection|Done in)", body):
        notes.append(body.rstrip())
for n in notes:
    print("  " + n.strip())
if len(rows) < 3:
    sys.exit(f"ERROR: only {len(rows)} timed round lines in {log} "
             "(BG_ROUND_TIMING line format changed?)")
rows.sort(key=lambda x: x[1])
meas = rows[1:]                                  # rounds 2..N
dt = meas[-1][0] - rows[0][0]
eps = (meas[-1][2] - rows[0][2]) / dt
eps3 = (meas[-1][2] - rows[1][2]) / (meas[-1][0] - rows[1][0])
ivl = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
coll = st.mean(r[4] for r in meas)
train = st.mean(r[5] for r in meas)
pool = st.mean(r[3] for r in meas)
other = st.mean(ivl) - coll - train
share = train / (coll + train)
print(f"\n{workload} W={workers} epr={epr}: rounds 2..{rows[-1][1]} "
      f"({len(meas)} measured, {dt:.1f}s)")
print(f"  ep/s (wall between round lines)  {eps:8.1f}   (rounds 3..N: {eps3:.1f})")
print(f"  round interval s  mean {st.mean(ivl):.3f}  median {st.median(ivl):.3f}  "
      f"min {min(ivl):.3f}  max {max(ivl):.3f}")
print(f"  collect s/round {coll:.3f}  train s/round {train:.3f}  "
      f"other {other:+.3f}  train share {100 * share:.1f}%")
print(f"  pool rows/round {pool:.0f}  ({pool / epr:.1f}/episode, "
      f"{pool / batch:.0f} batches of {batch})")
print(f"  round 1: collect {rows[0][4]:.2f}s train {rows[0][5]:.2f}s (not measured)")
ph = [phases[r[1]] for r in meas if r[1] in phases]
if ph:
    keys = [k for k in ph[0] if all(k in p for p in ph)]
    print("  phases s/round (BG_ROUND_PHASES, rounds 2..N): "
          + " ".join(f"{k} {st.mean(p[k] for p in ph):.4f}" for k in keys))
slow = [(r[1], i) for r, i in zip(rows[1:3], ivl[:2]) if i > 1.5 * st.median(ivl)]
if slow:
    print("  NOTE: slow early rounds (> 1.5x median interval): "
          + ", ".join(f"round {r} {i:.2f}s" for r, i in slow))
print(f"SUMMARY workload={workload} W={workers} epr={epr} rounds=2..{rows[-1][1]} "
      f"eps={eps:.1f} eps3={eps3:.1f} collect={coll:.3f} train={train:.3f} share={share:.3f} "
      f"pool={pool:.0f} log={log}")
PY
prc=$?
cat "$RUN_DIR/summary.txt"
exit "$prc"
