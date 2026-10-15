#!/usr/bin/env python3
"""Throughput sweep driver: runs bench.sh arms one at a time, in the given
order (interleave arms A B A B in the plan), each only after the 1-min
load has fallen below --max-load, and logs one TSV row per run.

    python experiments/c_collect/p4_sweep.py --plan zeroply --out DIR
    python experiments/c_collect/p4_sweep.py --plan oneply --dry-run --out DIR

Per run, besides bench.sh's SUMMARY: the 1-min load and the busy cores
(10 s) at the start, and the CPU used by processes outside the run's own
process tree ("foreign cores": /proc utime+stime deltas, sampled every
5 s; the mean and max of the samples over the run). A start waits for
load1 < --max-load AND busy cores < --max-busy; a run whose mean foreign
load exceeds --max-foreign is logged with clean=0 and re-run (up to
--retries times). bench.sh is called with MAX_LOAD = --max-load.
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "experiments", "c_collect", "bench.sh")
HZ = os.sysconf("SC_CLK_TCK")


def arm(label, workload, workers, pipeline=None, extra=""):
    # The arms pin the graphed step and the shm transport off (eager step,
    # pool pipe), the configuration of results/p4_throughput.md.
    env = {"WORKLOAD": workload, "WORKERS": str(workers),
           "BG_TRAIN_CUDA_GRAPH": "0", "BG_COLLECT_SHM": "0"}
    if pipeline is not None:
        env["PIPELINE"] = str(pipeline)
    if extra:
        env["EXTRA"] = extra
    return label, env


def plan_zeroply():
    """cubeful 0-ply: C at W 48/32/16 pipelined and not, stock W=48 both;
    a second pass in reverse order."""
    p = [arm("S48p", "cubeful-0ply", 48), arm("C48p", "cubeful-0ply-c", 48),
         arm("S48u", "cubeful-0ply", 48, 0), arm("C48u", "cubeful-0ply-c", 48, 0),
         arm("C32p", "cubeful-0ply-c", 32), arm("C32u", "cubeful-0ply-c", 32, 0),
         arm("C16p", "cubeful-0ply-c", 16), arm("C16u", "cubeful-0ply-c", 16, 0)]
    return p + p[::-1]


def plan_oneply():
    """cubeful 1-ply: C at W 48/32/60/64 against the GPU recipe at W=48."""
    p = [arm("G48", "cubeful-1ply-gpu", 48), arm("C48", "cubeful-1ply-c", 48),
         arm("C32", "cubeful-1ply-c", 32), arm("C60", "cubeful-1ply-c", 60),
         arm("C64", "cubeful-1ply-c", 64)]
    return p + p[::-1]


def plan_other():
    """cubeless 1-ply and dmp 0-ply at W=48, stock and C, A B A B."""
    cl = [arm("G48", "cubeless-1ply", 48), arm("C48", "cubeless-1ply-c", 48)]
    dm = [arm("S48p", "dmp-0ply", 48), arm("C48p", "dmp-0ply-c", 48)]
    return cl + cl + dm + dm


PLANS = {"zeroply": plan_zeroply, "oneply": plan_oneply, "other": plan_other}


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def proc_table():
    """{pid: (ppid, cpu ticks)} of every live process."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
            out[int(d)] = (int(rest[1]), int(rest[11]) + int(rest[12]))
        except (OSError, IndexError, ValueError):
            continue
    return out


def tree(table, root):
    kids = {}
    for pid, (ppid, _) in table.items():
        kids.setdefault(ppid, []).append(pid)
    seen, stack = set(), [root]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(kids.get(p, ()))
    return seen


class ForeignSampler(threading.Thread):
    """Cores used outside the process tree of `root`, every `dt` s."""

    def __init__(self, root, dt=5.0):
        super().__init__(daemon=True)
        self.root, self.dt, self.samples = root, dt, []
        self.stop = threading.Event()

    def run(self):
        me = os.getpid()
        prev, t_prev = proc_table(), time.monotonic()
        while not self.stop.wait(self.dt):
            cur, t = proc_table(), time.monotonic()
            mine = tree(cur, self.root) | {me}
            ticks = sum(c - prev[p][1] for p, (_, c) in cur.items()
                        if p not in mine and p in prev and c >= prev[p][1])
            self.samples.append(ticks / HZ / (t - t_prev))
            prev, t_prev = cur, t


def busy_cores(dt=10.0):
    """Cores in use on the whole box over the next `dt` s (between runs
    every process is foreign)."""
    a, t0 = proc_table(), time.monotonic()
    time.sleep(dt)
    b, t1 = proc_table(), time.monotonic()
    ticks = sum(c - a[p][1] for p, (_, c) in b.items() if p in a and c >= a[p][1])
    return ticks / HZ / (t1 - t0)


def wait_quiet(max_load, max_busy, poll=10.0):
    """Until load1 < max_load and fewer than max_busy cores are busy
    (another measurement shows up in the second test long before
    the load average moves). The busy cores are sampled only once the
    load test passes."""
    t0 = time.monotonic()
    last = -1e9
    while True:
        l1 = load1()
        busy = busy_cores() if l1 < max_load else None
        if busy is not None and busy < max_busy:
            return l1, busy, time.monotonic() - t0
        if time.monotonic() - last > 60:
            b = "not sampled (load test failed)" if busy is None else f"{busy:.1f}"
            print(f"  waiting: load1 {l1:.2f} (max {max_load}), busy cores {b} "
                  f"(max {max_busy})", flush=True)
            last = time.monotonic()
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", choices=sorted(PLANS), required=True)
    ap.add_argument("--out", required=True, help="run dirs and sweep.tsv go here")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--epr", type=int, default=4000)
    ap.add_argument("--max-load", type=float, default=4.0)
    ap.add_argument("--max-busy", type=float, default=3.0,
                    help="start only below this many busy cores (10 s sample)")
    ap.add_argument("--max-foreign", type=float, default=2.0,
                    help="a run whose mean foreign load exceeds this many cores is "
                         "logged as contaminated and re-run")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the plan's first N runs (resume a stopped sweep)")
    ap.add_argument("--retries", type=int, default=3,
                    help="re-runs of a contaminated arm before moving on")
    ap.add_argument("--only", default="", help="comma list of labels to run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    plan = PLANS[args.plan]()
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p[0] in keep]
    os.makedirs(args.out, exist_ok=True)
    tsv = os.path.join(args.out, "sweep.tsv")
    if not os.path.exists(tsv):
        with open(tsv, "w") as f:
            f.write("plan\tlabel\tworkload\tW\tpipeline\tload1\tbusy0\twaited_s"
                    "\tforeign_mean\tforeign_max\tclean\teps\teps3\tcollect\ttrain"
                    "\tshare\tpool\tlog\n")
    for i, (label, env) in enumerate(plan):
        if i < args.skip:
            continue
        attempt = 0
        while attempt <= args.retries:
            tag = f"{args.plan}_{i:02d}_{label}_{env['WORKLOAD']}_W{env['WORKERS']}"
            if attempt:
                tag += f"_retry{attempt}"
            print(f"[{i + 1}/{len(plan)}] {tag}", flush=True)
            if args.dry_run:
                break
            r = run_one(args, tsv, tag, label, env)
            if r == "refused":
                continue            # load rose after the gate: wait again, no retry used
            if r != "contaminated":
                break
            attempt += 1


def run_one(args, tsv, tag, label, env):
    """One bench.sh run: "clean", "contaminated" (re-run it), "refused"
    (bench.sh's load check, exit 3) or "failed"."""
    l1, busy0, waited = wait_quiet(args.max_load, args.max_busy)
    e = dict(os.environ, ROUNDS=str(args.rounds), EPR=str(args.epr),
             OUT_DIR=args.out, TAG=tag, MAX_LOAD=str(args.max_load), **env)
    p = subprocess.Popen(["bash", BENCH], cwd=REPO, env=e, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    samp = ForeignSampler(p.pid)
    samp.start()
    out = p.stdout.read()
    p.wait()
    samp.stop.set()
    samp.join()
    with open(os.path.join(args.out, f"{tag}.out"), "w") as f:
        f.write(out)
    m = re.search(r"^SUMMARY (.*)$", out, re.M)
    if p.returncode or not m:
        if p.returncode == 3:
            print("  refused by bench.sh's load check; waiting again", flush=True)
            return "refused"
        print(f"  FAILED rc={p.returncode}:\n{out[-2000:]}", flush=True)
        return "failed"
    kv = dict(x.split("=", 1) for x in m.group(1).split())
    fs = samp.samples or [float("nan")]
    f_mean = sum(fs) / len(fs)
    clean = f_mean <= args.max_foreign
    row = [args.plan, label, env["WORKLOAD"], env["WORKERS"], env.get("PIPELINE", "dflt"),
           f"{l1:.2f}", f"{busy0:.1f}", f"{waited:.0f}", f"{f_mean:.2f}", f"{max(fs):.2f}",
           str(int(clean)), kv["eps"], kv["eps3"], kv["collect"], kv["train"], kv["share"],
           kv["pool"], kv["log"]]
    with open(tsv, "a") as f:
        f.write("\t".join(row) + "\n")
    print(f"  load1 {l1:.2f} foreign {f_mean:.2f} (max {max(fs):.2f}) cores"
          f"{'' if clean else ' CONTAMINATED, re-run'}: ep/s {kv['eps']} "
          f"collect {kv['collect']} train {kv['train']} share {kv['share']}", flush=True)
    return "clean" if clean else "contaminated"


if __name__ == "__main__":
    sys.exit(main())
