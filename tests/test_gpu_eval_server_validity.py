"""Validity suite for the prob5 GPU eval server (`--gpu-eval-server`).

tests/test_gpu_eval_server_parity.py covers the steady-state forward.
This covers everything that could go wrong SILENTLY -- a path that
produces wrong numbers without raising, or a liveness failure that hangs
instead of erroring.

Run:  python3 tests/test_gpu_eval_server_validity.py
      (needs CUDA; skips cleanly without it)

Grouped by risk:
  T1  silent corruption -- wrong targets, no error
  T2  nondeterminism    -- characterise, don't just assert
  T3  robustness        -- must fail loudly, not hang or leak
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
import random

import numpy as np
import torch

import train_prob5 as P5
from model import ProbNetwork
from backgammon_engine import BoardState, opening_roll, switch_turn
from gpu_eval_server import EvalServer, RemoteEvaluator

MODEL = "best_models/cubeless_prob5_512_512_256_256.pt"
TOL = 1e-5
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def load_net():
    ck = torch.load(MODEL, map_location="cpu", weights_only=True)
    net = ProbNetwork(
        hidden_sizes=ck["hidden_sizes"], input_size=ck.get("input_size", 196),
        activation=ck.get("activation", "relu"),
        encoder_name=ck.get("encoder_name", "perspective196"),
        raw_logits=ck.get("raw_logits", False),
    )
    net.load_state_dict(ck.get("model_state_dict") or ck["state_dict"])
    net.eval()
    return net


def spec_of(net, **over):
    s = {
        "net_type": "prob5", "hidden_sizes": net.hidden_sizes,
        "input_size": net.input_size, "activation": net.activation,
        "encoder_name": net.encoder_name, "raw_logits": net.raw_logits,
    }
    s.update(over)
    return s


class Server:
    """Context manager: live EvalServer + N RemoteEvaluators in-process."""

    def __init__(self, net, slots=1, max_rows=8192, spec=None, sd=None):
        self.net, self.slots, self.max_rows = net, slots, max_rows
        self.spec = spec if spec is not None else spec_of(net)
        self.sd = sd if sd is not None else net.state_dict()

    def __enter__(self):
        self.srv = EvalServer(
            mp.get_context("spawn"), num_slots=self.slots,
            n_feat=self.net.input_size, out_dim=P5.NUM_OUTPUTS,
            max_rows=self.max_rows, build_spec=self.spec,
            init_state_dict=self.sd, device="cuda",
        )
        self.srv.start()
        h = self.srv.client_handles()
        self.remotes = []
        for _ in range(self.slots):
            self.remotes.append(RemoteEvaluator(h.free_slots.get(), h))
        return self

    def __exit__(self, *a):
        try:
            self.srv.stop()
        except Exception:
            pass
        return False


def walk_states(net, encode_fn, gpe_fn, n, seed=4242):
    """Real mid-game states from greedy 0-ply self-play."""
    from model import prob5_postprocess, prob5_to_equity
    rng = random.Random(seed)
    out = []
    state, dice = opening_roll(rng)
    while len(out) < n:
        if state.is_game_over():
            state, dice = opening_roll(rng)
            continue
        feats, nxt = gpe_fn(state, dice)
        if len(nxt) == 0:
            state = switch_turn(state)
        else:
            with torch.no_grad():
                p = net(torch.from_numpy(feats))
                if net.raw_logits:
                    p = torch.sigmoid(p)
                eq = prob5_to_equity(prob5_postprocess(p)).numpy()
            for j, ns in enumerate(nxt):
                if ns.is_game_over():
                    eq[j] = -float(ns.game_result())
            ch = nxt[int(np.argmin(eq))]
            out.append(state)
            state = BoardState(points=list(ch.points), bar=list(ch.bar),
                               off=list(ch.off), turn=ch.turn)
        dice = (rng.randint(1, 6), rng.randint(1, 6))
    return out


def real_X(net, n, seed=4242):
    """Encoded features from real self-play states -- the distribution the
    server actually sees.

    Gaussian noise is NOT a substitute here. Real features are board
    encodings: 99.4% of entries are exactly 0 or 1 and the range is [0,1],
    whereas randn is N(0,1). The resulting out-of-distribution activations
    amplify the fp32 CPU-vs-GPU reduction-order difference by orders of
    magnitude -- measured 8.6e-10 .. 8.9e-05 across runs against the 1e-5
    tolerance these tests assert, i.e. a ~1-in-5 spurious failure. On real
    states the same comparisons sit at ~2e-07.

    (t1_raw_logits already did this for the same reason; this just makes the
    pattern reusable.)
    """
    encode_fn, gpe_fn = P5._resolve_engine_fns(net.encoder_name)
    states = walk_states(net, encode_fn, gpe_fn, n, seed=seed)
    return np.stack([encode_fn(s) for s in states]).astype(np.float32)


# ── T1: silent corruption ────────────────────────────────────────────

def t1_weight_push(net):
    """update_weights() must actually take effect AND serve the new net.

    If this silently no-op'd, every round would compute targets from
    round-0 weights: training would flatline while every log line looked
    healthy. Highest-value test in the suite.
    """
    X = real_X(net, 64)
    other = ProbNetwork(hidden_sizes=net.hidden_sizes, input_size=net.input_size,
                        activation=net.activation, encoder_name=net.encoder_name,
                        raw_logits=net.raw_logits)
    torch.manual_seed(99)
    for p in other.parameters():
        torch.nn.init.normal_(p, std=0.05)
    other.eval()

    with Server(net) as S:
        r = S.remotes[0]
        a = r.forward(X)
        S.srv.update_weights(other.state_dict())
        b = r.forward(X)
    with torch.no_grad():
        want_a = net(torch.from_numpy(X)).numpy()
        want_b = other(torch.from_numpy(X)).numpy()
    changed = not np.allclose(a, b, atol=1e-4)
    da, db = np.abs(a - want_a).max(), np.abs(b - want_b).max()
    return record("T1 weight push takes effect and serves new weights",
                  changed and da < TOL and db < TOL,
                  f"changed={changed} pre|d|={da:.2e} post|d|={db:.2e}")


def t1_row_cap_fallback(net):
    """Above max_rows the remote returns None and _net_raw must fall back
    to a local CPU forward -- silently, and correctly."""
    small = 32
    X = real_X(net, small * 4, seed=17)
    with Server(net, max_rows=small) as S:
        r = S.remotes[0]
        assert r.forward(X) is None, "expected None above the row cap"
        with torch.no_grad():                        # as production calls it
            got = P5._net_raw(net, X, "cpu", r)      # must fall back
        under = r.forward(X[:small])                  # at cap: served
    with torch.no_grad():
        want = net(torch.from_numpy(X)).numpy()
    d = np.abs(got.detach().numpy() - want).max()
    du = np.abs(under - want[:small]).max()
    return record("T1 row-cap fallback taken and correct",
                  d < 1e-6 and du < TOL,
                  f"fallback|d|={d:.2e} at-cap|d|={du:.2e}")


def t1_raw_logits(net):
    """raw_logits=True: the server returns raw output, the CALLER applies
    sigmoid. Untested by the parity suite (prod model is raw_logits=False).

    Logits are UNBOUNDED, so an absolute tolerance is meaningless here --
    on out-of-distribution input this net emits logits near -1.8e4, where
    fp32 epsilon alone is ~1e-3 absolute. Assert on RELATIVE error, and on
    the post-sigmoid probabilities, which is what the caller consumes.
    """
    rl = ProbNetwork(hidden_sizes=net.hidden_sizes, input_size=net.input_size,
                     activation=net.activation, encoder_name=net.encoder_name,
                     raw_logits=True)
    rl.load_state_dict(net.state_dict())
    rl.eval()
    # Real encodings, not Gaussian noise: keeps the net in-distribution.
    encode_fn, gpe_fn = P5._resolve_engine_fns(net.encoder_name)
    states = walk_states(net, encode_fn, gpe_fn, 48, seed=101)
    X = np.stack([encode_fn(s) for s in states]).astype(np.float32)
    with Server(rl) as S:
        got = S.remotes[0].forward(X)
    with torch.no_grad():
        want = rl(torch.from_numpy(X)).numpy()
    rel = (np.abs(got - want) / np.maximum(np.abs(want), 1e-6)).max()
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    dp = np.abs(sig(got) - sig(want)).max()
    # GATE on the probability scale. That is what the caller consumes
    # (_net_raw -> sigmoid -> prob5_postprocess -> target), and it is the
    # only scale on which a tolerance is well defined: logits are
    # unbounded ABOVE (absolute error is meaningless) and pass through
    # ZERO (relative error is meaningless -- a logit of 1e-4 carrying a
    # 2.5e-8 absolute error reports rel=2.5e-4 while being numerically
    # perfect). rel is reported as informational only.
    return record("T1 raw_logits=True served raw (caller sigmoids)",
                  dp < TOL,
                  f"prob|d|={dp:.2e} (gate) rel={rel:.2e} (informational) "
                  f"logit range [{want.min():.1f},{want.max():.1f}]")


def t1_activation_short(net):
    """SHORT: prod is relu, so this only asserts the field is load-bearing
    (a wrong activation must change the output), not a full end-to-end leg."""
    tanh_net = ProbNetwork(hidden_sizes=net.hidden_sizes, input_size=net.input_size,
                           activation="tanh", encoder_name=net.encoder_name)
    tanh_net.eval()
    # Real encodings: this assertion is inverted (it wants a LARGE delta),
    # so randn would make it easier to pass. What matters is that a wrong
    # activation corrupts output on the distribution production actually
    # feeds the server.
    X = real_X(net, 32, seed=53)
    with Server(tanh_net, spec=spec_of(tanh_net, activation="relu")) as S:
        wrong = S.remotes[0].forward(X)
    with torch.no_grad():
        want = tanh_net(torch.from_numpy(X)).numpy()
    d = np.abs(wrong - want).max()
    return record("T1 wrong activation in build_spec DOES corrupt output",
                  d > 1e-3, f"max|d|={d:.2e} (must be large)")


# ── The headline check: 1-ply CPU vs GPU ─────────────────────────────

def t1_oneply_cpu_vs_gpu(net, n_states=400):
    """Full 1-ply target equivalence, CPU reference vs GPU server, over
    real self-play states -- including whether GREEDY MOVE CHOICE agrees,
    which is what actually forks a trajectory."""
    from model import prob5_postprocess, prob5_to_equity
    encode_fn, gpe_fn = P5._resolve_engine_fns(net.encoder_name)
    states = walk_states(net, encode_fn, gpe_fn, n_states)

    ref = np.stack([P5._oneply_target_vec(s, net, encode_fn, gpe_fn, "cpu", None)
                    for s in states])
    with Server(net) as S:
        r = S.remotes[0]
        got = np.stack([P5._oneply_target_vec(s, net, encode_fn, gpe_fn, "cpu", r)
                        for s in states])
    d = np.abs(ref - got)

    # Move-choice agreement: the decision the target feeds into.
    rng = random.Random(7)
    disagree = compared = 0
    with Server(net) as S:
        r = S.remotes[0]
        for s in states:
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            feats, nxt = gpe_fn(s, dice)
            if len(nxt) < 2:
                continue
            feats = feats.copy()
            def pick(remote):
                with torch.no_grad():
                    p = P5._net_raw(net, feats, "cpu", remote)
                    if net.raw_logits:
                        p = torch.sigmoid(p)
                    return int(np.argmin(prob5_to_equity(prob5_postprocess(p)).numpy()))
            compared += 1
            if pick(None) != pick(r):
                disagree += 1
    rate = disagree / max(compared, 1)
    ok = d.max() < TOL and rate < 0.01
    return record("1-PLY CPU vs GPU: targets + move choice", ok,
                  f"targets max|d|={d.max():.2e} mean={d.mean():.2e}; "
                  f"move disagreement {disagree}/{compared} ({rate*100:.3f}%)")


# ── T2: nondeterminism characterisation ──────────────────────────────

def t2_batch_composition(net):
    """The server concatenates slots into ONE forward. Row i of A@B depends
    only on row i mathematically, but cuBLAS picks kernels by shape, so
    co-batching can shift the last ulp. Bound it."""
    X = real_X(net, 128, seed=23)
    with Server(net, slots=4) as S:
        alone = S.remotes[0].forward(X)
        # Filler is real too: co-batching shifts kernel choice by SHAPE, but
        # the ulp bound only means something if the neighbours' magnitudes
        # are realistic.
        filler = [real_X(net, 512, seed=31 + i) for i in range(3)]
        import threading
        outs = {}
        def go(i, arr):
            outs[i] = S.remotes[i].forward(arr)
        ts = [threading.Thread(target=go, args=(i + 1, filler[i])) for i in range(3)]
        ts.append(threading.Thread(target=go, args=(0, X)))
        for t in ts: t.start()
        for t in ts: t.join()
        cobatched = outs[0]
    d = np.abs(alone - cobatched).max()
    return record("T2 batch-composition invariance (informational)",
                  d < 1e-4, f"alone vs co-batched max|d|={d:.2e}")


# ── T3: robustness ───────────────────────────────────────────────────

def t3_server_death_raises(net):
    """A dead server must RAISE, not hang forever. Pre-fix, event.wait()
    had no timeout and a dead server wedged every worker silently."""
    os.environ["GPU_EVAL_WAIT_TIMEOUT"] = "8"
    import importlib
    import gpu_eval_server as G
    importlib.reload(G)
    srv = G.EvalServer(mp.get_context("spawn"), num_slots=1,
                       n_feat=net.input_size, out_dim=P5.NUM_OUTPUTS,
                       max_rows=8192, build_spec=spec_of(net),
                       init_state_dict=net.state_dict(), device="cuda")
    srv.start()
    h = srv.client_handles()
    r = G.RemoteEvaluator(h.free_slots.get(), h)
    r.forward(np.random.randn(8, net.input_size).astype(np.float32))  # warm
    srv.proc.kill(); srv.proc.join(timeout=10)
    t0 = time.time()
    try:
        r.forward(np.random.randn(8, net.input_size).astype(np.float32))
        ok, detail = False, "returned instead of raising"
    except RuntimeError as e:
        ok = "did not answer" in str(e)
        detail = f"raised after {time.time()-t0:.1f}s"
    finally:
        os.environ.pop("GPU_EVAL_WAIT_TIMEOUT", None)
        try: srv.stop()
        except Exception: pass
    return record("T3 dead server raises instead of hanging", ok, detail)


def t3_no_orphan_on_sigterm():
    """SIGTERM to the trainer must not leave the server holding GPU memory.

    Observed pre-fix: server reparented to init with 444 MB resident.
    """
    script = (
        "import subprocess,sys,time,os,signal\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "train_prob5.py",
         "--resume", MODEL, "--oneply", "--num-episodes", "100000",
         "--episodes-per-round", "500", "--lr", "1e-8",
         "--workers", "4", "--device", "cpu", "--gpu-eval-server",
         "--save", "/tmp/orphan_probe.pt"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # wait for the server to come up
    up = False
    t0 = time.time()
    while time.time() - t0 < 180:
        line = proc.stdout.readline()
        if not line:
            break
        if "GPU eval server: ON" in line:
            up = True
            break
    if not up:
        proc.kill()
        return record("T3 no orphaned server after SIGTERM", False,
                      "server never came up")
    time.sleep(10)
    proc.terminate()
    proc.wait(timeout=60)
    time.sleep(6)
    stray = subprocess.run(
        "ps -eo cmd | grep -c '[s]pawn_main'", shell=True,
        capture_output=True, text=True).stdout.strip()
    used = subprocess.run(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader",
        shell=True, capture_output=True, text=True).stdout.strip()
    shm = subprocess.run(
        "ls /dev/shm/psm_* 2>/dev/null | wc -l", shell=True,
        capture_output=True, text=True).stdout.strip()
    ok = stray == "0" and shm == "0"
    return record("T3 no orphaned server / shm after SIGTERM", ok,
                  f"stray_procs={stray} gpu={used} shm_segments={shm}")


def main():
    # Seeded for reproducibility. The numeric tests now draw their inputs
    # from real self-play states via real_X() rather than randn, which is
    # what actually fixed their ~1-in-5 spurious failures; this seed only
    # pins the remaining incidental randomness (t3 liveness probes, torch
    # init in t1_weight_push's `other` net).
    np.random.seed(20260818)
    torch.manual_seed(20260818)
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net = load_net()
    print(f"model {MODEL}  hidden={net.hidden_sizes} act={net.activation}\n")

    print("T1 -- silent corruption")
    t1_weight_push(net)
    t1_row_cap_fallback(net)
    t1_raw_logits(net)
    t1_activation_short(net)
    print("\n1-ply CPU vs GPU equivalence")
    t1_oneply_cpu_vs_gpu(net)
    print("\nT2 -- nondeterminism")
    t2_batch_composition(net)
    print("\nT3 -- robustness")
    t3_server_death_raises(net)
    t3_no_orphan_on_sigterm()

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'='*60}\n{len(RESULTS)-n_fail}/{len(RESULTS)} passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} -- {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
