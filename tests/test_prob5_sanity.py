"""prob_agent.prob5_sanity zeroes the gammons / backgammons the board
makes impossible, per gnubg 1.08 SanityCheck (eval.c). On for Prob5Jan
(JanowskiAgent), off by default for plain prob5 (ProbAgent board_sanity).

Reference: `_gnubg_ref`, a line-by-line reimplementation of SanityCheck's
"impossible" rules and nesting clamps on gnubg's TanBoard (side 1 = on
roll). `TestRealGnubg` compiles SanityCheck itself from the gnubg
source when present. `TestCCopiesParity`: the C copies (bg_engine.c,
nn_eval.c) give prob_agent's masks and values.
"""

import os
import random
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from backgammon_engine import (BoardState, WHITE, BLACK, get_legal_plays,
                               switch_turn, opening_roll, is_race)
from encoding import get_encoder
from model import prob5_postprocess
import prob_agent
from prob_agent import ProbAgent, prob5_sanity

ENC = get_encoder("perspective196")
GNUBG_EVAL_C = os.path.expanduser("~/gnubg-1.08.003/eval.c")


# ── gnubg frame ──────────────────────────────────────────────────────


def _anboard(state):
    """gnubg TanBoard: [0] = opponent, [1] = on roll; index = the
    owner's point - 1, 24 = bar."""
    a = [[0] * 25, [0] * 25]
    for side, color in ((1, state.turn), (0, 1 - state.turn)):
        for i in range(24):
            v = state.points[i] if color == WHITE else -state.points[23 - i]
            if v > 0:
                a[side][i] = v
        a[side][24] = state.bar[color]
    return a


def _state(an, turn):
    """BoardState from a TanBoard with `turn` on roll (None if two
    sides share a point)."""
    pts = [0] * 24
    bar = [0, 0]
    off = [0, 0]
    for side, color in ((1, turn), (0, 1 - turn)):
        for i in range(24):
            n = an[side][i]
            if not n:
                continue
            j = i if color == WHITE else 23 - i
            if pts[j]:
                return None
            pts[j] = n if color == WHITE else -n
        bar[color] = an[side][24]
        off[color] = 15 - sum(an[side])
    return BoardState(points=pts, bar=bar, off=off, turn=turn)


def _counts(an):
    """SanityCheck's ac, anBack, anCross, anGammonCross (eval.c)."""
    ac, back, cross, gcross = [0, 0], [0, 0], [0, 0], [1, 1]
    for j in range(2):
        nciq = 0
        for i in range(6):
            if an[j][i]:
                back[j] = i
                nciq += an[j][i]
        ac[j] = cross[j] = nciq
        for q, (lo, hi) in enumerate(((6, 12), (12, 18), (18, 24)), start=1):
            nciq = 0
            for i in range(lo, hi):
                if an[j][i]:
                    back[j] = i
                    nciq += an[j][i]
            ac[j] += nciq
            cross[j] += (q + 1) * nciq
            gcross[j] += q * nciq
        if an[j][24]:
            back[j] = 24
            ac[j] += an[j][24]
            cross[j] += 5 * an[j][24]
            gcross[j] += 4 * an[j][24]
    return ac, back, cross, gcross


def _gnubg_ref(an, out):
    """SanityCheck's impossible rules + clamps (float32, as gnubg)."""
    o = np.array(out, dtype=np.float32)
    ac, back, cross, gcross = _counts(an)
    contact = back[0] + back[1] >= 24
    if ac[0] < 15:
        o[1] = o[2] = 0.0
    elif not contact:
        if cross[1] > 8 * gcross[0]:
            o[1] = 0.0
        if back[0] < 18:
            o[2] = 0.0
    if ac[1] < 15:
        o[3] = o[4] = 0.0
    elif not contact:
        if cross[0] > 8 * gcross[1] - 4:
            o[3] = 0.0
        if back[1] < 18:
            o[4] = 0.0
    if o[1] > o[0]:
        o[1] = o[0]
    lose = np.float32(1.0) - o[0]
    if o[3] > lose:
        o[3] = lose
    if o[2] > o[1]:
        o[2] = o[1]
    if o[4] > o[3]:
        o[4] = o[3]
    return o


def _ours(states, probs):
    x = torch.from_numpy(np.stack([ENC.encode(s) for s in states]))
    return prob5_sanity(torch.as_tensor(probs, dtype=torch.float32), x).numpy()


# ── position generators ──────────────────────────────────────────────


def _playout_states(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        state, dice = opening_roll(rng)
        while not state.is_game_over():
            plays = get_legal_plays(state, dice)
            state = rng.choice(plays)[1] if plays else switch_turn(state)
            if not state.is_game_over():
                out.append(state)
            dice = (rng.randint(1, 6), rng.randint(1, 6))
    return out[:n]


def _spread(rng, n, lo, hi, an_side):
    for _ in range(n):
        an_side[rng.randint(lo, hi)] += 1


def _random_board(rng, kind):
    """kind: 'race' (no contact), 'contact', 'bar'. Random off counts."""
    for _ in range(1000):
        an = [[0] * 25, [0] * 25]
        n = [rng.choice([15, 15, rng.randint(1, 15)]) for _ in range(2)]
        if kind == "race":
            a = rng.randint(0, 23)          # my rearmost bound
            b = rng.randint(0, 23 - a)      # opp's: a + b < 24
            for side, hi in ((1, a), (0, b)):
                _spread(rng, n[side], 0, hi, an[side])
        else:
            for side in (0, 1):
                _spread(rng, n[side], 0, 23, an[side])
            if kind == "bar":
                for side in (0, 1):
                    k = min(rng.randint(0, 3), n[side])
                    for _ in range(k):
                        pts = [i for i in range(24) if an[side][i]]
                        an[side][rng.choice(pts)] -= 1
                        an[side][24] += 1
        st = _state(an, rng.choice([WHITE, BLACK]))
        if st is not None and _anboard(st) == an:
            return st
    raise RuntimeError("no board")


def _all_boards():
    rng = random.Random(7)
    boards = _playout_states(1500, seed=3)
    for kind, k in (("race", 3000), ("contact", 600), ("bar", 600)):
        boards += [_random_board(rng, kind) for _ in range(k)]
    boards += _hand_built()
    return boards


def _hand_built():
    """Crossover bounds met exactly / exceeded by one, a race with
    backgammon possible, contact with checkers off and on the bar."""
    def pts(**kw):
        a = [0] * 25
        for k, v in kw.items():
            a[int(k[1:])] = v
        return a
    home15 = pts(i0=3, i1=3, i2=3, i3=3, i4=3)       # anGammonCross 1
    out = [
        # win gammon: my anCross 8 = 8 * 1 (possible), 9 (impossible)
        [home15, pts(i6=4)],
        [home15, pts(i6=4, i0=1)],
        # lose gammon: opp anCross 4 = 8 * 1 - 4 (possible), 5 (impossible)
        [pts(i0=4), pts(i0=15)],
        [pts(i0=5), pts(i0=15)],
        [pts(i6=2), pts(i0=15)],
        # race, opp checkers in my home: backgammon possible
        [pts(i20=1, i22=14), pts(i0=15)],
        # contact, both sides have borne off, bar
        [pts(i0=2, i1=2, i2=2, i3=2, i4=2, i5=2, i23=2),
         pts(i12=3, i13=3, i14=3, i15=3, i24=1)],
    ]
    states = []
    for an in out:
        for turn in (WHITE, BLACK):
            st = _state(an, turn)
            assert st is not None and _anboard(st) == an
            states.append(st)
    return states


@pytest.fixture(scope="module")
def boards():
    return _all_boards()


# ── helper vs reference ──────────────────────────────────────────────


class TestHelperVsGnubgRef:

    @pytest.mark.parametrize("impl", ["c", "numpy"])
    def test_random_probs(self, boards, impl, monkeypatch):
        if impl == "numpy":
            monkeypatch.setattr(prob_agent, "_BG_FAST_AVAILABLE", False)
        elif not prob_agent._BG_FAST_AVAILABLE:
            pytest.skip("bg_fast not built")
        rng = np.random.default_rng(0)
        probs = rng.random((len(boards), 5)).astype(np.float32)
        ours = _ours(boards, probs)
        for s, p, o in zip(boards, probs, ours):
            np.testing.assert_array_equal(o, _gnubg_ref(_anboard(s), p))

    def test_c_encoder_features(self, boards):
        bg_fast = pytest.importorskip("bg_fast")
        rng = np.random.default_rng(1)
        probs = torch.from_numpy(rng.random((len(boards), 5)).astype(np.float32))
        x_py = torch.from_numpy(np.stack([ENC.encode(s) for s in boards]))
        x_c = torch.from_numpy(np.stack([bg_fast.encode_state(s) for s in boards]))
        assert torch.equal(prob5_sanity(probs, x_py), prob5_sanity(probs, x_c))

    def test_contact_matches_is_race(self, boards):
        for s in boards:
            _, back, _, _ = _counts(_anboard(s))
            assert (back[0] + back[1] < 24) == is_race(s)

    def test_coverage(self, boards):
        """Every rule fires and stays silent somewhere; the crossover
        bounds are hit exactly and missed by one."""
        hits = {k: set() for k in ("opp_off", "my_off", "wg", "wbg", "lg", "lbg")}
        edge_w, edge_l = set(), set()
        for s in boards:
            ac, back, cross, gc = _counts(_anboard(s))
            race = back[0] + back[1] < 24
            hits["opp_off"].add(ac[0] < 15)
            hits["my_off"].add(ac[1] < 15)
            if race and ac[0] == 15:
                hits["wg"].add(cross[1] > 8 * gc[0])
                hits["wbg"].add(back[0] < 18)
                edge_w.add(cross[1] - 8 * gc[0])
            if race and ac[1] == 15:
                hits["lg"].add(cross[0] > 8 * gc[1] - 4)
                hits["lbg"].add(back[1] < 18)
                edge_l.add(cross[0] - (8 * gc[1] - 4))
        for k, v in hits.items():
            assert v == {True, False}, k
        assert {0, 1} <= edge_w and {0, 1} <= edge_l
        assert any(s.bar[0] or s.bar[1] for s in boards)
        assert {s.turn for s in boards} == {WHITE, BLACK}

    def test_untouched_when_nothing_impossible(self, boards):
        rng = np.random.default_rng(2)
        probs = torch.from_numpy(rng.random((len(boards), 5)).astype(np.float32))
        ours = _ours(boards, probs)
        plain = prob5_postprocess(probs).numpy()
        for s, o, p in zip(boards, ours, plain):
            ac, back, _, _ = _counts(_anboard(s))
            if ac == [15, 15] and back[0] + back[1] >= 24:
                np.testing.assert_array_equal(o, p)


# ── real gnubg SanityCheck ───────────────────────────────────────────


_HARNESS_HEAD = r"""
#include <stdio.h>
typedef unsigned int TanBoard[2][25];
typedef struct { int nPoints, nChequers; } bcx;
static bcx *pbc1 = 0;
#define unlikely(x) (x)
#define g_assert(x)
#define OUTPUT_WIN 0
#define OUTPUT_WINGAMMON 1
#define OUTPUT_WINBACKGAMMON 2
#define OUTPUT_LOSEGAMMON 3
#define OUTPUT_LOSEBACKGAMMON 4
#define NUM_OUTPUTS 5
#define MaxTurns(id) (-1)
#define PositionBearoff(a, b, c) 0
"""

_HARNESS_MAIN = r"""
int main(void) {
    TanBoard b; float o[5];
    for (;;) {
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 25; i++)
                if (scanf("%u", &b[j][i]) != 1) return 0;
        for (int k = 0; k < 5; k++) scanf("%a", &o[k]);
        SanityCheck((const unsigned int (*)[25])b, o);
        printf("%a %a %a %a %a\n", o[0], o[1], o[2], o[3], o[4]);
    }
}
"""


# ── the two C copies vs prob_agent ──────────────────────────────────

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_C_HARNESS = r"""
#include "nn_eval.c"
int prob5_impossible_mask(const float *x, int n, unsigned char *mask);  /* bg_engine.c */
void h_engine_mask(const float *x, int n, unsigned char *mask) { prob5_impossible_mask(x, n, mask); }
void h_nn_mask(const float *x, int n, unsigned char *mask) {
    for (int r = 0; r < n; r++) {
        float p[5] = {1, 1, 1, 1, 1};
        prob5_sanity_zero(x + 196 * r, p);
        for (int j = 0; j < 4; j++) mask[4 * r + j] = p[j + 1] == 0.0f;
    }
}
void h_nn_reduce(const float *x, const float *raw, int n, float *out) {
    for (int r = 0; r < n; r++) prob5_reduce(x + 196 * r, raw + 5 * r, 1, out + 5 * r);
}
"""


def _edge_boards():
    """Bearoffs (none off, some off, one checker each), gammon-impossible
    (a side has borne off) and backgammon-impossible races."""
    def pts(**kw):
        a = [0] * 25
        for k, v in kw.items():
            a[int(k[1:])] = v
        return a
    ans = [
        [pts(i0=3, i1=3, i2=3, i3=3, i4=3), pts(i0=3, i1=3, i2=3, i3=3, i5=3)],
        [pts(i0=2, i3=1), pts(i5=4, i1=2)],
        [pts(i0=1), pts(i5=1)],
        [pts(i5=15), pts(i0=1)],
        [pts(i10=15), pts(i0=15)],
        [pts(i17=15), pts(i5=10, i1=5)],
        [pts(i18=15), pts(i4=15)],
        [pts(i22=1, i0=14), pts(i0=15)],
    ]
    out = []
    for an in ans:
        for a in (an, an[::-1]):
            for turn in (WHITE, BLACK):
                st = _state(a, turn)
                assert st is not None and _anboard(st) == a
                out.append(st)
    return out


@pytest.fixture(scope="module")
def sanity_lib(tmp_path_factory):
    """_C_HARNESS + bg_engine.c, built from source."""
    import ctypes
    d = tmp_path_factory.mktemp("sanity_c")
    src = d / "h.c"
    src.write_text(_C_HARNESS)
    so = d / "h.so"
    r = subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(src),
         os.path.join(_REPO, "c_engine", "bg_engine.c"),
         "-I", os.path.join(_REPO, "c_inference"), "-lm"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lib = ctypes.CDLL(str(so))
    vp, ci = ctypes.c_void_p, ctypes.c_int
    for fn in (lib.h_engine_mask, lib.h_nn_mask):
        fn.argtypes, fn.restype = [vp, ci, vp], None
    lib.h_nn_reduce.argtypes, lib.h_nn_reduce.restype = [vp, vp, ci, vp], None
    return lib


@pytest.fixture(scope="module")
def sanity_x(boards):
    return np.ascontiguousarray(
        np.stack([ENC.encode(s) for s in boards + _edge_boards()]), dtype=np.float32)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
class TestCCopiesParity:
    """bg_engine.c prob5_impossible_mask and nn_eval.c prob5_sanity_zero /
    prob5_reduce (built from source) against prob_agent: identical masks,
    and prob5_reduce(sanity) bitwise prob5_sanity."""

    @staticmethod
    def _mask(fn, x):
        m = np.zeros((len(x), 4), dtype=np.uint8)
        fn(x.ctypes.data, len(x), m.ctypes.data)
        return m.astype(bool)

    def test_masks_equal(self, sanity_lib, sanity_x):
        lib, x = sanity_lib, sanity_x
        py = prob_agent._prob5_impossible_np(x)
        py = np.zeros((len(x), 4), bool) if py is None else py
        eng = self._mask(lib.h_engine_mask, x)
        nn = self._mask(lib.h_nn_mask, x)
        np.testing.assert_array_equal(eng, py)
        np.testing.assert_array_equal(nn, py)
        assert py.any(axis=0).all() and (~py).any(axis=0).all()

    def test_reduce_equals_prob5_sanity(self, sanity_lib, sanity_x):
        lib, x = sanity_lib, sanity_x
        rng = np.random.default_rng(5)
        raw = rng.random((len(x), 5)).astype(np.float32)
        raw[::7] = 1.0
        raw[1::7] = 0.0
        raw[2::7, 1:] = 1e-7
        out = np.empty_like(raw)
        lib.h_nn_reduce(x.ctypes.data, raw.ctypes.data, len(x), out.ctypes.data)
        py = prob5_sanity(torch.from_numpy(raw), torch.from_numpy(x)).numpy()
        np.testing.assert_array_equal(out, py)


def _sanitycheck_source():
    src = open(GNUBG_EVAL_C).read()
    start = src.index("extern void\nSanityCheck(")
    end = src.index("\n}\n", start) + 3
    return src[start:end]


@pytest.mark.skipif(not os.path.exists(GNUBG_EVAL_C),
                    reason="gnubg 1.08 source not present")
class TestRealGnubg:
    """Raw prod-net probabilities through gnubg's own SanityCheck vs
    prob5_sanity. Positions where gnubg's certain-result rules fire are
    skipped; in contact gnubg also zeroes outputs < 1e-4."""

    def test_prod_net(self, boards):
        from model import load_model
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "best_models",
            "cubeless_prob5_512_512_256_256.pt")
        net = load_model(path).eval()
        pick = boards[::12]
        x = torch.from_numpy(np.stack([ENC.encode(s) for s in pick]))
        with torch.no_grad():
            raw = net(x)
            if net.raw_logits:
                raw = torch.sigmoid(raw)
        ours = prob5_sanity(raw, x).numpy()
        raw = raw.numpy()
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "s.c")
            with open(c, "w") as f:
                f.write(_HARNESS_HEAD + _sanitycheck_source() + _HARNESS_MAIN)
            exe = os.path.join(d, "s")
            r = subprocess.run(["gcc", "-O0", "-o", exe, c],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            lines = []
            for s, p in zip(pick, raw):
                an = _anboard(s)
                lines.append(" ".join(map(str, an[0] + an[1])) + " "
                             + " ".join(float(v).hex() for v in p))
            out = subprocess.run([exe], input="\n".join(lines) + "\n",
                                 capture_output=True, text=True).stdout
        g = np.array([[float.fromhex(v) for v in ln.split()]
                      for ln in out.strip().split("\n")], dtype=np.float32)
        assert len(g) == len(pick)
        compared = changed = 0
        for s, p, gg, o in zip(pick, raw, g, ours):
            if gg[0] != p[0] or gg[1] == 1.0 or gg[3] == 1.0:
                continue  # a certain-result rule fired
            _, back, _, _ = _counts(_anboard(s))
            contact = back[0] + back[1] >= 24
            for k in range(1, 5):
                if contact and o[k] < 1e-4:
                    assert gg[k] == 0.0
                else:
                    assert gg[k] == o[k], (s, k, p, gg, o)
            compared += 1
            changed += bool((o != prob5_postprocess(
                torch.from_numpy(p[None]))[0].numpy()).any())
        assert compared >= 300 and changed >= 50


# ── agents ───────────────────────────────────────────────────────────


def _barely_trained_prob5(seed):
    from model import ProbNetwork
    torch.manual_seed(seed)
    net = ProbNetwork(hidden_sizes=[8], encoder_name="perspective196")
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.5)
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    x, y = torch.rand(64, net.input_size), torch.rand(64, 5)
    for _ in range(5):
        opt.zero_grad()
        ((net(x) - y) ** 2).mean().backward()
        opt.step()
    return net.eval()


def _decode(x):
    """TanBoard from one perspective196 row (independent of the helper)."""
    an = [[0] * 25, [0] * 25]
    for side, base, bar in ((1, 0, 96), (0, 98, 194)):
        for k in range(24):
            f = x[base + 4 * k: base + 4 * k + 4]
            n = int(round(f[0] + f[1] + f[2] + 2 * f[3]))
            an[side][k if side == 1 else 23 - k] = n
        an[side][24] = int(round(2 * x[bar]))
    return an


class _Spy:
    """Counts the rows the net evaluates and records every _probs output."""

    def __init__(self, agent, monkeypatch):
        self.net_rows = 0
        self.rows = []
        agent.network.register_forward_hook(self._hook)
        orig = agent._probs

        def probs(x):
            p = orig(x)
            self.rows.append((x.detach().cpu().numpy(),
                              p.detach().cpu().numpy()))
            return p
        monkeypatch.setattr(agent, "_probs", probs)

    def _hook(self, _m, inp, _out):
        self.net_rows += inp[0].shape[0]

    def check(self):
        n = 0
        for xs, ps in self.rows:
            for x, p in zip(xs, ps):
                an = _decode(x)
                np.testing.assert_array_equal(p, _gnubg_ref(an, p))
                n += 1
        assert n == self.net_rows > 0
        return n


def _game_positions(n, seed):
    """Positions with dice, biased to races and bear-offs."""
    states = _playout_states(max(4 * n, 600), seed)
    rng = random.Random(seed)
    late = [s for s in states if is_race(s) or s.off[0] or s.off[1]]
    pick = rng.sample(late, n // 2) + rng.sample(states, n // 2)
    return [(s, (rng.randint(1, 6), rng.randint(1, 6))) for s in pick]


def _drive_prob_agent(agent, plies):
    for s, dice in _game_positions(3 if plies else 200, seed=11):
        plays = get_legal_plays(s, dice)
        if plays:
            agent.choose_checker_action(s, dice, plays)
            agent.choose_next_state(s, dice)
    agent.evaluate_batch([s for s, _ in _game_positions(50, seed=12)])
    agent.evaluate(_hand_built()[0])


@pytest.mark.parametrize("plies", [0, 1])
def test_prob_agent_opt_in_obeys_rules(plies, monkeypatch):
    agent = ProbAgent(_barely_trained_prob5(1), plies=plies, board_sanity=True)
    spy = _Spy(agent, monkeypatch)
    _drive_prob_agent(agent, plies)
    assert spy.check() > 500


@pytest.mark.parametrize("plies", [0, 1])
@pytest.mark.parametrize("fast", [True, False])
def test_prob_agent_default_is_clamp_only(plies, fast, monkeypatch):
    """Plain prob5 by default: every _probs output is exactly
    prob5_postprocess of the net, on a sample where prob5_sanity would
    change many rows."""
    agent = ProbAgent(_barely_trained_prob5(1), plies=plies,
                      use_fast_engine=fast)
    assert agent.board_sanity is False
    spy = _Spy(agent, monkeypatch)
    _drive_prob_agent(agent, plies)
    net_rows = spy.net_rows
    n = changed = 0
    for xs, ps in spy.rows:
        x = torch.from_numpy(xs)
        with torch.no_grad():
            raw = agent.network(x)
        assert torch.equal(torch.from_numpy(ps), prob5_postprocess(raw))
        changed += int((prob5_sanity(raw, x).numpy() != ps).any(axis=1).sum())
        n += len(xs)
    assert n == net_rows > 500 and changed > 20


def test_prob_agent_default_decisions_match_clamp_only():
    """Default ProbAgent decisions equal those of an agent whose _probs
    is the clamp-only form; the opt-in agent's differ somewhere."""
    net = _barely_trained_prob5(4)
    plain, ref, opt = (ProbAgent(net), ProbAgent(net),
                       ProbAgent(net, board_sanity=True))

    def clamp_only(x):
        with torch.no_grad():
            return prob5_postprocess(ref.network(x))
    ref._probs = clamp_only
    differ = 0
    for s, dice in _game_positions(150, seed=21):
        a, b = plain.choose_next_state(s, dice), ref.choose_next_state(s, dice)
        c = opt.choose_next_state(s, dice)
        key = lambda st: None if st is None else (tuple(st.points),
                                                  tuple(st.bar), tuple(st.off))
        assert key(a) == key(b)
        differ += key(a) != key(c)
    assert differ > 0


def test_board_sanity_settings():
    from janowski_agent import JanowskiAgent
    net = _barely_trained_prob5(5)
    assert JanowskiAgent(net).board_sanity is True
    assert JanowskiAgent(net, board_sanity=False).board_sanity is False
    assert ProbAgent(net, board_sanity=True).board_sanity is True
    bs = prob_agent.board_sanity_from_env
    assert bs("V", False, {}) is False and bs("V", True, {}) is True
    assert bs("V", True, {"V": "0"}) is False
    assert bs("V", False, {"V": "1"}) is True
    with pytest.raises(ValueError, match="V must be 0 or 1"):
        bs("V", False, {"V": "yes"})


@pytest.mark.parametrize("oneply", [False, True])
def test_janowski_agent_obeys_rules(oneply, monkeypatch):
    from janowski_agent import JanowskiAgent
    from modes import MatchState
    agent = JanowskiAgent(_barely_trained_prob5(2), oneply=oneply,
                          bearoff_exact=False)
    spy = _Spy(agent, monkeypatch)
    ms = MatchState()
    for s, dice in _game_positions(4 if oneply else 120, seed=13):
        if get_legal_plays(s, dice):
            agent.choose_checker_action_cubeful(s, dice, ms)
        agent.offer_double(s, ms)
        agent.respond_to_double(s, ms)
    assert spy.check() > 300


def test_zeroing_moves_values():
    """The rules reach the opt-in agent's values: a race where the net's
    gammon outputs are impossible scores as plain prob5 would not."""
    agent = ProbAgent(_barely_trained_prob5(3), board_sanity=True)
    st = _hand_built()[0]
    x = agent._encode_batch([st])
    with torch.no_grad():
        raw = agent.network(x)
    plain = float((prob5_postprocess(raw) * torch.tensor(
        [2.0, 1, 1, -1, -1])).sum() - 1)
    assert agent.evaluate(st) != pytest.approx(plain)
    assert ProbAgent(agent.network).evaluate(st) == pytest.approx(plain)
