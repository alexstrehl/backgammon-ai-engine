"""
eval_matchplay.py -- Head-to-head N-point match evaluation.

Plays full matches (games chained via MatchState.start_game/after_game)
between a matchplay TDAgent and an opponent, alternating colors for
fairness, and reports the match win rate with a binomial CI.

Opponents (--model2):
  random        RandomAgent (never doubles, always takes)
  gnubg-ext[-full][:N]  real gnubg 1.08 over its external socket
                interface -- score-aware chequer play, native match
                cube handling, Kazaross-XG2 MET. Preferred referee.
  gnubg[:N]     gnubg-nn wrapper: score-BLIND chequer play and a
                Snowie MET. Kept for continuity with older numbers.
  dmp:PATH      cubeless TDAgent on a 196-input net (never doubles,
                always takes, 0-ply checker play) — isolates the value
                of cube + score awareness
  dmpcube:PATH  same DMP checker play, but always doubles when allowed
                and always takes — near-optimal doctrine at 2pt
                matches, so a strong dummy baseline
  model:PATH    another matchplay TDAgent (full cube + score play)
  p5jan:PATH[:x=V]  prob5 net + gnubg-style Janowski match cube layer
                (janowski_match.py), the promotion reads' reference

Forward path: --fast-forward BACKEND (or BG_FAST_FORWARD) runs the
matchplay nets' forward on a c_api backend, and a p5jan side's WHOLE
decision (prob5 forward, Janowski conversion, checker / cube choice) in C
(JanowskiMatchAgent.enable_fast_decisions; c_api/README.md "p5jan /
Janowski match play"). A `ff-<backend>:` prefix on one side's path
(`--model1 ff-simd:PATH`, `model:ff-off:PATH`, `p5jan:ff-torch:PATH`)
overrides it for that side only; `p5jan:ffnet-<backend>:PATH` keeps a
p5jan side on the net-only forward with its decisions in Python (the path
before the C decisions, for A/B; same decisions up to near-ties).

--paired: common-random-number dice. Matches come in pairs from 0-0 with
the seats swapped and game g of both matches drawn from one dice stream
keyed on (--seed, pair, g); the CI is computed over pairs and the report
gives the variance-reduction factor vs independent matches. Design notes
(dice alignment, opening roll, Crawford) above pair_seed_base().
--combine A.npz B.npz ...: pool --raw-out files (by pair when paired).

Also prints the model's implicit match-equity table (MWC at each
reachable away-score combination) — compare against published MET
values to sanity-check score awareness.
"""

import argparse
import math
import os
import multiprocessing as mp
import random

from backgammon_engine import BoardState, opening_roll, switch_turn
from modes import MatchState


def play_match(agent_white, agent_black, match_length, rng,
               dice_for_game=None, stats=None):
    """Play one full match. Returns 0 if WHITE wins the match, 1 if BLACK.

    ``dice_for_game`` (paired dice): a callable ``g -> rng`` giving the dice
    source for game ``g`` (0-based) of the match; every die of that game,
    opening roll included, is drawn from it and from nothing else. ``None``
    (default) draws the whole match from ``rng`` as before. ``stats`` (an
    optional dict) receives ``n_games``.
    """
    ms = MatchState(
        jacoby=False, match_length=match_length,
        my_away=match_length, opp_away=match_length,
    )
    n_games = 0
    while True:
        gms = ms.start_game()
        if dice_for_game is not None:
            rng = dice_for_game(n_games)
        n_games += 1
        state, dice = opening_roll(rng)
        is_opening = True
        while True:
            player = state.turn
            actor = agent_white if player == 0 else agent_black

            # Cube phase
            if not is_opening and gms.can_offer(player):
                offer = actor.offer_double(state, gms)
                if offer.should_double:
                    responder = agent_black if player == 0 else agent_white
                    same_agent = responder is actor
                    takes = responder.respond_to_double(
                        state, gms, hint=offer if same_agent else None,
                    )
                    if takes:
                        gms = gms.after_take(player)
                    else:
                        ms = gms.after_game(1, winner_is_me=(player == 0))
                        break  # game over (drop)

            # Checker phase
            result = actor.choose_checker_action_cubeful(state, dice, gms)
            if result is None:
                state = switch_turn(state)
            else:
                next_state = result[0]
                if next_state.is_game_over():
                    gr = next_state.game_result()
                    ms = gms.after_game(gr, winner_is_me=(player == 0))
                    break  # game over
                state = next_state
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            is_opening = False

        if ms.match_over():
            if stats is not None:
                stats["n_games"] = n_games
            # ms away scores are WHITE's view.
            return 0 if ms.my_away <= 0 else 1


# ── Paired dice ──────────────────────────────────────────────────────
#
# Matches come in pairs (2p, 2p+1). Both matches of pair p start at 0-0;
# model1 sits WHITE for the whole of the first match and BLACK for the
# whole of the second. Game g of either match draws every die -- opening
# roll included -- from its own stream seeded SHA-256(pair_seed_base ||
# p || g). Because a game consumes exactly two opening dice per opening
# attempt and then exactly one roll per turn (a roll is drawn after every
# turn, including no-move turns; a drop ends the game before the next
# draw), draw k of the stream IS the dice of turn k: the stream is a
# per-(pair, game, turn) derivation, and the two matches stay aligned
# however their games diverge -- a longer game in one match never shifts
# the dice of any other game.
#
# Opening roll: the white die / black die go to the same COLOURS in both
# matches, so the net that won the opening roll in one match loses it
# (with the same dice) in the other, and roll k of game g goes to the
# other net in the mirror. Crawford / post-Crawford: nothing special --
# MatchState applies the rule in each match on its own; the dice key is
# the game INDEX, not the score, so once the scores of the two matches
# diverge the same game index may be played at different scores (or be
# the Crawford game in one match only), and a match that ends early
# simply leaves the rest of the pair's streams unused.
#
# Each match on its own is a fair match with iid dice and model1 WHITE in
# exactly half of them (as in the unpaired alternation), so the paired
# mean is unbiased; only the within-pair correlation changes. The seed
# base is a pure function of --seed and pairs are indexed globally, so a
# paired run is reproducible and independent of --workers.


def pair_seed_base(seed):
    """32-hex-char seed base of a paired run with ``--seed seed``."""
    import hashlib
    return hashlib.sha256(
        b"eval_matchplay/paired/" + str(int(seed)).encode()
    ).hexdigest()[:32]


def game_seed(seed_base, pair_id, game_index):
    """Dice seed of game ``game_index`` of either match of pair
    ``pair_id`` (global, 0-based) under ``seed_base``."""
    import hashlib
    h = hashlib.sha256(bytes.fromhex(seed_base)
                       + int(pair_id).to_bytes(8, "little")
                       + int(game_index).to_bytes(4, "little")).digest()
    return int.from_bytes(h, "little")


def play_pair(a1, a2, match_length, seed_base, pair_id, stats=None):
    """Play pair ``pair_id``: model1 WHITE, then model1 BLACK, same dice.
    Returns ``(won_a, won_b)`` -- model1's match outcome (1/0) in each.
    ``stats`` (optional list) gets one dict per match (``n_games``)."""
    def dice_for_game(g):
        return random.Random(game_seed(seed_base, pair_id, g))

    out = []
    for a1_white in (True, False):
        st = {}
        if a1_white:
            w = play_match(a1, a2, match_length, None, dice_for_game, st) == 0
        else:
            w = play_match(a2, a1, match_length, None, dice_for_game, st) == 1
        out.append(int(w))
        if stats is not None:
            stats.append(st)
    return out[0], out[1]


_FF_BACKENDS = ("off", "torch", "simd", "cblas", "ref")


def _split_ff(path):
    """`ff-<backend>:PATH` -> (backend, PATH); `ffnet-<backend>:PATH` ->
    ("net-<backend>", PATH), a p5jan side's net-only forward; plain PATH
    -> (None, PATH). A per-side override of --fast-forward /
    BG_FAST_FORWARD, e.g. to play a net against itself on two paths."""
    for prefix, tag in (("ffnet-", "net-"), ("ff-", "")):
        if path.startswith(prefix):
            head, _, rest = path.partition(":")
            b = head[len(prefix):]
            if b not in _FF_BACKENDS or (tag and b == "off"):
                raise ValueError(f"unknown fast-forward backend in {path!r}")
            return tag + b, rest
    return None, path


def _apply_ff(agent, backend):
    """Set (or clear, "off") `agent`'s c_api path. A TDAgent: its net
    forward (enable_fast_forward). A p5jan JanowskiMatchAgent: the whole
    decision in C (enable_fast_decisions), or with "net-<backend>" the
    net-only forward with the decisions in Python (the previous path)."""
    if backend is None:
        return agent
    net_only = backend.startswith("net-")
    b = backend[len("net-"):] if net_only else backend
    b = None if b == "off" else b
    if hasattr(agent, "enable_fast_decisions"):  # JanowskiMatchAgent
        from c_api.fast_forward import FastForward
        agent._fast_ff = None if b is None else FastForward(agent.network, b)
        agent.enable_fast_decisions(None if net_only else b)
    elif net_only:
        raise ValueError(f"ffnet-{b}: is for p5jan sides; a TD side's ff-{b}: "
                         f"already moves only its net forward")
    else:
        agent.enable_fast_forward(b)
    return agent


def _decision_path(agent):
    """Where `agent`'s decisions run (the --raw-out provenance)."""
    ff = getattr(agent, "_fast_ff", None)
    if hasattr(agent, "enable_fast_decisions"):
        if agent._fast_dec is not None:
            return f"p5jan c-{agent._fast_dec.backend}"
        return f"p5jan net-{ff.backend}" if ff is not None else "p5jan torch"
    if getattr(agent, "is_matchplay", False):
        return f"td ff-{ff.backend}" if ff is not None else "td torch"
    return type(agent).__name__


def _build_p5jan(spec):
    """`p5jan:[ff-B:|ffnet-B:]PATH[:x=VAL]` -- prob5 net + gnubg-style
    match-play Janowski cube layer (janowski_match.py). Always MET-exact
    (that is the point of the construction); `x=` overrides the gnubg
    position-class cube-life policy with a constant. With a backend (the
    prefix, else --fast-forward / BG_FAST_FORWARD) the decisions run in C;
    `ffnet-B:` keeps them in Python on B's net-only forward."""
    from c_api.fast_forward import backend_from_env
    from janowski_match import JanowskiMatchAgent
    from model import ProbNetwork

    ff, rest = _split_ff(spec[len("p5jan:"):])
    x_policy = None
    if ":x=" in rest:
        rest, _, xs = rest.rpartition(":x=")
        x_policy = float(xs)
    agent = JanowskiMatchAgent(ProbNetwork.load(rest), x_policy=x_policy)
    return _apply_ff(agent, ff if ff is not None else backend_from_env())


def _build_agents(model1_path, model2_spec, met_path=None, met2_path=None,
                  referee_met_path=None):
    from model import TDNetwork
    from td_agent import TDAgent
    from agents import RandomAgent

    def _load_met(path):
        if not path:
            return None
        from met import MatchEquityTable
        return MatchEquityTable.from_xml(path)

    # Each side plays as it was TRAINED. In a cross-arm head-to-head one
    # model may be MET-grounded and the other not; forcing a single
    # setting on both would measure two agents neither of which was
    # trained that way.
    met = _load_met(met_path)
    met2 = _load_met(met2_path if met2_path is not None else met_path)

    if model1_path.startswith("p5jan:"):
        a1 = _build_p5jan(model1_path)
    elif model1_path.startswith("gnubg-ext"):
        # gnubg vs gnubg (e.g. --model1 gnubg-ext:1 --model2 gnubg-ext):
        # sizes the ply gap in match wins with the same referee stack.
        from gnubg_external import GnubgExternalAgent
        head, _, ply_s = model1_path.partition(":")
        a1 = GnubgExternalAgent(
            plies=int(ply_s) if ply_s else 0,
            full=head.endswith("-full"),
            met_path=referee_met_path,
        )
    else:
        # "1ply:<path>" -- same weights, successors ranked by their own
        # 1-ply value instead of the raw network read (TDAgent.oneply_moves).
        ff1, model1_path = _split_ff(model1_path)
        om1 = model1_path.startswith("1ply:")
        if om1:
            model1_path = model1_path[len("1ply:"):]
        net1 = TDNetwork.load(model1_path)
        a1 = _apply_ff(TDAgent(net1, met=met, oneply_moves=om1), ff1)
        assert a1.is_matchplay, \
            f"--model1 must be a matchplay net; got encoder={net1.encoder_name}"

    if model2_spec == "random":
        a2 = RandomAgent()
    elif model2_spec.startswith("gnubg-ext"):
        # "gnubg-ext[-full][:PLIES]" -- real gnubg 1.08 over its external
        # socket interface. Preferred over the gnubg-nn wrapper for match
        # play: score-aware chequer play, native match cube handling, and
        # the Kazaross-XG2 MET rather than gnubg-nn's Snowie.
        #
        # The referee is pinned to ONE table regardless of how the model
        # under test was trained, so every arm of an ablation faces an
        # identical opponent.
        from gnubg_external import GnubgExternalAgent
        head, _, ply_s = model2_spec.partition(":")
        a2 = GnubgExternalAgent(
            plies=int(ply_s) if ply_s else 0,
            full=head.endswith("-full"),
            met_path=referee_met_path,
        )
    elif model2_spec.startswith("gnubg"):
        # "gnubg" or "gnubg:N" (checker plies; cube plies = max(N, 1)).
        from agents import GnubgNNMatchplayAgent
        plies = int(model2_spec.split(":", 1)[1]) if ":" in model2_spec else 0
        a2 = GnubgNNMatchplayAgent(plies=plies)
    elif model2_spec.startswith("dmp:"):
        net2 = TDNetwork.load(model2_spec[len("dmp:"):])
        a2 = TDAgent(net2)
        assert not a2.is_matchplay and not a2.is_cubeful
    elif model2_spec.startswith("dmpcube:"):
        from agents import AlwaysCubeAgent
        net2 = TDNetwork.load(model2_spec[len("dmpcube:"):])
        base = TDAgent(net2)
        assert not base.is_matchplay and not base.is_cubeful
        a2 = AlwaysCubeAgent(base)
    elif model2_spec.startswith("p5jan:"):
        a2 = _build_p5jan(model2_spec)
    elif model2_spec.startswith("model:"):
        ff2, m2 = _split_ff(model2_spec[len("model:"):])
        om2 = m2.startswith("1ply:")
        if om2:
            m2 = m2[len("1ply:"):]
        net2 = TDNetwork.load(m2)
        a2 = _apply_ff(TDAgent(net2, met=met2, oneply_moves=om2), ff2)
        assert a2.is_matchplay
    else:
        raise ValueError(f"Unknown --model2 spec: {model2_spec!r}")
    return a1, a2


_WORKER = None


def _init_worker(model1_path, model2_spec, match_length, met_path=None,
                 met2_path=None, referee_met_path=None):
    global _WORKER
    from proc_utils import die_with_parent
    die_with_parent()       # a killed parent never strands this worker
    import torch
    torch.set_num_threads(1)
    a1, a2 = _build_agents(model1_path, model2_spec, met_path, met2_path,
                           referee_met_path)
    _WORKER = (a1, a2, match_length)


def _run_chunk(args):
    """Returns per-match outcomes, not just the total: pooled CIs have
    to stay recomputable from raws after the fact."""
    import numpy as np
    n_matches, seed = args
    a1, a2, match_length = _WORKER
    rng = random.Random(seed)
    won = np.zeros(n_matches, dtype=np.int8)
    white = np.zeros(n_matches, dtype=np.int8)
    for i in range(n_matches):
        # Alternate colors: even i -> model1 is WHITE.
        if i % 2 == 0:
            won[i] = play_match(a1, a2, match_length, rng) == 0
            white[i] = 1
        else:
            won[i] = play_match(a2, a1, match_length, rng) == 1
    return won, white, seed


def _run_pairs(args):
    """Paired-dice chunk: pairs [pair_start, pair_start + n_pairs)."""
    import numpy as np
    pair_start, n_pairs, seed_base = args
    a1, a2, match_length = _WORKER
    won = np.zeros(2 * n_pairs, dtype=np.int8)
    white = np.tile(np.array([1, 0], dtype=np.int8), n_pairs)
    pair_id = np.repeat(np.arange(pair_start, pair_start + n_pairs,
                                  dtype=np.int64), 2)
    n_games = np.zeros(2 * n_pairs, dtype=np.int16)
    for p in range(n_pairs):
        st = []
        won[2 * p], won[2 * p + 1] = play_pair(
            a1, a2, match_length, seed_base, pair_start + p, st)
        n_games[2 * p], n_games[2 * p + 1] = (st[0]["n_games"],
                                              st[1]["n_games"])
    return won, white, pair_id, n_games


def paired_stats(won, pair_id):
    """Pair-level statistics of a paired run (or a pool of paired runs
    whose pair ids are distinct -- pass ``(seed_base, pair)`` tuples
    flattened into unique ints, or any array of distinct labels).

    ``won``: model1's per-match outcome; the two matches of each pair
    must be adjacent (``pair_id[2p] == pair_id[2p+1]``). Returns a dict:
    ``wr`` (mean), ``se_paired`` = sd(pair means)/sqrt(n_pairs),
    ``se_unpaired`` = sqrt(wr(1-wr)/n) (what the same number of
    independent matches would give), ``factor`` = se_unpaired^2 /
    se_paired^2 (1.0 = no gain; how many times fewer matches a paired
    read needs), and the pair-outcome counts ``n_both`` / ``n_split`` /
    ``n_none`` (model1 won 2 / 1 / 0 of the pair).
    """
    import numpy as np
    won = np.asarray(won, dtype=np.float64).ravel()
    pid = np.asarray(pair_id).ravel()
    if won.size % 2 or won.size != pid.size \
            or not np.array_equal(pid[0::2], pid[1::2]):
        raise ValueError("paired runs store the two matches of a pair "
                         "adjacently with equal pair ids")
    pm = (won[0::2] + won[1::2]) / 2.0
    n_pairs, n = pm.size, won.size
    wr = float(won.mean())
    var_pm = float(pm.var(ddof=1)) if n_pairs > 1 else float("nan")
    se_p = math.sqrt(var_pm / n_pairs) if n_pairs > 1 else float("nan")
    se_u = math.sqrt(max(wr * (1 - wr), 1e-12) / n)
    factor = (se_u ** 2 / se_p ** 2) if se_p and se_p > 0 else float("inf")
    return {
        "wr": wr, "n": n, "n_pairs": n_pairs,
        "se_paired": se_p, "se_unpaired": se_u, "factor": factor,
        "n_both": int((pm == 1).sum()), "n_split": int((pm == 0.5).sum()),
        "n_none": int((pm == 0).sum()),
    }


def factor_ci(n_both, n_split, n_none, n_boot=20000, seed=1):
    """95% bootstrap CI (over pairs; multinomial on the three pair
    outcomes) for the variance-reduction factor."""
    import numpy as np
    n = n_both + n_split + n_none
    r = np.random.default_rng(seed)
    c = r.multinomial(n, np.array([n_both, n_split, n_none]) / n, n_boot)
    b, h, z = c[:, 0], c[:, 1], c[:, 2]
    wr = (b + 0.5 * h) / n
    e2 = (b + 0.25 * h) / n
    var_pm = (e2 - wr ** 2) * n / (n - 1)
    var_match = wr * (1 - wr)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (var_match / 2) / var_pm
    return tuple(float(x) for x in np.percentile(f, [2.5, 97.5]))


def print_paired(ps, label):
    lo, hi = ps["wr"] - 1.96 * ps["se_paired"], ps["wr"] + 1.96 * ps["se_paired"]
    z = (ps["wr"] - 0.5) / ps["se_paired"] if ps["se_paired"] > 0 else 0.0
    ulo = ps["wr"] - 1.96 * ps["se_unpaired"]
    uhi = ps["wr"] + 1.96 * ps["se_unpaired"]
    print(f"\n{label}: paired dice, {ps['n_pairs']} pairs / {ps['n']} matches")
    print(f"  pairs: model1 won both {ps['n_both']}, split {ps['n_split']}, "
          f"neither {ps['n_none']}")
    print(f"  paired   {ps['wr']:.4%}  95% CI [{lo:.4%}, {hi:.4%}]  "
          f"(+/-{1.96 * ps['se_paired'] * 100:.4f}pp, z={z:+.2f}; "
          f"pair-level SE)")
    print(f"  unpaired {ps['wr']:.4%}  95% CI [{ulo:.4%}, {uhi:.4%}]  "
          f"(+/-{1.96 * ps['se_unpaired'] * 100:.4f}pp; binomial SE = what "
          f"as many INDEPENDENT matches would give; not valid here)")
    if ps["n_split"] == ps["n_pairs"]:
        print("  variance-reduction factor: inf (every pair split -- "
              "mirror self-play?)")
    else:
        flo, fhi = factor_ci(ps["n_both"], ps["n_split"], ps["n_none"])
        print(f"  variance-reduction factor: {ps['factor']:.2f}x  "
              f"[{flo:.2f}, {fhi:.2f}]  (independent matches needed per "
              f"paired match for the same CI)")


def combine(paths):
    """Pool --raw-out files: paired runs by pair (never with unpaired
    runs, never two runs with one seed base), unpaired runs binomially."""
    import numpy as np
    ds = [np.load(f, allow_pickle=True) for f in paths]
    kinds = {bool("pair_id" in d.files) for d in ds}
    if len(kinds) > 1:
        raise SystemExit("refusing to pool paired and unpaired runs")
    keys = {(str(d["model1"]), str(d["model2"]), int(d["match_length"]),
             str(d["met"]), str(d["met2"])) for d in ds}
    if len(keys) > 1:
        raise SystemExit(f"refusing to pool different matchups: {keys}")
    label = (f"{int(ds[0]['match_length'])}pt {ds[0]['model1']} vs "
             f"{ds[0]['model2']} ({len(ds)} files)")
    if kinds == {True}:
        bases = [str(d["pair_seed_base"]) for d in ds]
        if len(set(bases)) != len(bases):
            raise SystemExit("refusing to pool: two runs share a pair seed "
                             "base (same --seed) -- duplicate dice")
        won = np.concatenate([d["won"] for d in ds])
        # distinct labels across files: file index * 2^40 + pair id
        pid = np.concatenate([np.asarray(d["pair_id"], dtype=np.int64)
                              + (i << 40) for i, d in enumerate(ds)])
        print_paired(paired_stats(won, pid), label)
    else:
        seeds = [int(s) for d in ds for s in np.asarray(d["seeds"]).ravel()]
        if len(set(seeds)) != len(seeds):
            print("WARNING: duplicate worker seeds across files")
        won = np.concatenate([d["won"] for d in ds]).astype(np.float64)
        n = won.size
        wr = float(won.mean())
        se = math.sqrt(max(wr * (1 - wr), 1e-12) / n)
        print(f"\n{label}: {int(won.sum())}/{n} = {wr:.4%}  95% CI "
              f"[{wr - 1.96 * se:.4%}, {wr + 1.96 * se:.4%}]")


def mwc_report(agent, match_length):
    """Print the model's implicit match-equity table: P(win match) at
    the start of a game for every reachable away-score combination."""
    print(f"\nImplicit MET (new-game MWC, {match_length}pt match):")
    print("  (my_away, opp_away, crawford_done) -> P(I win match)")
    for my in range(match_length, 0, -1):
        for opp in range(match_length, 0, -1):
            done = (my == 1 or opp == 1)
            ms = MatchState(
                jacoby=False, match_length=match_length,
                my_away=my, opp_away=opp,
                crawford_done=(done and not (my == match_length
                                             or opp == match_length)),
            )
            # Always the NETWORK's own value, never the pinned table —
            # the point of this diagnostic is what the net believes. With
            # a MET pinned, `err` is how far it has drifted from truth.
            v = agent._eval_new_game_net(ms, 0)
            tag = ""
            if not ms.crawford_done and (my == 1 or opp == 1):
                tag = "  (Crawford game)"
            if agent.met is not None:
                truth = agent.met.mwc(
                    ms.my_away, ms.opp_away, ms.crawford_done, ms.is_crawford,
                )
                tag = f"  [MET {truth:.3f}, err {v - truth:+.3f}]{tag}"
            print(f"  ({my}-away, {opp}-away) -> {v:.3f}{tag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model1")
    p.add_argument("--model2", default="random",
                   help="random | dmp:PATH | dmpcube:PATH | model:PATH | "
                        "p5jan:[ff-B:|ffnet-B:]PATH[:x=V] (ff-B: p5jan's "
                        "decisions in C on backend B; ffnet-B: its net "
                        "forward only, decisions in Python)")
    p.add_argument("--matches", type=int, default=500)
    p.add_argument("--match-length", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--met", default=None, metavar="TABLE",
                   help="Play with a real match equity table ('auto' = "
                        "the bundled met_tables/Kazaross-XG2.xml, falling "
                        "back to gnubg's system table; a table name from "
                        "/usr/share/gnubg/met; or a path). MUST match how the "
                        "model was trained: a --met-trained model "
                        "evaluated without it reverts v_dp, the take "
                        "point and terminal gammon pricing to the "
                        "network's own drifting estimates.")
    p.add_argument("--gnubg-met", default="auto", metavar="TABLE",
                   help="MET the gnubg-ext REFEREE plays with. Pinned to "
                        "one table for every arm of an ablation so the "
                        "opponent is identical regardless of how the model "
                        "under test was trained. 'none' leaves gnubg on its "
                        "own default (already Kazaross-XG2).")
    p.add_argument("--met2", default=None, metavar="TABLE",
                   help="MET for --model2 when it is a matchplay model "
                        "trained differently from --model1. Use the "
                        "literal 'none' for an ungrounded opponent. "
                        "Defaults to --met.")
    p.add_argument("--raw-out", default=None, metavar="PATH",
                   help="Save per-match outcomes + provenance to an .npz "
                        "so the CI can be recomputed and runs pooled.")
    p.add_argument("--met-only", action="store_true",
                   help="Only print the implicit MET, skip head-to-head")
    p.add_argument("--paired", action="store_true",
                   help="Paired (common-random-number) dice: matches come "
                        "in pairs from 0-0 with the same dice, model1 WHITE "
                        "in one and BLACK in the other; game g of both "
                        "draws every die (opening roll included) from a "
                        "stream keyed on (--seed, pair, g). Each match is "
                        "still fair; the CI is computed over pairs. "
                        "--matches must be even. Results do not depend on "
                        "--workers. See the comment above pair_seed_base().")
    p.add_argument("--fast-forward", default=None, metavar="BACKEND",
                   choices=["off", "torch", "simd", "cblas", "ref"],
                   help="Run the matchplay TDAgents' net forward on a c_api "
                        "backend (c_api/fast_forward.py; build with "
                        "c_api/build.sh): torch = bitwise the default path, "
                        "simd = fastest (~1e-7 drift). A p5jan side runs its "
                        "whole decision in C on the backend (prob5 forward, "
                        "Janowski conversion, checker and cube choice; "
                        "simd: ~4e-7 from the torch path, decisions equal "
                        "up to near-ties); --model2 p5jan:ffnet-B:PATH keeps "
                        "it on the net-only forward with Python decisions "
                        "(the previous path, for A/B). Sets BG_FAST_FORWARD "
                        "for this process and its workers. Default: off, "
                        "unless BG_FAST_FORWARD is already set.")
    p.add_argument("--combine", nargs="+", metavar="NPZ",
                   help="Pool --raw-out files and print the pooled result "
                        "(paired runs by pair, never mixed with unpaired "
                        "runs), then exit.")
    args = p.parse_args()

    if args.combine:
        combine(args.combine)
        return
    if not args.model1:
        p.error("--model1 is required")
    if args.paired and args.matches % 2:
        p.error(f"--paired plays whole pairs: --matches must be even "
                f"(got {args.matches})")

    if args.fast_forward is not None:
        os.environ["BG_FAST_FORWARD"] = args.fast_forward
    from c_api.fast_forward import backend_from_env
    fast_forward = backend_from_env() or "off"
    if fast_forward != "off":
        print(f"fast forward: {fast_forward} (c_api backend for matchplay "
              f"TDAgents' forward and p5jan sides' decisions)")

    from met import resolve_met_path
    met_path = resolve_met_path(args.met)
    if args.met2 is None:
        met2_path = met_path
    elif args.met2 == "none":
        met2_path = None
    else:
        met2_path = resolve_met_path(args.met2)
    referee_met_path = (
        None if args.gnubg_met == "none" else resolve_met_path(args.gnubg_met)
    )
    a1, a2 = _build_agents(args.model1, args.model2, met_path, met2_path,
                           referee_met_path)
    decision_paths = [_decision_path(a1), _decision_path(a2)]
    print(f"decisions: model1 {decision_paths[0]}, model2 {decision_paths[1]}")
    if hasattr(a1, "_eval_new_game_net"):
        mwc_report(a1, args.match_length)
    else:
        print("\n(model1 has no learned implicit MET -- p5jan is "
              "MET-exact by construction; skipping the report)")
    if args.met_only:
        return

    if args.paired:
        seed_base = pair_seed_base(args.seed)
        n_pairs = args.matches // 2
        w = max(1, min(args.workers, n_pairs))
        per = [n_pairs // w] * w
        for i in range(n_pairs % w):
            per[i] += 1
        chunks, acc = [], 0
        for n_p in per:
            chunks.append((acc, n_p, seed_base))
            acc += n_p
        if w == 1:
            _init_worker(args.model1, args.model2, args.match_length,
                         met_path, met2_path, referee_met_path)
            chunk_results = [_run_pairs(chunks[0])]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                w, initializer=_init_worker,
                initargs=(args.model1, args.model2, args.match_length,
                          met_path, met2_path, referee_met_path),
            ) as pool:
                chunk_results = pool.map(_run_pairs, chunks)
    elif args.workers <= 1:
        _init_worker(args.model1, args.model2, args.match_length,
                     met_path, met2_path, referee_met_path)
        chunk_results = [_run_chunk((args.matches, args.seed))]
    else:
        per = [args.matches // args.workers] * args.workers
        for i in range(args.matches % args.workers):
            per[i] += 1
        chunks = [(n, args.seed + 1000 * i) for i, n in enumerate(per) if n]
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            len(chunks), initializer=_init_worker,
            initargs=(args.model1, args.model2, args.match_length,
                      met_path, met2_path, referee_met_path),
        ) as pool:
            chunk_results = pool.map(_run_chunk, chunks)

    import numpy as np
    won = np.concatenate([c[0] for c in chunk_results])
    white = np.concatenate([c[1] for c in chunk_results])
    wins = int(won.sum())

    if args.raw_out:
        import subprocess
        # Provenance. subprocess.run does NOT raise on a non-zero exit, so the
        # old `try/except -> "unknown"` never fired for the common failure:
        # running from a directory that is not a git checkout, where git exits
        # 128 and stdout is empty. That wrote commit="" and looked like a
        # populated field with nothing in it. Check the return code, and record
        # a dirty tree -- a bare hash from a modified worktree is a lie.
        _here = os.path.dirname(os.path.abspath(__file__))

        def _git(*a):
            try:
                p = subprocess.run(["git", *a], capture_output=True,
                                   text=True, cwd=_here)
            except Exception:
                return None
            return p.stdout.strip() if p.returncode == 0 else None

        commit = _git("rev-parse", "HEAD")
        if not commit:
            commit = "not-a-git-checkout"
        elif _git("status", "--porcelain"):
            commit += "-dirty"
        extra = {}
        if args.paired:
            extra = dict(
                paired=True, pair_seed_base=seed_base,
                pair_id=np.concatenate([c[2] for c in chunk_results]),
                n_games=np.concatenate([c[3] for c in chunk_results]),
                seeds=np.array([args.seed]),
            )
        else:
            extra = dict(seeds=np.array([c[2] for c in chunk_results]))
        np.savez(
            args.raw_out, won=won, model1_is_white=white,
            model1=args.model1, model2=args.model2,
            match_length=args.match_length, seed=args.seed,
            met=met_path or "", met2=met2_path or "",
            referee_met=referee_met_path or "", commit=commit,
            fast_forward=fast_forward, decision_paths=np.array(decision_paths),
            **extra,
        )
        print(f"Raw per-match results -> {args.raw_out}")

    if args.paired:
        pid = np.concatenate([c[2] for c in chunk_results])
        print_paired(paired_stats(won, pid),
                     f"{args.match_length}pt matches vs {args.model2}")
        return

    n = args.matches
    wr = wins / n
    se = math.sqrt(max(wr * (1 - wr), 1e-9) / n)
    lo, hi = wr - 1.96 * se, wr + 1.96 * se
    print(f"\n{args.match_length}pt matches vs {args.model2}: "
          f"{wins}/{n} = {wr:.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")


if __name__ == "__main__":
    main()
