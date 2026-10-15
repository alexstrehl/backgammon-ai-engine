"""
audit_collector.py -- Mechanical audit of the matchplay collector's
(encoding, target) pairs.

Wraps a TDAgent in a spy that records every call the collector makes
(state, match-state, phase, values), then independently re-verifies
each emitted sample:

  cube samples:    enc == encode(state, gms, ca=1); target == the
                   optimal-action backup from the recorded offer cache
                   (or the drop-branch grounding).
  checker samples: enc == encode(state, gms, ca=0); non-terminal
                   target == 1 - V(chosen successor, opp flags)
                   recomputed fresh (catches selection/bootstrap
                   encoding desync).
  boundaries:      target == 1.0 / _eval_new_game(after_game(...)),
                   with the game_result and winner_is_me mapping
                   recomputed from the recorded successor; upweight
                   duplicates checked.
  perspective:     away-score columns decoded from every encoding and
                   checked against the recorded match state flipped to
                   the on-roll player's view.

Usage:
    python audit_collector.py --model models/foo.pt --episodes 30 \
        [--from-errors /tmp/cube_errors.json --index 0]
"""

import argparse
import random

import numpy as np
import torch

from backgammon_engine import BoardState
from modes import MatchplayMode, MatchState, cube_perspective
from model import TDNetwork
from td_agent import TDAgent


class SpyAgent(TDAgent):
    """Records every collector-facing call, then delegates."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log = []
        self._eval_depth = 0

    def _encode_matchplay(self, state, match_state, is_cube_action=False):
        enc = super()._encode_matchplay(state, match_state, is_cube_action)
        # Only collector-direct encodes are samples; evaluate_matchplay
        # (and through it offer_double/_eval_new_game) also encodes.
        if self._eval_depth == 0:
            self.log.append(("enc", state, match_state, is_cube_action, enc))
        return enc

    def evaluate_matchplay(self, state, match_state, is_cube_action=False):
        self._eval_depth += 1
        try:
            return super().evaluate_matchplay(
                state, match_state, is_cube_action)
        finally:
            self._eval_depth -= 1

    def offer_double_matchplay(self, state, match_state):
        offer = super().offer_double_matchplay(state, match_state)
        self.log.append(("offer", state, match_state, dict(offer._cache or {})))
        return offer

    def respond_to_double_matchplay(self, state, match_state, hint=None):
        takes = super().respond_to_double_matchplay(state, match_state, hint)
        self.log.append(("respond", state, match_state, takes))
        return takes

    def choose_checker_action_matchplay(self, state, dice, match_state,
                                        rng=None):
        # `rng` threads the episode's Random into Boltzmann selection
        # (trainer.py:775 passes it unconditionally); without it here the
        # spy raised TypeError on every call and the audit could not run.
        r = super().choose_checker_action_matchplay(state, dice, match_state,
                                                    rng=rng)
        self.log.append(("choose", state, dice, match_state,
                         None if r is None else r[0],
                         None if r is None else r[1]))
        return r



def _oneply_target_reference(agent, state, gms):
    """Independent slow-path recomputation of the matchplay 1-ply
    checker target: pure-python movegen, one fresh evaluate_matchplay
    per successor, terminals via after_game -> _eval_new_game, forced
    passes via switch_turn. Mirrors the SEMANTICS of
    TDAgent.value_oneply_checker_matchplay while sharing none of its
    batching/fast-path code, so a fast-path bug cannot self-verify."""
    from backgammon_engine import get_legal_plays, switch_turn

    mover = state.turn
    opponent = 1 - mover
    total = 0.0
    forced = None
    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
            plays = get_legal_plays(state, (d1, d2))
            if not plays:
                if forced is None:
                    ov = switch_turn(state)
                    forced = 1.0 - agent.evaluate_matchplay(
                        ov, gms, is_cube_action=gms.can_offer(ov.turn))
                total += prob * forced
                continue
            best = None
            for _, s in plays:
                if s.is_game_over():
                    gr = s.game_result()
                    nm = gms.after_game(gr, winner_is_me=(mover == 0))
                    val = (1.0 if nm.match_over()
                           else agent._eval_new_game(nm, mover,
                                                     scope="terminal"))
                else:
                    val = 1.0 - agent.evaluate_matchplay(
                        s, gms, is_cube_action=gms.can_offer(opponent))
                if best is None or val > best:
                    best = val
            total += prob * best
    return min(1.0, max(0.0, total))


def audit_episode(agent, enc_arr, targets, fresh_agent, oneply=False,
                  oracle=None):
    """Verify one episode's samples against the spy log. Returns a
    list of discrepancy strings (empty = clean)."""
    problems = []
    n_base = agent.encoder._base.num_features
    log = agent.log
    enc_events = [e for e in log if e[0] == "enc"]
    if len(enc_events) > len(targets):
        problems.append(
            f"enc events {len(enc_events)} > targets {len(targets)}")

    # Walk emitted rows; consume log events. Upweighted boundary rows
    # are byte-identical duplicates of their predecessor.
    li = 0  # index into enc_events
    i = 0   # index into emitted rows
    choose_events = [e for e in log if e[0] == "choose"]
    offer_events = [e for e in log if e[0] == "offer"]
    ci = oi = 0

    while i < len(targets):
        # Duplicate (upweight) rows: identical to previous row.
        if i > 0 and np.array_equal(enc_arr[i], enc_arr[i - 1]) \
                and targets[i] == targets[i - 1] and li >= len(enc_events):
            i += 1
            continue
        if i > 0 and np.array_equal(enc_arr[i], enc_arr[i - 1]) \
                and targets[i] == targets[i - 1] \
                and li < len(enc_events) \
                and not np.array_equal(enc_arr[i], enc_events[li][4]):
            i += 1
            continue
        if li >= len(enc_events):
            problems.append(f"row {i}: no matching enc event")
            break
        _, state, gms, ca, enc_logged = enc_events[li]

        # 1. emitted row matches the logged encoding
        if not np.array_equal(enc_arr[i], enc_logged):
            problems.append(f"row {i}: emitted enc != logged enc")

        # 2. re-encode independently (fresh agent, same weights)
        enc_ref = fresh_agent._encode_matchplay(state, gms, ca)
        if not np.array_equal(enc_logged, enc_ref):
            problems.append(f"row {i}: enc != independent re-encode")

        # 3. away-score perspective decode
        extras = enc_arr[i][n_base:]
        layout = agent.encoder.match_extras(
            cube_perspective(gms.cube_owner, state.turn),
            gms.cube_value,
            gms.my_away if state.turn == 0 else gms.opp_away,
            gms.opp_away if state.turn == 0 else gms.my_away,
            gms.is_crawford, ca,
        )
        if not np.array_equal(extras, layout):
            problems.append(f"row {i}: extras layout mismatch "
                            f"(persp/away flip?)")

        t = targets[i]
        if oracle is not None:
            # Uniform oracle-row invariant: EVERY row's target equals
            # the oracle value of its encoded (state, match_state,
            # is_cube_action) -- pre-move rows, cube rows, the
            # double/take-branch rows, and epsilon-set successor rows
            # alike.
            from trainer import _oracle_target
            want = _oracle_target(oracle, state, gms, ca)
            if abs(t - want) > 1e-6:
                problems.append(
                    f"row {i}: oracle target {t:.6f} != recomputed "
                    f"{want:.6f} (ca={ca})")
        elif ca:
            # cube sample: optimal-action backup or drop grounding
            if oi < len(offer_events):
                c = offer_events[oi][3]
                oi += 1
                v_opt = max(c["v_nd"], min(c["v_dt"], c["v_dp"]))
                grounded_drop = abs(t - v_opt) > 1e-6
                if grounded_drop:
                    # Must be an on-policy drop: target is exactly the
                    # boundary grounding after scoring cube_value.
                    _, ostate, ogms, _c = offer_events[oi - 1]
                    after = ogms.after_game(
                        1, winner_is_me=(ostate.turn == 0))
                    want = (1.0 if after.match_over()
                            else fresh_agent._eval_new_game(
                                after, ostate.turn))
                    if abs(t - want) > 1e-5:
                        problems.append(
                            f"row {i}: drop target {t:.4f} != optimal "
                            f"backup {v_opt:.4f} AND != drop grounding "
                            f"{want:.4f}")
            else:
                problems.append(f"row {i}: cube sample without offer")
        elif not ca:
            # checker sample
            if ci < len(choose_events):
                _, cs, dice, cgms, succ, bootstrap = choose_events[ci]
                ci += 1
                if succ is None:
                    pass  # forced pass; target checked by range only
                elif succ.is_game_over():
                    gr = succ.game_result()
                    new_ms = cgms.after_game(
                        gr, winner_is_me=(cs.turn == 0))
                    if new_ms.match_over():
                        want = 1.0
                    else:
                        want = fresh_agent._eval_new_game(new_ms, cs.turn)
                    if abs(t - want) > 1e-5:
                        problems.append(
                            f"row {i}: boundary target {t:.4f} != "
                            f"recomputed {want:.4f} (gr={gr})")
                elif oneply:
                    want = _oneply_target_reference(fresh_agent, cs, cgms)
                    if abs(t - want) > 5e-5:
                        problems.append(
                            f"row {i}: 1-ply target {t:.6f} != slow-path "
                            f"reference {want:.6f} (d={t - want:+.2e})")
                else:
                    opp = 1 - cs.turn
                    opp_ca = cgms.can_offer(opp)
                    v_opp = fresh_agent.evaluate_matchplay(
                        succ, cgms, is_cube_action=opp_ca)
                    want = 1.0 - v_opp
                    if abs(t - want) > 1e-4:
                        problems.append(
                            f"row {i}: bootstrap {t:.4f} != recomputed "
                            f"1-V(succ)={want:.4f} (opp_ca={opp_ca})")
        if not (0.0 <= t <= 1.0):
            problems.append(f"row {i}: target {t} out of [0,1]")
        li += 1
        i += 1
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--from-errors", default=None,
                   help="audit train_from_position's collector seeded "
                        "at this errors-file position instead")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--oracle", default=None, metavar="SPEC",
                   help="audit the supervised-oracle target path "
                        "(mode.oracle_spec): every target re-verified "
                        "against a direct oracle recomputation")
    p.add_argument("--oracle-move-eps", type=float, default=None)
    p.add_argument("--td-move-eps", type=float, default=0.0,
                   help="audit a --td-move-eps run: the collector emits an "
                        "extra row per near-tied successor, so the spy log "
                        "must line up with those too")
    p.add_argument("--oneply", action="store_true",
                   help="audit the 1-ply checker-target path "
                        "(value_oneply_checker_matchplay), verified "
                        "against an independent slow-path reference")
    args = p.parse_args()

    torch.set_num_threads(4)
    net = TDNetwork.load(args.model)
    spy = SpyAgent(net)
    fresh = TDAgent(net)
    rng = random.Random(args.seed)

    total_rows = 0
    all_problems = []
    for ep in range(args.episodes):
        spy.log = []
        spy.clear_matchplay_memo()
        fresh.clear_matchplay_memo()
        if args.from_errors:
            import json
            from train_from_position import load_position, \
                collect_from_position
            err = json.load(open(args.from_errors))[args.index]
            s0, gms0 = load_position(err)
            enc, tgts, _ = collect_from_position(spy, s0, gms0, rng)
        else:
            from trainer import collect_episode
            mode = MatchplayMode(3, oracle_spec=args.oracle,
                                 oracle_move_eps=args.oracle_move_eps,
                                 td_move_eps=args.td_move_eps)
            enc, tgts = collect_episode(spy, mode, rng,
                                        oneply=args.oneply)
        oracle = None
        if args.oracle:
            from trainer import _get_oracle
            oracle = _get_oracle(args.oracle)
        probs = audit_episode(spy, enc, tgts, fresh,
                              oneply=args.oneply, oracle=oracle)
        total_rows += len(tgts)
        for pr in probs:
            all_problems.append(f"ep{ep}: {pr}")

    print(f"audited {args.episodes} episodes, {total_rows} samples")
    if all_problems:
        print(f"PROBLEMS ({len(all_problems)}):")
        for pr in all_problems[:40]:
            print(" ", pr)
    else:
        print("CLEAN: all encodings and targets verified")


if __name__ == "__main__":
    main()
