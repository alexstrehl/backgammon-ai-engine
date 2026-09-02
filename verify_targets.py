"""Independent verification of every emitted matchplay target.

audit_collector walks the spy log in lockstep with emitted rows, which
desyncs as soon as the collector emits EXTRA rows (oracle eps-sets, and
now --td-move-eps): its bookkeeping mis-pairs a row with another row's
state, producing large phantom mismatches. The encodings themselves line
up one-for-one, so the walk is the fragile part, not the data.

This verifier drops the walk entirely: for each logged (state,
match_state, is_cube_action) it recomputes the expected target from
audit_collector._oneply_target_reference -- pure-python movegen, one
fresh evaluate_matchplay per successor, sharing none of the batched fast
path -- and compares against the emitted value. Rows are classified so
the KNOWN mixed-ply cube behaviour is reported separately from real
defects:

  checker / eps rows : target == T1(state), composed with the cube option
                       max(T1, min(v_dt, v_dp)) when the window is live
  cube rows (ca=1)   : target == max(v_nd_0ply, min(v_dt, v_dp)) -- the
                       matchplay collector composes cube targets from
                       0-PLY reads even under --oneply (see the
                       --cube-targets-1ply guard in trainer.collect_episode)

Usage:
    python verify_targets.py --model M.pt [--episodes 2] [--td-move-eps 0.01]
"""
import argparse
import random

import audit_collector as A
from met import MatchEquityTable, resolve_met_path
from model import TDNetwork
from modes import MatchplayMode
from td_agent import TDAgent
from trainer import collect_episode

TOL = 3e-6


def verify_episode(spy, fresh, enc, tgts):
    encs = [e for e in spy.log if e[0] == "enc"]
    clean = cube_0ply = 0
    problems = []
    for i, (_, state, gms, ca, _e) in enumerate(encs):
        if i >= len(tgts):
            break
        ref = A._oneply_target_reference(fresh, state, gms)
        cache = fresh.offer_double_matchplay(state, gms)._cache if ca else None
        if cache is not None:
            ref = max(ref, min(cache["v_dt"], cache["v_dp"]))
        if abs(tgts[i] - ref) <= TOL:
            clean += 1
            continue
        if cache is not None:
            v_nd0 = fresh.evaluate_matchplay(state, gms, is_cube_action=False)
            if abs(tgts[i] - max(v_nd0, min(cache["v_dt"],
                                            cache["v_dp"]))) <= TOL:
                cube_0ply += 1
                continue
        problems.append(
            f"row {i} (ca={ca}): emitted {tgts[i]:.8f} != reference "
            f"{ref:.8f}  (d={tgts[i]-ref:+.2e})")
    return clean, cube_0ply, problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--match-length", type=int, default=3)
    p.add_argument("--td-move-eps", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()

    met = MatchEquityTable.from_xml(resolve_met_path("auto"))
    mode = MatchplayMode(a.match_length, td_move_eps=a.td_move_eps)
    tot_clean = tot_cube = 0
    all_problems = []
    for ep in range(a.episodes):
        spy = A.SpyAgent(TDNetwork.load(a.model), met=met)
        fresh = TDAgent(TDNetwork.load(a.model), met=met)
        enc, tgts = collect_episode(spy, mode, random.Random(a.seed + ep),
                                    oneply=True)
        c, q, probs = verify_episode(spy, fresh, enc, tgts)
        tot_clean += c
        tot_cube += q
        all_problems += [f"ep{ep}: {x}" for x in probs]
    print(f"verified {tot_clean} rows against the independent 1-ply "
          f"reference; {tot_cube} cube rows matched the documented 0-ply "
          f"composition")
    if all_problems:
        print(f"PROBLEMS ({len(all_problems)}):")
        for x in all_problems:
            print("  " + x)
        raise SystemExit(1)
    print("clean")


if __name__ == "__main__":
    main()
