"""Native C cost path vs the H-H agent, on both committed benchmark pools
(all items), Fox and Aureus at 0-ply: hh_best_move_cubeless / hh_best_move
must pick the agent's move (choose_checker_action / _cubeful) and report
the same successor count. Needs the shim and both models
(bench_common.hh_model_path). Run from the repo root; committed run:
experiments/hedgehog/results/validate_costpath_e8847f4_20261010.txt."""
import os, sys
REPO=os.environ.get("BG_REPO") or os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")); sys.path.insert(0, REPO); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO,"experiments","benchmarks"))
from backgammon_engine import get_legal_plays
from fixedpool_bench import load_cubeful_pool
from bench_common import load_fixedpool, hh_model_path
from third_party_opponents.hedgehog.hedgehog_agent import HedgehogAgent, board_to_hh, _OWNER_MAP
from third_party_opponents.hedgehog.hh_lib import HH

chk = load_fixedpool()
ck, cu = load_cubeful_pool()
for model in ("fox", "aureus"):
    ag = HedgehogAgent(hh_model_path(model), depth=0)
    # CUBELESS: agent choose_checker_action vs hh_best_move_cubeless
    mism=0; cnt_mism=0; N=0
    for s,d in chk:
        plays = get_legal_plays(s,d)
        if not plays: continue
        N+=1
        a_hh = board_to_hh(ag.choose_checker_action(s,d,plays)[1])
        n_c, out = ag.hh.best_move_cubeless(board_to_hh(s), s.turn, d[0], d[1])
        if a_hh != out: mism+=1
        if len(plays) != n_c: cnt_mism+=1
    print(f"{model} CUBELESS n={N}: move mismatches={mism} | successor-count mismatches={cnt_mism} (agent-dedup vs HH explore_delta)")
    # CUBEFUL: agent choose_checker_action_cubeful vs hh_best_move (per-pos cube)
    mism2=0; cnt2=0; M=0
    for s,d,ms in ck:
        plays = get_legal_plays(s,d)
        if not plays: continue
        M+=1
        HH.set_jacoby(ms.jacoby)
        a_hh = board_to_hh(ag.choose_checker_action_cubeful(s,d,ms)[0])
        cv=min(int(ms.cube_value),64); own=_OWNER_MAP[int(ms.cube_owner)]
        n_c, out = ag.hh.best_move(board_to_hh(s), s.turn, d[0], d[1], cv, own, 0)
        if a_hh != out: mism2+=1
        if len(plays) != n_c: cnt2+=1
    print(f"{model} CUBEFUL  n={M}: move mismatches={mism2} | successor-count mismatches={cnt2}")
