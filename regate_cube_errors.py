"""
regate_cube_errors.py -- Re-score a model on positions mined by
mine_cube_errors.py: how many cube decisions now agree with gnubg,
and how much MWC the disagreements cost.

Usage:
    python regate_cube_errors.py --model models/foo.pt \
        --errors /tmp/cube_errors.json
"""

import argparse
import json

import numpy as np
import torch

from train_from_position import load_position


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--errors", default="/tmp/cube_errors.json")
    args = p.parse_args()

    torch.set_num_threads(8)
    from model import TDNetwork
    from td_agent import TDAgent
    net = TDNetwork.load(args.model)
    agent = TDAgent(net)

    errs = json.load(open(args.errors))
    dbl_fixed = dbl_still = take_fixed = take_still = 0
    loss_before, loss_after = 0.0, 0.0
    for e in errs:
        state, gms = load_position(e)
        offer = agent.offer_double_matchplay(state, gms)
        c = offer._cache
        d_m = offer.should_double
        t_m = c["v_dp"] > c["v_dt"]
        mwc_nd, mwc_dt, mwc_dp = (e["gnubg_mwc_nd"], e["gnubg_mwc_dt"],
                                  e["gnubg_mwc_dp"])
        g_d, g_t = e["gnubg_double"], e["gnubg_take"]

        def val(d, t):
            return mwc_nd if not d else (mwc_dt if t else mwc_dp)

        v_opt = val(g_d, g_t)
        dbl_loss = (v_opt - val(d_m, g_t)) * 1000.0
        take_loss = abs(mwc_dt - mwc_dp) * 1000.0 if (g_d and t_m != g_t) \
            else 0.0

        if e["dbl_loss_mmwc"] > 0.5:
            loss_before += e["dbl_loss_mmwc"]
            loss_after += max(0.0, dbl_loss)
            if dbl_loss <= 0.5:
                dbl_fixed += 1
            else:
                dbl_still += 1
        if e["take_loss_mmwc"] > 0.5:
            loss_before += e["take_loss_mmwc"]
            loss_after += take_loss
            if take_loss <= 0.5:
                take_fixed += 1
            else:
                take_still += 1

    n_dbl = dbl_fixed + dbl_still
    n_take = take_fixed + take_still
    print(f"model: {args.model}")
    print(f"  double errors fixed: {dbl_fixed}/{n_dbl}"
          f"  take errors fixed: {take_fixed}/{n_take}")
    print(f"  total mMWC loss on these decisions: "
          f"{loss_before:.0f} -> {loss_after:.0f}"
          f"  ({100*(1-loss_after/max(loss_before,1e-9)):.0f}% reduction)")


if __name__ == "__main__":
    main()
