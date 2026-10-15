"""Probe a matchplay model's 2a2a cube behavior across race positions
of varying strength, side by side with gnubg's score-aware verdicts.

Usage: python probe_matchplay_cube.py MODEL.pt
"""

import sys

from backgammon_engine import BoardState
from model import TDNetwork
from modes import MatchState
from td_agent import TDAgent


def race_position(white_pips_ahead: int) -> BoardState:
    """Simple symmetric-ish race, WHITE on roll, adjustable lead.
    Both sides have 10 checkers racing home (no contact)."""
    pts = [0] * 24
    # WHITE: 10 checkers in/near home (indices 0..9 area)
    pts[5] = 5
    pts[7] = 3
    pts[9] = 2
    # BLACK mirrored (indices 14..18)
    pts[18] = -5
    pts[16] = -3
    pts[14] = -2
    s = BoardState(points=pts, bar=[0, 0], off=[5, 5], turn=0)
    # Shift WHITE checkers forward to create a lead (toward index 0).
    shift = white_pips_ahead
    while shift >= 2 and pts[7] > 0:
        pts[7] -= 1
        pts[5] += 1
        shift -= 2
    while shift >= 4 and pts[9] > 0:
        pts[9] -= 1
        pts[5] += 1
        shift -= 4
    return s


def main():
    model_path = sys.argv[1]
    net = TDNetwork.load(model_path)
    agent = TDAgent(net)
    ms = MatchState(jacoby=False, match_length=2, my_away=2, opp_away=2)

    try:
        from agents import GnubgNNMatchplayAgent
        gnubg = GnubgNNMatchplayAgent(plies=1)
    except ImportError:
        gnubg = None

    print(f"2a2a cube probe: {model_path}")
    print("lead | model P(win match) | model: dbl v_nd/v_dt/v_dp | gnubg: dbl take")
    for lead in (-8, -4, 0, 4, 8, 12, 16):
        s = race_position(max(lead, 0))
        if lead < 0:
            # Give BLACK the lead instead by switching who it favors:
            # evaluate the mirrored situation by shifting BLACK; cheap
            # approximation — shift WHITE backward is not supported, so
            # mirror via the same position with lead for the opponent.
            s = race_position(-lead)
            # swap colors: negate points, swap bar/off, keep WHITE on roll
            pts = [-p for p in reversed(s.points)]
            s = BoardState(points=pts, bar=[0, 0],
                           off=[s.off[1], s.off[0]], turn=0)
        offer = agent.offer_double_matchplay(s, ms)
        c = offer._cache
        take = agent.respond_to_double_matchplay(s, ms, hint=offer)
        line = (f"{lead:+4d} | {c['v_nd']:.3f} | "
                f"dbl={int(offer.should_double)} "
                f"nd={c['v_nd']:.3f} dt={c['v_dt']:.3f} dp={c['v_dp']:.3f} "
                f"take={int(take)}")
        if gnubg is not None:
            gd, gt = gnubg._cube_decision(s, ms)
            line += f" | gnubg: dbl={int(gd)} take={int(gt)}"
        print(line)


if __name__ == "__main__":
    main()
