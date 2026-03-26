"""
td_agent.py -- TDAgent: torch-based backgammon agent.

Requires torch and a trained TDNetwork model.
Lightweight agents with no torch dependency live in agents.py.
"""

import numpy as np
import torch

from backgammon_engine import switch_turn
from encoding import encode_state, get_encoder
from model import TDNetwork
from agents import Agent


class TDAgent(Agent):
    """Uses a TDNetwork to evaluate positions (perspective encoding).

    Perspective encoding: switch_turn on resulting states so the network
    evaluates from the opponent's view. V = P(opponent wins), mover picks argmin.
    """

    def __init__(self, network: TDNetwork, encoder=None):
        self.network = network
        if encoder is None:
            encoder_name = getattr(network, 'encoder_name', 'perspective196')
            self.encoder = get_encoder(encoder_name)
        else:
            self.encoder = encoder

    def choose_play(self, state, dice, plays):
        encoded = np.stack([
            self.encoder.encode(switch_turn(s)) for _, s in plays
        ])
        x = torch.tensor(encoded, dtype=torch.float32)
        with torch.no_grad():
            values = self.network(x)
        idx = torch.argmin(values).item()
        return plays[idx]
