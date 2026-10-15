"""Input surgery: matchplay1h64 (337) -> matchplay1h64met (350).

Pure append: the 13 MET outcome-price columns go at the end,
zero-initialised -- exact function preservation everywhere.
"""
import argparse
import numpy as np
import torch
from model import TDNetwork

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", required=True)
args = p.parse_args()

old = TDNetwork.load(args.src)
assert old.encoder_name == "matchplay1h64_perspective196", old.encoder_name
assert old.input_size == 337, old.input_size

new = TDNetwork(
    hidden_sizes=list(old.hidden_sizes), input_size=350,
    encoder_name="matchplay1h64met_perspective196",
    output_mode=old.output_mode,
    activation=old.activation,
)
sd_old, sd_new = old.state_dict(), new.state_dict()
first_key = None
for k, v in sd_old.items():
    if v.dim() == 2 and v.shape[1] == 337:
        assert first_key is None
        first_key = k
assert first_key is not None, 'no 337-wide layer found'
W = sd_old[first_key].numpy()
Wn = np.zeros((W.shape[0], 350), dtype=W.dtype)
Wn[:, :337] = W
for k, v in sd_old.items():
    sd_new[k] = torch.from_numpy(Wn) if k == first_key else v.clone()
new.load_state_dict(sd_new)
new.save(args.dst)
print(f"saved {args.dst}: 350 in, {new.encoder_name}")
