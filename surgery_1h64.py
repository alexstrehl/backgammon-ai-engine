"""Input surgery: matchplay1h (207) -> matchplay1h64 (337).

Function-preserving column map for every input the old net can see:
  base/persp/ca/crawford copied; cube slots 1/2/4 copied and the old
  >=8 column duplicated into 8/16/32/>=64 (identical behavior where
  the old net aliased them); away scalar w becomes one-hot columns
  k*w for k=1..64 (exactly w*away on the old support, and a monotone
  linear prior beyond it); post_crawford starts at zero.
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
assert old.encoder_name == "matchplay1h_perspective196", old.encoder_name
assert old.input_size == 207, old.input_size

new = TDNetwork(
    hidden_sizes=list(old.hidden_sizes), input_size=337,
    encoder_name="matchplay1h64_perspective196",
    output_mode=old.output_mode,
    activation=old.activation,
)
sd_old, sd_new = old.state_dict(), new.state_dict()
first_key = None
for k, v in sd_old.items():
    if v.dim() == 2 and v.shape[1] == 207:
        assert first_key is None, "two 207-wide layers?"
        first_key = k
assert first_key is not None
W = sd_old[first_key].numpy()          # [h, 207]
H = W.shape[0]
Wn = np.zeros((H, 337), dtype=W.dtype)
Wn[:, 0:196] = W[:, 0:196]             # base board
Wn[:, 196:199] = W[:, 196:199]         # persp 3-hot
Wn[:, 199] = W[:, 199]                 # is_cube_action
Wn[:, 200] = W[:, 200]                 # cube 1
Wn[:, 201] = W[:, 201]                 # cube 2
Wn[:, 202] = W[:, 202]                 # cube 4
for c in (203, 204, 205, 206):         # cube 8/16/32/>=64 <- old >=8
    Wn[:, c] = W[:, 203]
for k in range(1, 65):                 # away one-hot from scalars
    Wn[:, 207 + k - 1] = k * W[:, 204]
    Wn[:, 271 + k - 1] = k * W[:, 205]
Wn[:, 335] = W[:, 206]                 # crawford
# 336 (post_crawford) stays zero
for k, v in sd_old.items():
    if k == first_key:
        sd_new[k] = torch.from_numpy(Wn)
    else:
        sd_new[k] = v.clone()
new.load_state_dict(sd_new)
new.save(args.dst)
print(f"saved {args.dst}: hidden={list(old.hidden_sizes)} "
      f"in=337 enc={new.encoder_name}")
