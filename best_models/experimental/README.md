# Experimental models

Deeper/wider variants of the production cubeful and prob5 nets. Small
measured gains, but not the models of record — the top-level README's
model table is the supported set.

| file | mode | md5 | measured edge |
|---|---|---|---|
| `cubeful_money_512_512_256_256_256.pt` | cubeful-money | `e20105f1` | +2.4 mEq [+0.8, +4.0] p=0.002 vs production 4L (12M games, capped mean) |
| `cubeful_money_768_512_256_256_256_256.pt` | cubeful-money | `9097ea9c` | **+7.2 mEq [+4.4, +9.9]** vs production 4L (0 of 10,000 equity-bootstrap resamples adverse; 4M games, capped mean) |
| `cubeless_prob5_512_512_256_256_256.pt` | cubeless prob5 | `0227a087` | +1.0 mEq [+0.1, +1.9] p=0.024 vs the production 4L256 (10M games) |

The 5-layer nets grew from their production counterparts by near-identity
depth expansion (`eps=0.001`) followed by 1-ply training legs. The
`[768,512,256,256,256,256]` net (877k params) continues that line: a 6th
layer on the 5L cubeful, then the first hidden layer width-expanded
512 → 768, with 1-ply legs after each expansion.
