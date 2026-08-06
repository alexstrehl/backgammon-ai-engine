# Retraining

Exact commands for reproducing the 4L model chain from scratch. One log per
model, in dependency order — each starts from an output of an earlier log (the prob5 log branches off the cubeless log's step-2 intermediate `probe_1e-5.pt`, not its final checkpoint):

| log | produces | steps |
|---|---|---|
| [`model_logs/dmp_4L_model_log.md`](model_logs/dmp_4L_model_log.md) | `dmp_512_512_256_256_repro.pt` | 15 |
| [`model_logs/cubeless_money_4L_model_log.md`](model_logs/cubeless_money_4L_model_log.md) | `cm4L_c2_ep4000000.pt` | 4 |
| [`model_logs/cubeful_money_4L_model_log.md`](model_logs/cubeful_money_4L_model_log.md) | `cf4L_e7.pt` | 17 |
| [`model_logs/cubeless_prob5_model_log.md`](model_logs/cubeless_prob5_model_log.md) | `p5b_5g.pt` | 22 |

Run everything from the repo root. `$PY` is `.venv/bin/python`; `$M` is the
model directory for that stage.

**Read the notes at the bottom of each log before starting.** Two legs
(cubeless step 4, cubeful step 2) were stopped early, and because the LR anneal
is linear over the *declared* `--num-episodes`, their end LR is not the number
in the schedule. Reproducing those rows literally gives a different run.
