# Cubeless money 4L — model log

**Produces** `cm4L_c2_ep4000000.pt` — a cubeless *equity* net,
`[512,512,256,256]`, encoder `perspective196`.
**Starts from** `dmp_512_512_256_256_repro.pt`
([`dmp_4L_model_log.md`](dmp_4L_model_log.md)).
**Consumed by** [`cubeful_money_4L_model_log.md`](cubeful_money_4L_model_log.md).

Four steps. The DMP net supplies the trunk; the head is re-initialised to a
scalar equity output.

| # | Change | Ply | Ep. (M) | Start LR | End LR |
|---|---|---|---|---|---|
| 1 | DMP → cubeless equity head **(wu20)** | 0 | 15.0 | 5e-5 | 5e-6 |
| 2 | 1-ply LR probe | 1 | 1.0 | 1e-5 | const |
| 3 | — | 1 | 9.0 | 1e-5 | 1e-6 |
| 4 | — | 1 | 4.0 | 9e-6 | **7.38e-6 effective** |

## Commands

```bash
# 1  warm-start the equity head off DMP
$PY train_batch.py --game-mode cubeless-money --output-mode equity \
  --warm-start-equity $MDMP/dmp_512_512_256_256_repro.pt \
  --num-episodes 15000000 --episodes-per-round 1000 --optimizer adam \
  --lr 5e-5 --end-lr 5e-6 --warmup-cycles 20 \
  --workers 32 --device cuda --bf16-collect --pipeline-collect \
  --save $M/cm4L_s0.pt

# 2  1-ply LR probe
$PY train_batch.py --game-mode cubeless-money --resume $M/cm4L_s0.pt --oneply \
  --num-episodes 1000000 --episodes-per-round 1000 --optimizer adam \
  --lr 1e-5 --workers 48 --bf16-collect --pipeline-collect \
  --save $M/probe_1e-5.pt

# 3  1-ply
$PY train_batch.py --game-mode cubeless-money --resume $M/probe_1e-5.pt --oneply \
  --num-episodes 9000000 --episodes-per-round 1000 --optimizer adam \
  --lr 1e-5 --end-lr 1e-6 --workers 48 --bf16-collect --pipeline-collect \
  --save $M/cm4L_c1.pt

# 4  1-ply -- START THIS AS A 20M LEG AND STOP IT AT 4M (see note below).
#    The checkpoint the cubeful stage consumes is cm4L_c2_ep4000000.pt.
$PY train_batch.py --game-mode cubeless-money --resume $M/cm4L_c1.pt --oneply \
  --num-episodes 20000000 --episodes-per-round 1000 --optimizer adam \
  --lr 9e-6 --end-lr 9e-7 --workers 48 --bf16-collect --pipeline-collect \
  --save-every 1000000 --save $M/cm4L_c2.pt
```

## Notes for retraining

- **Step 4's end LR is not 9e-7.** The LR anneal is linear over the *declared*
  `--num-episodes` (`trainer.py:822`:
  `lr = start + (end-start) * episodes_played/num_episodes`). Declaring 20M and
  stopping at 4M means only 20% of the ramp is traversed, so the leg actually
  ends at **7.38e-6**. Declaring `--num-episodes 4000000 --end-lr 9e-7` would
  run the whole leg at a far lower LR and is **not** the same run. Either
  reproduce it as written and stop at 4M, or use the exact equivalent:
  `--num-episodes 4000000 --lr 9e-6 --end-lr 7.38e-6`.
- **The warm-start is not optional.** Random init on cubeless equity sticks at
  about −70 mEq; the DMP trunk is what makes the task learnable.
- Step 1 is sized at 15M because that is what the 3L predecessor needed to go
  from −14.1 mEq (at 5M) to +13.2 mEq.
- `--bf16-collect --pipeline-collect` are throughput flags only. `--workers 32`
  vs `48` reflects what else shared the box, not a recipe choice.
