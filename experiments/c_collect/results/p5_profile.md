# Profile: where a bgc_collect position's time goes

`-DBGC_PROFILE` counters (`c_api/puretd_internal.h`, compiled out by default; an A/A
of the default build is 0.999 / 0.988 on cubeful 0-ply / 1-ply), before the levers of
[p5_results.md](p5_results.md). Prod 4L cubeful net (`cubeful_money_512_512_256_256.pt`),
`simd` backend (AVX-512 path), Jacoby on, one process pinned to one core of a
Threadripper 9980X, 1-min load ~4.

    OUT=$OUT/prof CFLAGS_EXTRA=-DBGC_PROFILE bash c_api/build.sh simd
    PURETD_LIBDIR=$OUT/prof taskset -c 61 .venv/bin/python \
        experiments/c_collect/p5_profile.py --episodes1 60 --episodes0 3000

Profiled rates: 42.3 ep/s at 1-ply, 851 at 0-ply (unprofiled: 43.7 and 892; the timer
pairs cost ~3% at 0-ply, ~0% at 1-ply).

## Cubeful 1-ply targets (one turn = one 1-ply target + the 0-ply play)

Per call, us. V1 = `value1` (the target), SEL = `bgc_best_move` at 0-ply (the play),
CUBE = `bgc_offer_double` (0-ply, both target plies), COLLECT = the collector's own
sample rows and loop (per episode).

| context | calls | us/call | share | movegen | encode | cube cols | forward | reduce | other | rows/forward |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 2198 | 615.5 | 95.3% | 81.9 | 20.3 | 0.23 | 511.8 | 0.29 | 1.0 | 429.3 |
| SEL | 2198 | 27.0 | 4.2% | 3.5 | 0.72 | 0.03 | 22.5 | 0.06 | 0.14 | 18.2 |
| CUBE | 1544 | 4.6 | 0.5% | - | 0.09 | 0.04 | 4.35 | - | 0.11 | 2.0 |
| COLLECT (per episode) | 60 | 11.7 | 0.0% | - | 6.8 | - | - | - | 4.9 | - |

By step: forward 83.2%, move generation 13.2%, encode 3.3%, the rest < 0.5%. Per
target: 427.8 replies, 429.3 forwarded rows, one forward call. 9.4% of the rows are
duplicate boards reached by different rolls.

## Cubeful 0-ply targets (one turn = the cube phase + the 0-ply play)

| context | calls | us/call | share | movegen | encode | cube cols | forward | reduce | other | rows/forward |
|---|---|---|---|---|---|---|---|---|---|---|
| SEL | 111790 | 27.9 | 88.3% | 3.7 | 1.12 | 0.03 | 22.8 | 0.06 | 0.13 | 19.0 |
| CUBE | 78011 | 4.3 | 9.6% | - | 0.11 | 0.04 | 4.09 | - | 0.11 | 2.0 |
| VAL (forced pass) | 10062 | 4.4 | 1.3% | - | 0.09 | 0.02 | 4.22 | - | 0.09 | 1.0 |
| COLLECT (per episode) | 3000 | 9.4 | 0.8% | - | 5.4 | - | - | - | 4.0 | - |

By step: forward 82.5%, move generation 11.8%, encode 4.3%, rest 1.4%. The small
forwards (CUBE 2 rows, VAL 1 row) are 10.9% of the 0-ply time.

## The forward

`simd` per layer (k entries = the tile's nonzero-k list length x tile rows, per row;
MACs over 16-wide padded outputs):

| layer | K x M | 1-ply: share of forward | k entries/row | density | GMAC/s | 0-ply: share | k entries/row |
|---|---|---|---|---|---|---|---|
| 0 | 200x512 | 13.0% | 36.6 | 0.183 | 121 | 13.7% | 35.4 |
| 1 | 512x512 | 52.8% | 165.3 | 0.323 | 134 | 51.5% | 159.4 |
| 2 | 512x256 | 20.9% | 113.3 | 0.221 | 116 | 20.7% | 107.5 |
| 3 | 256x256 | 10.8% | 56.7 | 0.221 | 112 | 10.9% | 53.9 |
| 4 | 256x1 | 2.5% | 33.8 | 0.132 | 18 | 3.2% | 31.7 |

1-ply: 147k MACs walked per row of 561k dense (26%), 123 GMAC/s (~70% of the core's
fp32 FMA peak), 1.20 us per row; 0-ply: 1.37 us per row. Hot-weight forward cost by
batch size (`bench_forward.py --backends simd`): n=1 2.1 us, 2 2.7, 4 5.4, 8 13.5, 16
25.5, 32 51, 64 106, 128 207, 400 663. In the collector the 1-2 row forwards take
4.1-4.4 us: the weights (2.25 MB) do not fit the 1 MB L2.

## Levers

| lever | addresses | ceiling from this profile |
|---|---|---|
| 1. selection rows from the expansion | SEL forward (22.5 us of 642 per 1-ply turn) | ~3.5% at 1-ply; 0 at 0-ply (no expansion) |
| 2. cross-roll dedup | 9.4% of V1 rows | ~7.8% of the 1-ply time, minus the hashing |
| 3. cube rows folded into a batch | CUBE 2-row forward: 0.5% at 1-ply; 9.6% at 0-ply (+1.3% VAL) | 0-ply only: a 2-row call costs ~4.2 us alone, ~2.4 us of compute inside a batch, so ~4% |
| 4. call shape | V1 is already one call per target; `BGC_CHUNK` (4096) is never reached (p90 ~940 rows); the backend chunks by 96 internally | the small calls are lever 3 |
| 5. incremental first layer | layer 0 = 13% of the forward; a reply changes ~4-16 of the parent's ~36 nonzero inputs | up to ~8-10% of the forward |
| 6. bf16 (VDPBF16PS, 2x MACs per instruction) | every layer, compute-bound | up to ~2x on the forward (83%), outside the parity bounds |
