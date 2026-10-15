# In-C throughput levers on the collector

Prod 4L nets, `simd` backend (AVX-512), Threadripper 9980X. Profile before any lever:
[p5_profile.md](p5_profile.md).

## Levers (single core, pinned)

Paired A/B, one subprocess per arm in alternating order (A B B A ...), 4 rounds x the
same 5 seeds; ep/s pooled over the calls; B/A = per-seed time ratio (median,
min..max). Cubeful money, Jacoby on.

    taskset -c $CORE .venv/bin/python experiments/c_collect/ab_timing.py \
        --a $OUT/lib_before --b c_api [--cells cubeful0,cubeful1] [--rounds 4]

| lever | cubeful 1-ply ep/s | cubeful 0-ply ep/s | kept | load |
|---|---|---|---|---|
| step 0 (profile counters, off) | 42.7 (A/A: 0.988) | 869.8 (0.999) | - | 3.1 |
| 1. 0-ply play from the 1-ply expansion | 42.7 -> 44.8 (+4.9%, 1.025..1.082) | 869.8 -> 859.6 (path unchanged) | yes | 5.0-6.7 |
| 2. cross-roll row dedup | 44.4 -> 48.1 (+8.2%, 1.073..1.091) | 866.4 -> 871.5 (unchanged) | yes | 7.7-8.9 |
| 3. cube rows folded into the play's batch (0-ply) | 48.1 -> 48.3 (path unchanged) | always fold: 875.8 -> 809.7 (-7.6%); fold when the previous target says no double is likely: 874.2 -> 859.8 (-1.6%) | no | 5.9-8.9 |
| 4. call shape / backend blocking | see below: every variant within -2.7%..+0.9% | within -4.5%..+0.7% | no (defaults stay) | 4.5-17 |
| 5. incremental (NNUE-style) first layer | 49.0 -> 52.0 (+6.2%, 1.057..1.068) | 882.2 -> 919.1 (+3.9%, 1.031..1.058) | yes, on by default | 6.6-4.3 |
| 6. bf16 forward for the 1-ply target rows (`-DBGC_BF16`, OFF by default) | 52.0 -> 57.4 (+10.4%, 1.099..1.109) | 915.8 -> 912.4 (path unchanged) | built only on request; outside the parity bounds, needs its own A/B gate | 5.2 (others idle) |

Cumulative, default build, before the levers vs after the last one (same harness):

| cell | before the levers ep/s | final ep/s | B/A per seed, median (min..max) |
|---|---|---|---|
| cubeful 1-ply | 43.4 | 52.0 | 1.195 (1.188..1.207) |
| cubeless money 1-ply | 27.2 | 32.4 | 1.192 (1.186..1.198) |
| cubeful 0-ply | 882.9 | 910.6 | 1.029 (1.020..1.049) |
| dmp 0-ply | 551.7 | 575.4 | 1.037 (1.020..1.189) |

Parity after the last lever (default build): `tests/test_c_collect.py` +
`tests/test_c_api.py` on ref+simd 168 passed, 8 skipped; `CCOLLECT_FULL=1` on simd
(200 seeds per cell) 35 passed, every cell agrees to the end, |dtarget| max 2.86e-6
(0-ply) and 2.74e-6 (1-ply).

Lever 4: the calls are already one per ply (`BGC_CHUNK` = 4096 rows is never reached:
391 rows on average after lever 2). Inside the simd backend (bitwise neutral), B/A vs
the defaults (CHUNK 96, mr 6, nv 4, sparse lists for every tile), 2 rounds x 5 seeds:

| variant | cubeful 1-ply | cubeful 0-ply |
|---|---|---|
| CHUNK 48 / 64 / 128 / 192 / 256 (build-time) | 0.979 / 0.977 / 1.000 / 1.009 / 0.996 | (19-row calls: one chunk) |
| BGC_SIMD_MR=4 / 5 / 7 / 8 | 0.976 / 0.988 / 0.973 / 0.983 | 0.981 / 0.991 / 0.990 / 1.007 |
| BGC_SIMD_NV=2 / 8 | 0.975 / 0.998 | 0.955 / 0.992 |
| BGC_SIMD_SPARSE_MR=6 / 0 (lists only for short tiles / never) | 0.999 / 0.441 | 1.001 / 0.431 |

The defaults are the optimum within noise.

Lever 5: successor rows differ from their parent in 5.7 of ~37 nonzero inputs on
average; layer 0 becomes the parent's pre-activation plus those terms. Layer 0 drops
from 13.0% to 6.5% of the forward. vs the plain forward: max |d| 7.2e-7 on 0-ply
scores, 1.8e-7 on 1-ply values. `BGC_SIMD_DELTA0=0` turns it off.

Lever 6 (bf16, `CFLAGS_EXTRA=-DBGC_BF16`, off by default): value1's rows (the 1-ply
targets only) go through VDPBF16PS kernels for hidden layers 1.. (bf16 weights and
activations, fp32 accumulation); layer 0 and the output layer stay fp32. The forward is
18% cheaper per row (0.93 vs 1.13 us). Speed: cubeful 1-ply 52.0 -> 57.4 ep/s (+10.4%),
cubeless 1-ply 32.3 -> 36.1 (+11.7%), dmp 1-ply 29.7 -> 33.7 (+13.6%); cubeful 0-ply
unchanged. Target error vs fp32 (400 pool positions per net; 20 1-ply collector
episodes, the same rows in both):

    OUT=$OUT/bf16 CFLAGS_EXTRA=-DBGC_BF16 bash c_api/build.sh simd

| net | where | n | max abs | mean abs | p99 abs | signed mean |
|---|---|---|---|---|---|---|
| cubeful 4L | pool value_1ply | 400 | 9.1e-3 | 1.0e-3 | 5.0e-3 | +2.4e-4 |
| cubeful 4L | collector checker targets | 677 | 4.3e-3 | 8.6e-4 | 3.4e-3 | +2.7e-4 |
| cubeless money 4L | pool value_1ply | 400 | 7.2e-3 | 9.2e-4 | 3.6e-3 | -2.9e-4 |
| cubeless money 4L | collector targets | 1133 | 7.7e-3 | 1.0e-3 | 5.4e-3 | -3.6e-4 |
| dmp 4L | pool value_1ply | 400 | 1.2e-3 | 1.5e-4 | 9.5e-4 | +3.4e-5 |
| dmp 4L | collector targets | 1176 | 1.4e-3 | 1.5e-4 | 8.3e-4 | +1.1e-5 |

The signed mean is a systematic shift of 0.3-0.4 mEq per target, outside the parity
bounds: bf16 needs its own A/B twin gate before any recipe uses it.

Lever 3: folding the cube rows into the play's batch costs more than the two calls
(the simd tile walks the union of the mover's and the opponent's nonzero inputs:
30.6 us folded vs 22.8 + 4.1 us).

## Multi-process scaling

N independent processes, each pinned to its own physical core (spread evenly over the
64), each running `Engine.collect` back to back on its own seeds for 20 s (1-ply: 2
episodes per call, 0-ply: 50). Worker MHz = mean `cpuinfo_avg_freq` of the worker
cores at 1/4, 1/2, 3/4 of the window; "per MHz" divides the efficiency by the clock
ratio to N = 1.

### Before the levers

| cell | N | aggregate ep/s | ep/s per process, mean (min..max) | efficiency vs N=1 | worker MHz | efficiency per MHz | load (1-min, others runnable) |
|---|---|---|---|---|---|---|---|
| cubeful 1-ply | 1 | 41 | 41.0 | 1.00 | 5062 | 1.00 | 10.8, - |
| cubeful 1-ply | 16 | 613 | 38.3 (32.6..40.6) | 0.93 | 4713 | 1.00 | 8.3, 2 |
| cubeful 1-ply | 32 | 1171 | 36.6 (32.3..40.2) | 0.89 | 4517 | 1.00 | 10.8, 1 |
| cubeful 1-ply | 48 | 1540 | 32.1 (29.7..36.3) | 0.78 | 3977 | 1.00 | 17.1, 2 |
| cubeful 1-ply | 64 | 1794 | 28.0 (25.7..31.0) | 0.68 | 3549 | 0.98 | 26.1, 3 |
| cubeful 0-ply | 1 | 866 | 866.0 | 1.00 | 5048 | 1.00 | 37.2, 1 |
| cubeful 0-ply | 16 | 12666 | 791.6 (655.0..815.9) | 0.91 | 4716 | 0.98 | 27.2, 2 |
| cubeful 0-ply | 32 | 24182 | 755.7 (657.4..806.2) | 0.87 | 4488 | 0.98 | 24.4, 2 |
| cubeful 0-ply | 48 | 31732 | 661.1 (589.7..726.4) | 0.76 | 3945 | 0.98 | 26.8, 1 |
| cubeful 0-ply | 64 | 37221 | 581.6 (510.6..631.4) | 0.67 | 3522 | 0.96 | 33.1, 2 |

### After lever 2

| cell | N | aggregate ep/s | ep/s per process, mean (min..max) | efficiency vs N=1 | worker MHz | efficiency per MHz | load (1-min, others runnable) |
|---|---|---|---|---|---|---|---|
| cubeful 1-ply | 1 | 46 | 46.2 | 1.00 | 5046 | 1.00 | 36.7, - |
| cubeful 1-ply | 16 | 688 | 43.0 (36.5..45.2) | 0.93 | 4707 | 1.00 | 26.8, 2 |
| cubeful 1-ply | 32 | 1318 | 41.2 (36.0..44.9) | 0.89 | 4518 | 1.00 | 24.1, 1 |
| cubeful 1-ply | 48 | 1737 | 36.2 (33.1..41.0) | 0.78 | 3976 | 0.99 | 26.6, 3 |
| cubeful 1-ply | 64 | 2025 | 31.6 (29.1..35.4) | 0.68 | 3572 | 0.97 | 33.0, 1 |
| cubeful 0-ply | 1 | 862 | 862.2 | 1.00 | 5046 | 1.00 | 42.1, 1 |
| cubeful 0-ply | 16 | 12675 | 792.2 (649.5..814.2) | 0.92 | 4718 | 0.98 | 30.7, 2 |
| cubeful 0-ply | 32 | 24152 | 754.7 (663.4..804.4) | 0.88 | 4485 | 0.98 | 25.7, 1 |
| cubeful 0-ply | 48 | 31827 | 663.1 (591.1..728.2) | 0.77 | 3961 | 0.98 | 27.8, 1 |
| cubeful 0-ply | 64 | 37188 | 581.1 (514.6..636.0) | 0.67 | 3525 | 0.96 | 33.8, 1 |

### After the last lever (default build)

| cell | N | aggregate ep/s | ep/s per process, mean (min..max) | efficiency vs N=1 | worker MHz | efficiency per MHz | load (1-min, others runnable) |
|---|---|---|---|---|---|---|---|
| cubeful 1-ply | 1 | 49 | 48.8 | 1.00 | 5062 | 1.00 | 12.3, 1 |
| cubeful 1-ply | 16 | 729 | 45.5 (38.8..48.1) | 0.93 | 4713 | 1.00 | 9.3, 2 |
| cubeful 1-ply | 32 | 1399 | 43.7 (38.7..47.6) | 0.90 | 4528 | 1.00 | 11.5, 1 |
| cubeful 1-ply | 48 | 1843 | 38.4 (35.0..43.4) | 0.79 | 3996 | 1.00 | 17.6, 2 |
| cubeful 1-ply | 64 | 2143 | 33.5 (30.4..37.5) | 0.69 | 3587 | 0.97 | 27.4, 1 |
| cubeful 0-ply | 1 | 897 | 897.1 | 1.00 | 5051 | 1.00 | 38.1, 1 |
| cubeful 0-ply | 16 | 13249 | 828.1 (811.7..840.7) | 0.92 | 4715 | 0.99 | 27.9, 2 |
| cubeful 0-ply | 32 | 25004 | 781.4 (755.1..836.6) | 0.87 | 4472 | 0.98 | 24.8, 1 |
| cubeful 0-ply | 48 | 32766 | 682.6 (562.5..751.2) | 0.76 | 3958 | 0.97 | 27.1, 1 |
| cubeful 0-ply | 64 | 38368 | 599.5 (495.2..656.3) | 0.67 | 3551 | 0.95 | 33.3, 1 |

vs before the levers: 1-ply +19.7% at 48 (1540 -> 1843) and +19.5% at 64 (1794 ->
2143); 0-ply +3.3% / +3.1%. bf16 build, cubeful 1-ply: 53 / 808 / 1537 / 1992 / 2275
ep/s at N = 1 / 16 / 32 / 48 / 64 (+8% at 48, +6% at 64 over the default build).

The collector does not flatten on memory bandwidth: per clock cycle it scales
linearly to 64 cores at 1-ply (0.97-1.00) and within 2-5% at 0-ply. The all-core
AVX-512 clock drops from 5.05 GHz (one core) to 3.55 GHz (64), which alone gives the
0.67-0.69 efficiency at 64.
