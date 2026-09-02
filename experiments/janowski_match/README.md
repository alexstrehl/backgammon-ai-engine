# janowski_match evals — p5jan (prob5 + analytic match cube layer)

The p5jan results quoted in MATCHPLAY.md, and the command that
reproduces them. All vs the pinned gnubg 1.08.003 built from source with
both bearoff DBs (GNUBG_CMD=/usr/local/bin/gnubg), referee MET
Kazaross-XG2. Raw run logs are not retained in-repo; re-run to regenerate.

| arms | n | result (model1) |
|---|---|---|
| p5jan vs gnubg-ext 0-ply, 3pt | 20,000 | **52.1% [51.4, 52.8]** |
| p5jan vs gnubg-ext 1-ply, 3pt | 20,000 | **50.2% [49.5, 50.9]** — tie |
| gnubg-ext 1-ply vs 0-ply, 3pt | 20,000 | 52.3% [51.6, 53.0] |

Repro:

    python eval_matchplay.py \
        --model1 p5jan:best_models/cubeless_prob5_512_512_256_128.pt \
        --model2 gnubg-ext[:1] --matches 20000 --match-length 3 \
        --workers 8 --seed 7
