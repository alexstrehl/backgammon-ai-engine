# Cubeless Money Model Training Log

## Best Model

**File**: `best_models/cubeless_money_512x2x256_best.pt`
**Architecture**: [512, 512, 256] (495,105 params)
**Encoder**: perspective196 (196 features)
**Output**: equity (linear, unbounded)
**Result**: +21.7 mEq/game vs gnubg 0-ply (50K games, p=0.0004)

## Training Recipe

### Key insight: DMP warm-start is critical

Starting from pre-trained DMP weights vs random init:
- Random init → stuck at -70 mEq (no amount of training helps)
- DMP warm-start → +13.2 mEq after 15M episodes

### Step 1: Convert DMP model to equity output
```python
python3 -c "
from model import TDNetwork
net = TDNetwork.load('best_models/dmp_512_512_256_1ply.pt')
net_eq = TDNetwork(hidden_sizes=list(net.hidden_sizes), output_mode='equity', encoder_name='perspective196')
net_eq.load_state_dict(net.state_dict())
net_eq.save('models/dmp_512_512_256_as_equity.pt')
"
```

### Step 2: Train 5M episodes cubeless money
```bash
python3 train_batch.py --game-mode cubeless-money \
    --num-episodes 5000000 \
    --resume models/dmp_512_512_256_as_equity.pt \
    --optimizer adam --lr 5e-5 --end-lr 5e-6 \
    --batch-size 256 --episodes-per-round 1000 \
    --workers 32 --device cuda \
    --save models/cm_from_dmp_5M.pt
```
→ -14.1 mEq/game (10,191s, 491 ep/s)

### Step 3: Train 10M more episodes
```bash
python3 train_batch.py --game-mode cubeless-money \
    --num-episodes 10000000 \
    --resume models/cm_from_dmp_5M.pt \
    --optimizer adam --lr 5e-5 --end-lr 5e-6 \
    --batch-size 256 --episodes-per-round 1000 \
    --workers 32 --device cuda \
    --save models/cm_from_dmp_15M.pt
```
→ **+13.2 mEq/game** (20,317s, 492 ep/s)

## Full Training History

### From-scratch pipeline (stuck at -70 mEq)

| Step | Architecture | Total eps | mEq (50K games, ±12) |
|---|---|---|---|
| 1 | [80] | 2M | -263.5 |
| 2a | [150] width | 1M | -294.0 |
| 2b | [150,150] depth | 1M | -192.0 |
| 3 | [256,256] width | 1M | -168.5 |
| 4 | [512,512] width | 1M | -100.0 |
| 5 | [512,512,512] depth | 3M | -72.0 |
| 5+10M | [512,512,512] | 13M | -69.4 |
| +1-ply cycles | [512,512,512] | various | -68 to -71 |

All from-scratch models plateau at **~-70 mEq** regardless of training approach.

### DMP warm-start pipeline (successful)

| Step | Total eps | Type | LR | mEq (50K, ±12) |
|---|---|---|---|---|
| DMP warm-start + 5M | 5M | 0-ply | 5e-5→5e-6 | -14.1 |
| +10M more | 15M | 0-ply | 5e-5→5e-6 | +13.2 |
| +1M 1-ply (lr=3e-5) | 16M | 1-ply | 3e-5→3e-6 | +12.9 (same) |
| +2M polish | 18M | 0-ply | 1e-5→1e-6 | +10.3 (same) |
| **+1M 1-ply (lr=1.5e-5)** | **16M** | **1-ply** | **1.5e-5→1.5e-6** | **+21.7** ✓ best |
| +2M polish | 18M | 0-ply | 5e-6→5e-7 | +18.8 (same) |

Key: halving the 1-ply LR (1.5e-5 vs 3e-5) was the breakthrough.
The polish at very low LR (5e-6) neither helped nor hurt.

### Head-to-head comparisons (50K games)

| Matchup | mEq | p-value |
|---|---|---|
| Our best vs gnubg 0-ply | +21.7 | 0.0004 |
| Our best vs cubeful_money branch best 3-layer (-3.8 mEq) | +15.6 | 0.011 |
| Our best vs cubeful_money branch best 4-layer (+12 mEq) | +14.6 | 0.017 |

## Hardware

- CPU: AMD EPYC 7742 64-core (128 threads)
- RAM: 126 GB
- GPU: NVIDIA RTX 4090 (24 GB VRAM)
- Training speed: ~490 ep/s (0-ply), ~41 ep/s (1-ply)
- Total training time: ~8.5 hours for 15M episodes
