#!/usr/bin/env python3
"""Distil a student net from a teacher's 0-ply evaluations.

Rows come from position banks written by tools/collect_positions.py.
The teacher (one of our own checkpoints, any net type the collector
supports) labels every row at 0-ply and the student is fit to the labels:
  prob5 teacher    label = sigmoid + prob5_postprocess of the 5 outputs,
                   loss = per-output BCE (train_prob5's loss)
  scalar teacher   label = equity or probability,
                   loss = MSE (the trainer's td_mse_loss is 0.5*MSE)
A bank is accepted only if the teacher could have played it (the
collector's own rule) and its feature width matches.

Student: `--hidden` builds a fresh net with the teacher's input width,
encoder, activation and output mode. `--init-from` copies the leading
prefix of shape-matching hidden layers from a checkpoint of the
student's class (the head stays fresh). `--resume` continues an
existing student.

The output is a normal checkpoint (TDNetwork.save / ProbNetwork.save)
that the trainers can --resume. Its train_params record argv, the md5s
of teacher, banks and init/resume file, and the final validation
scores. Never overwrites.

Usage:
  python tools/distill.py --teacher best_models/cubeless_prob5_512_512_256_256.pt \\
      --init-from best_models/cubeless_prob5_512_512_256_256.pt --hidden 512,512 \\
      --rows bank.npz --epochs 20 --lr 1e-3 --end-lr 1e-5 --out student.pt
"""
import argparse
import hashlib
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

LABEL_CHUNK = 65536


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hidden(s: str):
    try:
        hidden = [int(x) for x in s.split(",")]
    except ValueError:
        raise SystemExit(f"--hidden {s!r}: expected e.g. 512 or 512,512")
    if not hidden or min(hidden) < 1:
        raise SystemExit(f"--hidden {s!r}: sizes must be positive")
    return hidden


def _describe(net) -> str:
    n = sum(p.numel() for p in net.parameters())
    return f"{type(net).__name__} hidden={net.hidden_sizes} {net.output_mode} {n:,} params"


def check_student(teacher, student, what: str) -> None:
    """The student must read the teacher's rows and mimic its outputs."""
    for attr in ("input_size", "encoder_name", "output_mode"):
        if getattr(student, attr) != getattr(teacher, attr):
            raise SystemExit(f"{what}: {attr}={getattr(student, attr)!r}, "
                             f"teacher has {getattr(teacher, attr)!r}")


def build_student(teacher, hidden, init_from):
    """Fresh student of the teacher's kind; `init_from` copies the
    leading prefix of shape-matching hidden layers (head fresh)."""
    from model import ProbNetwork, TDNetwork, load_model
    if teacher.output_mode == "prob5":
        stu = ProbNetwork(hidden_sizes=hidden, input_size=teacher.input_size,
                          activation=teacher.activation,
                          encoder_name=teacher.encoder_name,
                          raw_logits=teacher.raw_logits)
        key = lambda i: f"trunk.{2 * i}"
    else:
        stu = TDNetwork(hidden_sizes=hidden, input_size=teacher.input_size,
                        activation=teacher.activation,
                        encoder_name=teacher.encoder_name,
                        output_mode=teacher.output_mode)
        key = lambda i: f"hidden_layers.{i}"
    n_copied = 0
    if init_from:
        src = load_model(init_from)
        if type(src) is not type(stu) or src.activation != stu.activation:
            raise SystemExit(f"--init-from {init_from}: must be a {type(stu).__name__} "
                             f"with {stu.activation} activation")
        check_student(teacher, src, f"--init-from {init_from}")
        ss, ds = src.state_dict(), stu.state_dict()
        for i in range(len(hidden)):
            kw, kb = key(i) + ".weight", key(i) + ".bias"
            if kw not in ss or ss[kw].shape != ds[kw].shape:
                break   # prefix only: a later match would sit on random predecessors
            ds[kw], ds[kb] = ss[kw].clone(), ss[kb].clone()
            n_copied += 1
        if n_copied == 0:
            raise SystemExit(f"--init-from {init_from}: no hidden layer matches "
                             f"(layer 0 is {tuple(ss[key(0) + '.weight'].shape)}, "
                             f"student needs {tuple(ds[key(0) + '.weight'].shape)})")
        stu.load_state_dict(ds)
    init = (f"{n_copied}/{len(hidden)} hidden layers from {os.path.basename(init_from)}, head fresh"
            if init_from else "random")
    print(f"student  {_describe(stu)}; init {init}")
    return stu


def load_rows(paths, teacher, device):
    """Concatenate the banks' rows onto `device`. Returns (X, md5s)."""
    import numpy as np
    import torch
    from tools.collect_positions import check_game_mode
    encs, md5s = [], []
    for p in paths:
        d = np.load(p)
        e = d["enc"]
        if e.ndim != 2 or e.shape[1] != teacher.input_size:
            raise SystemExit(f"{p}: {e.shape[1] if e.ndim == 2 else '?'} features, "
                             f"teacher wants {teacher.input_size}")
        mode = str(d["game_mode"]) if "game_mode" in d.files else "?"
        if mode != "?":
            try:
                check_game_mode(teacher, mode)
            except SystemExit as err:
                raise SystemExit(f"{p}: {mode} bank, but the teacher cannot play it ({err})")
        print(f"rows     {p}: {len(e):,} ({mode}, acting {d['model']})", flush=True)
        encs.append(torch.from_numpy(e.astype(np.float32, copy=False)).to(device))
        md5s.append(_md5(p))
    X = torch.cat(encs, 0) if len(encs) > 1 else encs[0]
    if len(X) == 0:
        raise SystemExit("no rows")
    return X, md5s


def teacher_labels(teacher, X, device):
    import torch
    from model import prob5_postprocess
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), LABEL_CHUNK):
            o = teacher(X[i:i + LABEL_CHUNK].to(device))
            if teacher.output_mode == "prob5":
                if teacher.raw_logits:
                    o = torch.sigmoid(o)
                o = prob5_postprocess(o)
            outs.append(o)
    return torch.cat(outs, 0)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Distil a student net from a teacher's 0-ply labels on "
                    "position banks from tools/collect_positions.py.")
    ap.add_argument("--teacher", required=True, help="labelling net (.pt)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hidden", help="fresh student, e.g. 512 or 512,512")
    g.add_argument("--resume", help="continue an existing student (.pt)")
    ap.add_argument("--init-from", default=None,
                    help="with --hidden: copy the leading shape-matching hidden layers "
                         "from this checkpoint")
    ap.add_argument("--rows", nargs="+", required=True, metavar="NPZ",
                    help="position banks")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam, linear to --end-lr")
    ap.add_argument("--end-lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="held-out fraction for the validation report; 0 = none")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--rows-on-cpu", action="store_true",
                    help="keep the rows in host memory and copy each minibatch to --device "
                         "(for banks that do not fit in VRAM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.epochs < 1 or a.batch < 1:
        raise SystemExit("--epochs and --batch must be >= 1")
    if not 0 <= a.val_frac < 1:
        raise SystemExit("--val-frac must be in [0, 1)")
    if a.init_from and not a.hidden:
        raise SystemExit("--init-from needs --hidden")
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from model import load_model, prob5_postprocess, prob5_to_equity

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    xdev = torch.device("cpu") if a.rows_on_cpu else dev
    torch.manual_seed(a.seed)
    teacher = load_model(a.teacher).to(dev).eval()
    is_p5 = teacher.output_mode == "prob5"
    print(f"teacher  {a.teacher}: {_describe(teacher)}; device {dev}")
    if a.resume:
        stu = load_model(a.resume)
        check_student(teacher, stu, f"--resume {a.resume}")
        print(f"student  {a.resume}: {_describe(stu)} (resumed)")
    else:
        stu = build_student(teacher, parse_hidden(a.hidden), a.init_from)
    stu = stu.to(dev)

    X, rows_md5 = load_rows(a.rows, teacher, xdev)
    n = len(X)
    t0 = time.time()
    Y = teacher_labels(teacher, X, dev)
    rng = np.random.default_rng(a.seed)
    perm = torch.from_numpy(rng.permutation(n)).to(xdev)
    nv = int(a.val_frac * n)
    vi, ti = perm[:nv], perm[nv:]
    if len(ti) == 0:
        raise SystemExit("no training rows left after the validation split")
    print(f"labels   {n:,} rows in {time.time() - t0:.1f}s; train {len(ti):,} val {nv:,}",
          flush=True)
    unit = "mP" if teacher.output_mode == "probability" else "mEq"

    def loss_fn(pred, y):
        if is_p5:
            if stu.raw_logits:
                return F.binary_cross_entropy_with_logits(pred, y)
            return F.binary_cross_entropy(pred, y)
        return F.mse_loss(pred, y)

    def evaluate():
        """(val loss, mean |student - teacher| x 1000) on the held-out rows."""
        if nv == 0:
            return float("nan"), float("nan")
        stu.eval()
        ls = ae = 0.0
        with torch.no_grad():
            for i in range(0, nv, LABEL_CHUNK):
                idx = vi[i:i + LABEL_CHUNK]
                p, y = stu(X[idx].to(dev)), Y[idx]
                ls += float(loss_fn(p, y)) * len(idx)
                if is_p5:
                    if stu.raw_logits:
                        p = torch.sigmoid(p)
                    p, y = prob5_to_equity(prob5_postprocess(p)), prob5_to_equity(y)
                ae += float((p - y).abs().sum())
        stu.train()
        return ls / nv, 1000.0 * ae / nv

    opt = torch.optim.Adam(stu.parameters(), lr=a.lr)
    steps_per_epoch = (len(ti) + a.batch - 1) // a.batch
    total = steps_per_epoch * a.epochs
    order_gen = torch.Generator().manual_seed(a.seed)
    step = 0
    lr = a.lr
    val0, mae0 = evaluate()
    print(f"epoch   0/{a.epochs}  val_loss={val0:.5f}  mae={mae0:.1f} {unit}", flush=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        order = ti[torch.randperm(len(ti), generator=order_gen).to(xdev)]
        run = 0.0
        for i in range(0, len(order), a.batch):
            lr = a.lr + (a.end_lr - a.lr) * step / max(total - 1, 1)
            for pg in opt.param_groups:
                pg["lr"] = lr
            idx = order[i:i + a.batch]
            loss = loss_fn(stu(X[idx].to(dev)), Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss) * len(idx)
            step += 1
        val, mae = evaluate()
        el = time.time() - t0
        print(f"epoch {ep:3d}/{a.epochs}  lr={lr:.2e}  train={run / len(ti):.5f}  "
              f"val_loss={val:.5f}  mae={mae:.1f} {unit}  {el:.0f}s "
              f"(eta {el / ep * (a.epochs - ep):.0f}s)", flush=True)

    args_list = sys.argv[1:] if argv is None else list(argv)
    tp = {"argv": " ".join(["tools/distill.py", *args_list]),
          "teacher": os.path.basename(a.teacher), "teacher_md5": _md5(a.teacher),
          "rows": [os.path.basename(p) for p in a.rows], "rows_md5": rows_md5,
          "n_rows": n, "n_train": len(ti), "n_val": nv,
          "epochs": a.epochs, "lr": a.lr, "end_lr": a.end_lr, "batch": a.batch,
          "seed": a.seed, "val_loss": val, "val_mae": mae, "val_mae_unit": unit}
    for k in ("init_from", "resume"):
        if getattr(a, k):
            tp[k] = os.path.basename(getattr(a, k))
            tp[k + "_md5"] = _md5(getattr(a, k))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    stu.cpu().save(a.out, train_params=tp)
    print(f"saved    {a.out}  val_loss={val:.5f}  mae={mae:.1f} {unit}")
    return {"val_loss0": val0, "val_mae0": mae0, "val_loss": val, "val_mae": mae}


if __name__ == "__main__":
    main()
