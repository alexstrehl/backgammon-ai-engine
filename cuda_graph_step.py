"""CUDA-graph replay of the batch train step (`BG_TRAIN_CUDA_GRAPH`, on by
default on a cuda device; `BG_TRAIN_CUDA_GRAPH=0` runs the
stock eager step).

`Trainer.train_step_dev` on a 4L 512 net is launch-bound: ~52 kernel
launches per Adam step cost ~0.49 ms of host time against ~0.22 ms of GPU
work, and the host part grows 1.6x while collection workers load the box;
train_prob5's step, with its five per-output losses, ~100 launches against
~0.26 ms.
`GraphedTrainStep` captures the fixed-shape part of the step once and
replays it per batch:

  * the batch is gathered into static input buffers (`index_select`, a
    copy: the same values `pool[idx]` gives);
  * the graph holds forward, loss and backward, and in mode "adam"
    (torch.optim.Adam with default options) also the step-independent
    half of Adam (moment updates and the sqrt); mode "fwdbwd" works with
    any optimizer and grad_clip;
  * whatever depends on the step count runs eagerly after the replay:
    the stock `optimizer.step()` (fwdbwd), or Adam's bias-corrected tail
    with the same scalars and the same foreach calls as
    `torch.optim.adam._multi_tensor_adam` (adam). Mode adam's foreach
    lists cut each tensor into `ADAM_CHUNK`-element views, which keeps
    the same kernels busier (below);
  * the ragged last batch of a round, and the run's first full batch
    (Adam's lazy state init), run the stock eager step.

Bitwise contract: the replayed kernels are the ones the eager step
launches, in the same order, with the same arguments, so weights,
optimizer state and per-batch losses are bitwise identical to the eager
step (fp32, no TF32, nothing fused or compiled). The first graphed
step is also run eagerly from the same state and compared tensor by
tensor. On any difference the run falls back to
the eager step (with a WARN) from the eager result.
tests/test_train_step_bitwise.py checks whole runs.

What is graphed is a `StepSpec`: the network, its optimizer, a loss
callable `loss_fn(network(x), y)` returning the loss or a tuple whose
first element is the loss (the rest, e.g. train_prob5's per-output
losses, come out of the replay beside it), and the stock eager step it
replaces. `Trainer` gives one (`step_spec`). train_prob5 builds its own,
with (B, 5) targets, and keeps the network off the device between train
phases, so it brings the parameters back with `bind()` (into the storage
the graph replays on) instead of `network.to(device)`, and calls
`unbind()` before moving them away (tests/test_train_prob5_step_bitwise.py).
"""

from __future__ import annotations

import os
from typing import Callable, List, NamedTuple, Optional, Union

import torch

MODES = ("adam", "fwdbwd")
ENV = "BG_TRAIN_CUDA_GRAPH"

# Mode adam runs Adam's foreach ops on pieces of at most this many
# elements. multi_tensor_apply gives each 65,536-element chunk of a list
# one thread block, so the 10 tensors of a 4L 512 net (563k parameters)
# keep ~15 blocks busy on a 170-SM GPU. Same kernels on more, smaller
# list entries (the 7 Adam ops in a graph): one entry per tensor
# 61.5 us/step, 16384 -> 22.5, 8192 -> 24.6, 4096 -> 30.7 (more launches);
# bitwise equal at every size. BG_TRAIN_GRAPH_ADAM_CHUNK overrides it,
# 0 = one entry per tensor (as _multi_tensor_adam).
ADAM_CHUNK = 16384
CHUNK_ENV = "BG_TRAIN_GRAPH_ADAM_CHUNK"


def mode_from_env() -> Optional[str]:
    """`BG_TRAIN_CUDA_GRAPH`: unset or 1 = "auto" (adam where
    `unsupported_reason` allows it, else fwdbwd; the default on a cuda
    device); 0 = off, the stock eager step; or a mode
    name."""
    v = os.environ.get(ENV, "").strip()
    if v == "0":
        return None
    if v in ("", "1"):
        return "auto"
    if v not in MODES:
        raise ValueError(f"{ENV}={v!r}: expected 0, 1 or one of {MODES}")
    return v


def env_explicit() -> bool:
    """True when `BG_TRAIN_CUDA_GRAPH` is set, False for the default."""
    return os.environ.get(ENV, "").strip() != ""


def env_label() -> str:
    """`BG_TRAIN_CUDA_GRAPH` as the NOTE lines name it."""
    v = os.environ.get(ENV, "").strip()
    return f"{ENV}={v}" if v else f"{ENV} (default on; {ENV}=0 for the eager step)"


class StepSpec(NamedTuple):
    """The train step GraphedTrainStep replays.

    loss_fn(network(x), y) -> the loss, or a tuple (loss, *extras) of
        tensors (extras: e.g. per-output losses, detached on the way out).
    eager_step(x, y) -> the stock step on a batch: zero_grad, backward,
        (grad_clip,) optimizer.step(); returns loss_fn's result detached.
    device: where the step runs, for a network that is not kept there
        (GraphedTrainStep.bind moves it in); None = the parameters'.
    """
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable
    eager_step: Callable
    grad_clip: Optional[float] = None
    device: Union[str, torch.device, None] = None


def step_spec(trainer) -> StepSpec:
    """A StepSpec as is, or the one of a `trainer.Trainer`
    (`train_step_dev` and the agent's loss)."""
    if isinstance(trainer, StepSpec):
        return trainer
    return StepSpec(trainer.agent.network, trainer.optimizer,
                    trainer.agent.loss, trainer.train_step_dev,
                    trainer.grad_clip)


def resolve_mode(trainer, mode: str):
    """(mode, None) for a usable mode, (None, reason) otherwise. "auto"
    picks the first of MODES that is supported. `trainer`: a Trainer or
    a StepSpec."""
    why = None
    for m in (MODES if mode == "auto" else (mode,)):
        why = unsupported_reason(trainer, m)
        if why is None:
            return m, None
    return None, why


def unsupported_reason(trainer, mode: str) -> Optional[str]:
    """None if the step of `trainer` (a Trainer or a StepSpec) can be
    graphed in `mode`, else why not."""
    if mode not in MODES:
        return f"unknown mode {mode!r} (one of {', '.join(MODES)})"
    if not torch.cuda.is_available():
        return "CUDA not available"
    spec = step_spec(trainer)
    net = spec.network
    params = list(net.parameters())
    if not params:
        return "the network has no parameters"
    if spec.device is None:
        if any(p.device.type != "cuda" for p in params):
            return "the network is not on a CUDA device"
    elif torch.device(spec.device).type != "cuda":
        return f"device is {spec.device}"
    elif any(True for _ in net.buffers()):
        # bind() moves parameters only.
        return "registered buffers in a network that leaves the device"
    # Dropout draws from the graph-safe RNG (different numbers). BatchNorm
    # would update buffers under the warm-up passes.
    for m in net.modules():
        if isinstance(m, (torch.nn.modules.dropout._DropoutNd,
                          torch.nn.modules.batchnorm._BatchNorm)):
            return f"{type(m).__name__} in the network"
    if mode == "adam":
        opt = spec.optimizer
        if type(opt) is not torch.optim.Adam:
            return f"mode 'adam' needs torch.optim.Adam (got {type(opt).__name__})"
        if len(opt.param_groups) != 1:
            return "mode 'adam' needs one parameter group"
        g = opt.param_groups[0]
        bad = [k for k, ok in (
            ("amsgrad", not g["amsgrad"]),
            ("weight_decay", g["weight_decay"] == 0),
            ("maximize", not g["maximize"]),
            ("foreach", g["foreach"] in (None, True)),
            ("fused", not g["fused"]),
            ("capturable", not g["capturable"]),
            ("differentiable", not g["differentiable"]),
            ("tensor lr/betas", not any(torch.is_tensor(v) for v in
                                        (g["lr"], *g["betas"]))),
        ) if not ok]
        if bad:
            return "mode 'adam' needs default Adam options (" + ", ".join(bad) + ")"
        if spec.grad_clip is not None:
            return "mode 'adam' does not support grad_clip"
        if any(p.dtype != torch.float32 for p in params):
            return "mode 'adam' needs fp32 parameters"
    return None


class GraphedTrainStep:
    """Drop-in for the stock step, `eager_step(pool_x[idx], pool_y[idx])`
    (`trainer.train_step_dev` for a Trainer), on full batches of
    `batch_size` rows. `trainer`: a `trainer.Trainer` or a StepSpec.
    Not thread-safe; one per training run."""

    def __init__(self, trainer, batch_size: int, n_features: int,
                 mode: str = "fwdbwd", log=print, target_shape=()):
        why = unsupported_reason(trainer, mode)
        if why is not None:
            raise ValueError(f"cannot graph the train step: {why}")
        self.spec = step_spec(trainer)
        self.network = self.spec.network
        self.optimizer = self.spec.optimizer
        self.mode = mode
        self.batch_size = batch_size
        self.log = log
        self.device = (torch.device(self.spec.device)
                       if self.spec.device is not None
                       else next(self.network.parameters()).device)
        self.params: List[torch.Tensor] = [
            p for g in self.optimizer.param_groups for p in g["params"]]
        self.x = torch.zeros((batch_size, n_features), device=self.device)
        self.y = torch.zeros((batch_size, *target_shape), device=self.device)
        self.adam_chunk = int(os.environ.get(CHUNK_ENV, ADAM_CHUNK))
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.storage: Optional[List[torch.Tensor]] = None   # see bind()
        self.warm = False          # an eager step has run (lazy optimizer state)
        self.disabled = False      # set when the self-check failed
        self.replays = 0

    # ── public ───────────────────────────────────────────────────────

    def step(self, pool_x: torch.Tensor, pool_y: torch.Tensor,
             idx: torch.Tensor):
        """One optimizer step on rows `idx` of the device pool; returns
        what the eager step returns (the detached loss, or the tuple of
        loss_fn), without a host sync."""
        if (self.disabled or not self.warm or len(idx) != self.batch_size
                or pool_x.shape[1:] != self.x.shape[1:]
                or pool_y.shape[1:] != self.y.shape[1:]):
            # The run's first step is eager: it creates the optimizer's
            # lazy state, which the graph and the self-check need.
            self.warm = True
            return self.spec.eager_step(pool_x[idx], pool_y[idx])
        if self.graph is None:
            self._capture()
            return self._checked_step(pool_x, pool_y, idx)
        return self._replay(pool_x, pool_y, idx)

    def bind(self):
        """`network.to(device)` for a network that leaves the device
        between train phases (train_prob5: its collectors, C export and
        checkpoints read the CPU net), into the same device storage every
        time: the storage the graph replays on (`Module.to` would give
        the parameters new storage). The values are copied in as `.to`
        copies them. The grads are dropped. The stock step's first
        `zero_grad()` sets them to None before anything reads them. The
        first call is a plain `network.to(device)` and fixes the storage."""
        with torch.no_grad():
            if self.storage is None:
                self.network.to(self.device)
                self.storage = [p.data for p in self.params]
            else:
                for p, s in zip(self.params, self.storage):
                    if p.data_ptr() != s.data_ptr():
                        s.copy_(p.data)
                        p.data = s
        for p in self.params:
            p.grad = None

    def unbind(self):
        """Call before moving a bound network off the device: drops the
        grads, which may be the graph's own buffers (mode fwdbwd), so that
        `Module.to` does not move those tensors' data with the network."""
        for p in self.params:
            p.grad = None

    # ── internals ────────────────────────────────────────────────────

    def _forward_backward(self):
        out = self.spec.loss_fn(self.network(self.x), self.y)
        (out[0] if isinstance(out, tuple) else out).backward()
        return _detached(out)

    def _capture(self):
        opt = self.optimizer
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        # Warm-up on the capture stream (cuBLAS workspace, autograd
        # streams): forward and backward only, so no weight or optimizer
        # state changes. The grads are dropped below.
        with torch.cuda.stream(stream):
            for _ in range(2):
                for p in self.params:
                    p.grad = None
                self._forward_backward()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        for p in self.params:
            p.grad = None          # capture the grad-is-None (steal) path
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.out = self._forward_backward()
            if self.mode == "adam":
                with torch.no_grad():
                    self._adam_head()
        self.grads = [p.grad for p in self.params]
        if any(g is None for g in self.grads):
            raise RuntimeError("a parameter got no gradient under capture")
        if self.mode == "adam":
            self.steps = [opt.state[p]["step"] for p in self.params]
        self.graph = graph

    def _chunked(self, tensors: List[torch.Tensor]):
        """Flat views of `tensors` in pieces of at most `adam_chunk`
        elements, in order, and the index of each piece's tensor."""
        if self.adam_chunk <= 0:
            return list(tensors), list(range(len(tensors)))
        views, owner = [], []
        for k, t in enumerate(tensors):
            f = t.view(-1)
            for i in range(0, f.numel(), self.adam_chunk):
                views.append(f[i:i + self.adam_chunk])
                owner.append(k)
        return views, owner

    def _adam_head(self):
        # torch.optim.adam._multi_tensor_adam, capturable=False, up to the
        # first op whose scalar depends on the step count. The foreach
        # lists hold views of the parameters' tensors cut into
        # `adam_chunk`-element pieces (see ADAM_CHUNK). The elementwise
        # arithmetic is the same kernel's, element by element.
        opt = self.optimizer
        g = opt.param_groups[0]
        beta1, beta2 = g["betas"]
        st = opt.state
        self.params_c, self.owner = self._chunked(self.params)
        self.exp_avgs_c = self._chunked(
            [st[p]["exp_avg"] for p in self.params])[0]
        exp_avg_sqs = self._chunked([st[p]["exp_avg_sq"] for p in self.params])[0]
        grads = self._chunked([p.grad for p in self.params])[0]
        torch._foreach_lerp_(self.exp_avgs_c, grads, 1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)
        self.denom = torch._foreach_sqrt(exp_avg_sqs)

    def _adam_tail(self):
        # The rest of _multi_tensor_adam: the step count and the
        # bias-corrected update, eager, same expressions, same order.
        g = self.optimizer.param_groups[0]
        beta1, beta2 = g["betas"]
        lr = g["lr"]
        torch._foreach_add_(self.steps, torch.tensor(1.0, device="cpu"),
                            alpha=1.0)
        bias_correction1 = [1 - beta1 ** s.item() for s in self.steps]
        bias_correction2 = [1 - beta2 ** s.item() for s in self.steps]
        step_size = [(lr / bc) * -1 for bc in bias_correction1]
        bias_correction2_sqrt = [bc ** 0.5 for bc in bias_correction2]
        # Each piece takes its parameter's scalar.
        torch._foreach_div_(self.denom,
                            [bias_correction2_sqrt[i] for i in self.owner])
        torch._foreach_add_(self.denom, g["eps"])
        torch._foreach_addcdiv_(self.params_c, self.exp_avgs_c, self.denom,
                                [step_size[i] for i in self.owner])

    def _replay(self, pool_x, pool_y, idx):
        torch.index_select(pool_x, 0, idx, out=self.x)
        torch.index_select(pool_y, 0, idx, out=self.y)
        self.graph.replay()
        self.replays += 1
        if self.mode == "adam":
            with torch.no_grad():
                self._adam_tail()
        else:
            # An eager step in between (ragged batch) re-pointed .grad.
            for p, gr in zip(self.params, self.grads):
                if p.grad is not gr:
                    p.grad = gr
            if self.spec.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.params,
                                               self.spec.grad_clip)
            self.optimizer.step()
        return _cloned(self.out)

    def _state_tensors(self) -> List[torch.Tensor]:
        st = self.optimizer.state
        out = list(self.params)
        for p in self.params:
            out.extend(v for _, v in sorted(st[p].items())
                       if torch.is_tensor(v))
        return out

    def _checked_step(self, pool_x, pool_y, idx):
        """The first replay, verified against the eager step from the
        same state. Keeps the eager result either way."""
        live = self._state_tensors()
        before = [t.detach().clone() for t in live]
        loss_e = self.spec.eager_step(pool_x[idx], pool_y[idx])
        eager = [t.detach().clone() for t in live]
        with torch.no_grad():
            for t, b in zip(live, before):
                t.copy_(b)
        loss_g = self._replay(pool_x, pool_y, idx)
        same = _outputs_equal(loss_e, loss_g) and all(
            torch.equal(t, e) for t, e in zip(live, eager))
        if not same:
            with torch.no_grad():
                for t, e in zip(live, eager):
                    t.copy_(e)
            self.disabled = True
            self.graph = None
            self.log("  WARN: BG_TRAIN_CUDA_GRAPH: the graphed train step "
                     "is NOT bitwise equal to the eager step on this "
                     "setup; using the eager step for the rest of the run.",
                     flush=True)
            return loss_e
        self.log(f"  NOTE: BG_TRAIN_CUDA_GRAPH: train step captured as a "
                 f"CUDA graph (mode {self.mode}, batch {self.batch_size}); "
                 f"first replay bitwise equal to the eager step.", flush=True)
        return loss_g


# loss_fn results: a tensor or a tuple of tensors.

def _detached(out):
    return tuple(t.detach() for t in out) if isinstance(out, tuple) else out.detach()


def _cloned(out):
    return tuple(t.clone() for t in out) if isinstance(out, tuple) else out.clone()


def _outputs_equal(a, b) -> bool:
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)
                and all(torch.equal(x, y) for x, y in zip(a, b)))
    return torch.equal(a, b)
