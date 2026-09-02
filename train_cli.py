"""
train_cli.py -- Shared helpers for train_online.py and train_batch.py.

Keeps the per-script files small by factoring out network/mode
construction, the resume path, and a vs-Random eval helper.
"""

import argparse
import os
import random

import torch
from agents import RandomAgent
from backgammon_engine import BoardState, WHITE, get_legal_plays, switch_turn
from encoding import CubefulEncoder, CubefulTakeEncoder, MatchplayEncoder, MatchplayCube1HEncoder
from model import TDNetwork, suggested_depth_expand_eps
from modes import CubefulMoneyMode, CubelessMoneyMode, DMPMode, MatchplayMode


def parse_hidden_sizes(s: str):
    """Parse '80,40' → [80, 40]."""
    return [int(x) for x in s.split(",") if x]


def resolve_save_path(save: str, game_mode: str, hidden_sizes) -> str:
    """Normalize --save so both `path/to/file.pt` and `path/to/dir/`
    work, and so missing parent directories are created rather than
    causing torch.save to silently fail.

    If `save` is a directory (existing or trailing-separator), generate
    a deterministic filename inside it from the game mode and hidden
    sizes, e.g. `models/dmp_512_512_256.pt`.
    """
    if save.endswith(("/", "\\")) or os.path.isdir(save):
        hidden_str = "_".join(str(h) for h in hidden_sizes)
        save = os.path.join(save, f"{game_mode}_{hidden_str}.pt")
    parent = os.path.dirname(save)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return save


def build_mode(name: str, jacoby: bool = True,
               take_decision_state: bool = False,
               match_length: int = 2,
               score_curriculum: float = 0.0,
               oracle_spec: str = None,
               oracle_move_eps: float = None,
               oracle_plies: int = 0,
               oracle_act_prob: float = 0.0,
               td_move_eps: float = 0.0,
               oracle_1ply_eps: float = 0.0,
               oracle_act_only: bool = False):
    if name == "dmp":
        return DMPMode()
    if name == "cubeless-money":
        return CubelessMoneyMode()
    if name == "cubeful-money":
        return CubefulMoneyMode(
            jacoby=jacoby, take_decision_state=take_decision_state,
        )
    if name == "matchplay":
        return MatchplayMode(match_length=match_length,
                             score_curriculum=score_curriculum,
                             oracle_spec=oracle_spec,
                             oracle_move_eps=oracle_move_eps,
                             oracle_plies=oracle_plies,
                             oracle_act_prob=oracle_act_prob,
                             td_move_eps=td_move_eps,
                             oracle_1ply_eps=oracle_1ply_eps,
                             oracle_act_only=oracle_act_only)
    raise ValueError(f"Unknown game mode: {name}")


def apply_cubeful_money_upgrades(args) -> None:
    """In-place cubeful-money auto-upgrades on parsed CLI args: the
    encoder must include the cube one-hot and the output must be equity.
    Overrides silently if the user left the defaults; errors if they set
    conflicting flags. No-op unless game_mode == "cubeful-money" (and not
    resuming, where the saved architecture wins). Shared by train_online
    and train_batch.
    """
    if args.game_mode == "cubeful-money" and not args.resume:
        if args.encoder == "perspective196":
            args.encoder = "cubeful_perspective196"
        elif not args.encoder.startswith("cubeful_"):
            raise ValueError(
                f"cubeful-money requires a cubeful_* encoder; "
                f"got --encoder {args.encoder!r}"
            )
        if args.output_mode == "probability":
            args.output_mode = "equity"
        elif args.output_mode != "equity":
            raise ValueError(
                f"cubeful-money requires --output-mode equity; "
                f"got {args.output_mode!r}"
            )


def build_network(args) -> TDNetwork:
    """Construct, resume, or expand a TDNetwork from CLI args.

    Mutually-exclusive sources (in priority order):
      --resume       — load and continue with the saved architecture
      --expand       — load and width-expand to --hidden
      --expand-depth — load and depth-expand by appending one hidden layer
      (none)         — fresh network with --hidden / --output-mode / --encoder

    --output-mode and --encoder are inherited from the source network
    when resuming or expanding (CLI overrides are ignored with a warning).
    """
    # Mutual exclusivity guard
    warm_start_eq = getattr(args, "warm_start_equity", None)
    warm_start_cf = getattr(args, "warm_start_cubeful", None)
    warm_start_cft = getattr(args, "warm_start_cubeful_take", None)
    warm_start_mp = getattr(args, "warm_start_matchplay", None)
    warm_start_mp1h = getattr(args, "warm_start_matchplay1h", None)
    warm_start_mpcf = getattr(args, "warm_start_matchplay_cubeful", None)
    warm_start_mpmet = getattr(args, "warm_start_matchplay_met", None)
    sources = [args.resume, args.expand, args.expand_depth,
               warm_start_eq, warm_start_cf, warm_start_cft,
               warm_start_mp, warm_start_mp1h, warm_start_mpcf,
               warm_start_mpmet]
    if sum(1 for s in sources if s) > 1:
        raise ValueError(
            "--resume, --expand, --expand-depth, --warm-start-equity, "
            "--warm-start-cubeful, --warm-start-cubeful-take, "
            "--warm-start-matchplay, --warm-start-matchplay1h and "
            "--warm-start-matchplay-cubeful are mutually exclusive"
        )

    if args.resume:
        net = TDNetwork.load(args.resume)
        print(
            f"Resumed from {args.resume}: hidden={net.hidden_sizes} "
            f"output_mode={net.output_mode} encoder={net.encoder_name}"
        )
        cli_hidden = parse_hidden_sizes(args.hidden) if args.hidden else None
        if cli_hidden and cli_hidden != net.hidden_sizes:
            print(
                f"  WARN: --hidden {args.hidden} ignored "
                f"(saved network has {net.hidden_sizes})"
            )
        if args.output_mode and args.output_mode != net.output_mode:
            print(
                f"  WARN: --output-mode {args.output_mode} ignored "
                f"(saved network has {net.output_mode})"
            )
        return net

    if args.expand:
        source = TDNetwork.load(args.expand)
        target_hidden = parse_hidden_sizes(args.hidden)
        net = TDNetwork.width_expand(source, target_hidden)
        print(
            f"Width-expanded from {args.expand}: "
            f"{source.hidden_sizes} -> {target_hidden} "
            f"(output_mode={net.output_mode}, encoder={net.encoder_name})"
        )
        return net

    if args.expand_depth:
        source = TDNetwork.load(args.expand_depth)
        depth_size = getattr(args, "expand_depth_size", None)
        depth_eps = getattr(args, "expand_depth_eps", None)
        if depth_eps is None:
            w = depth_size or source.hidden_sizes[-1]
            raise SystemExit(
                "ERROR: --expand-depth requires --expand-depth-eps. There is "
                "no default on purpose -- the safe value is net-specific "
                "(measurements in model.py, DEPTH_EXPAND_EPS_SCALE).\n"
                f"  --expand-depth-eps 0.0     exact copy, but symmetric\n"
                f"  --expand-depth-eps {suggested_depth_expand_eps(w):.6f}  suggested "
                f"starting point for width {w}\n"
                "MEASURE the equity deviation on real positions first."
            )
        net = TDNetwork.depth_expand(source, new_layer_size=depth_size,
                                     eps=depth_eps)
        used_eps = depth_eps
        print(
            f"Depth-expanded from {args.expand_depth}: "
            f"{source.hidden_sizes} -> {net.hidden_sizes} "
            f"(output_mode={net.output_mode}, encoder={net.encoder_name}, "
            f"eps={used_eps:g})"
        )
        return net

    if warm_start_eq:
        keep_output = getattr(args, "keep_output_layer", False)
        net = _warm_start_equity(warm_start_eq, reinit_output=not keep_output)
        return net

    if warm_start_cf:
        net = _warm_start_cubeful(warm_start_cf)
        return net

    if warm_start_cft:
        net = _warm_start_cubeful_take(warm_start_cft)
        return net

    if warm_start_mp:
        net = _warm_start_matchplay(warm_start_mp)
        return net

    if warm_start_mp1h:
        net = _warm_start_matchplay1h(warm_start_mp1h)
        return net

    if warm_start_mpcf:
        if getattr(args, "met_features", False):
            from encoding import MatchplayMETEncoder
            return _warm_start_matchplay_cubeful(
                warm_start_mpcf, encoder=MatchplayMETEncoder(),
            )
        return _warm_start_matchplay_cubeful(warm_start_mpcf)

    if warm_start_mpmet:
        net = _warm_start_matchplay_met(warm_start_mpmet)
        return net

    return TDNetwork(
        hidden_sizes=parse_hidden_sizes(args.hidden),
        output_mode=args.output_mode,
        encoder_name=args.encoder,
    )


def _warm_start_equity(path: str, reinit_output: bool = True) -> TDNetwork:
    """Load a probability-output model and convert to equity output.

    Copies all hidden layer weights. If reinit_output=True, re-initializes
    the output layer; if False, copies the output layer weights as-is.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    old_output = checkpoint.get("output_mode", "probability")
    if old_output != "probability":
        raise ValueError(
            f"--warm-start-equity expects a probability-output model, "
            f"got output_mode='{old_output}'. Use --resume for equity models."
        )
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 196)
    old_activation = checkpoint.get("activation", "relu")
    old_encoder = checkpoint.get("encoder_name", "perspective196")

    net = TDNetwork(
        hidden_sizes=old_hidden,
        input_size=old_input,
        activation=old_activation,
        output_mode="equity",
        encoder_name=old_encoder,
    )

    old_state = checkpoint.get("state_dict", checkpoint)
    new_state = net.state_dict()

    for key in new_state:
        if "output" in key and reinit_output:
            pass  # keep random init
        elif key in old_state:
            new_state[key] = old_state[key]

    net.load_state_dict(new_state)
    out_msg = "re-initialized" if reinit_output else "copied from source"
    print(f"Warm-started equity from: {path}")
    print(f"  Output: probability -> equity (output layer {out_msg})")
    print(f"  Hidden: {old_hidden}, input: {old_input}, encoder: {old_encoder}")
    return net


def _warm_start_cubeful(path: str) -> TDNetwork:
    """Load a 196-input model and extend to cubeful (200-input).

    Copies the first 196 columns of the input layer from the pretrained
    model. The 4 new columns (cube features) get Kaiming initialization.
    Output mode is set to equity.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 196)
    old_activation = checkpoint.get("activation", "relu")

    encoder = CubefulEncoder()
    new_input = encoder.num_features  # 200 (196 + CUBE_FEATURES)

    net = TDNetwork(
        hidden_sizes=old_hidden,
        input_size=new_input,
        activation=old_activation,
        output_mode="equity",
        encoder_name=encoder.name,
    )

    old_state = checkpoint.get("state_dict", checkpoint)
    new_state = net.state_dict()

    for key in new_state:
        if key == "hidden_layers.0.weight":
            old_weight = old_state[key]
            new_state[key][:, :old_input] = old_weight
        elif key in old_state:
            new_state[key] = old_state[key]

    net.load_state_dict(new_state)
    print(f"Warm-started cubeful from: {path}")
    print(f"  Input: {old_input} -> {new_input}, output -> equity")
    print(f"  Hidden: {old_hidden}, encoder: {encoder.name}")
    return net


def _warm_start_cubeful_take(path: str) -> TDNetwork:
    """Load a cubeful model (200-input) and extend to cubeful_take (201-input).

    Copies the first 200 columns of the input layer from the pretrained
    cubeful model; the new is_cube_take_decision column is initialized
    to zero. With zero weights for the new bit, the warm-started model
    is bit-identical to the source on every state the source could
    represent (is_cube_take_decision is only ever set on the new
    take_decision state, which the source never sees), so training
    starts from a known-good policy and the new state's value is
    purely learned during fine-tuning.

    Output mode stays equity. Accepts either a CubefulEncoder source
    (encoder='cubeful_<base>') or, for safety, errors on non-cubeful
    sources since the input layout would not align.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 200)
    old_activation = checkpoint.get("activation", "relu")
    old_encoder = checkpoint.get("encoder_name", "cubeful_perspective196")
    if not old_encoder.startswith("cubeful_"):
        raise ValueError(
            f"--warm-start-cubeful-take expects a cubeful source model "
            f"(encoder='cubeful_<base>'); got encoder={old_encoder!r}. "
            f"For a cubeless source, use --warm-start-cubeful first."
        )

    encoder = CubefulTakeEncoder()
    new_input = encoder.num_features  # 201
    if new_input != old_input + 1:
        raise ValueError(
            f"--warm-start-cubeful-take expects source input_size+1 == "
            f"new input_size; got {old_input}+1 != {new_input}"
        )

    net = TDNetwork(
        hidden_sizes=old_hidden,
        input_size=new_input,
        activation=old_activation,
        output_mode="equity",
        encoder_name=encoder.name,
    )

    old_state = checkpoint.get("state_dict", checkpoint)
    new_state = net.state_dict()

    for key in new_state:
        if key == "hidden_layers.0.weight":
            old_weight = old_state[key]
            # Zero-init the new column (is_cube_take_decision bit).
            new_state[key].zero_()
            new_state[key][:, :old_input] = old_weight
        elif key in old_state:
            new_state[key] = old_state[key]

    net.load_state_dict(new_state)
    print(f"Warm-started cubeful_take from: {path}")
    print(f"  Input: {old_input} -> {new_input} "
          f"(new is_cube_take_decision column zero-initialized)")
    print(f"  Hidden: {old_hidden}, encoder: {encoder.name}")
    return net


def _warm_start_matchplay(path: str) -> TDNetwork:
    """Load a 196-input model (DMP or cubeless-money) and extend it
    to the matchplay encoder layout (204 inputs).

    Equity sources (cubeless money): the output layer is copied as-is
    and reinterpreted through the probability head’s sigmoid, giving
    sigmoid(equity logit) — monotone in winning-ness and roughly
    calibrated (equity 0 → P 0.5). The hidden layers carry
    gammon-discriminating features that a DMP source lacks.

    Copies the first 196 columns of the input layer from the source;
    the 8 new match-feature columns (cube one-hot, is_cube_action,
    cube_value, away scores, crawford) are zero-initialized, so the
    warm-started model initially predicts the source's P(win game)
    regardless of cube or score — exactly correct at 1-away/1-away
    (DMP) and a sensible monotone starting point elsewhere. Score and
    cube sensitivity is then purely learned during matchplay training.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 196)
    old_activation = checkpoint.get("activation", "relu")
    old_encoder = checkpoint.get("encoder_name", "perspective196")
    old_output = checkpoint.get("output_mode", "probability")
    if old_encoder != "perspective196":
        raise ValueError(
            f"--warm-start-matchplay expects a perspective196 source "
            f"(e.g. a DMP model); got encoder={old_encoder!r}."
        )
    if old_output not in ("probability", "equity"):
        raise ValueError(
            f"--warm-start-matchplay expects a probability- or "
            f"equity-output source; got output_mode={old_output!r}."
        )

    encoder = MatchplayEncoder()
    new_input = encoder.num_features  # 204
    if new_input != old_input + 8:
        raise ValueError(
            f"--warm-start-matchplay expects source input_size+8 == "
            f"new input_size; got {old_input}+8 != {new_input}"
        )

    net = TDNetwork(
        hidden_sizes=old_hidden,
        input_size=new_input,
        activation=old_activation,
        output_mode="probability",
        encoder_name=encoder.name,
    )

    old_state = checkpoint.get("state_dict", checkpoint)
    new_state = net.state_dict()

    for key in new_state:
        if key == "hidden_layers.0.weight":
            old_weight = old_state[key]
            # Zero-init the 8 new match-feature columns.
            new_state[key].zero_()
            new_state[key][:, :old_input] = old_weight
        elif key in old_state:
            new_state[key] = old_state[key]

    net.load_state_dict(new_state)
    print(f"Warm-started matchplay from: {path}")
    print(f"  Input: {old_input} -> {new_input} "
          f"(8 new match-feature columns zero-initialized)")
    print(f"  Hidden: {old_hidden}, encoder: {encoder.name}")
    return net


def _warm_start_matchplay1h(path: str) -> TDNetwork:
    """Load a matchplay (204-input, raw cube scalar) net and convert
    it to the matchplay1h layout (207 inputs, cube value one-hot).

    Function-preserving: the old cube_value column contributed
    w * cube_value to each first-layer unit; setting the one-hot
    column for cube level 2^k to w * 2^k reproduces the identical
    activation at every reachable cube, so the warm-started net plays
    exactly like the source until training decouples the levels.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 196)
    old_activation = checkpoint.get("activation", "relu")
    old_encoder = checkpoint.get("encoder_name", "perspective196")
    old_output = checkpoint.get("output_mode", "probability")
    if old_encoder != "matchplay_perspective196":
        raise ValueError(
            f"--warm-start-matchplay1h expects a matchplay_perspective196 "
            f"source; got encoder={old_encoder!r}."
        )
    if old_output != "probability":
        raise ValueError(
            f"--warm-start-matchplay1h expects probability output; "
            f"got {old_output!r}."
        )
    encoder = MatchplayCube1HEncoder()
    n_base = encoder._base.num_features  # 196
    new_input = encoder.num_features     # 207
    assert old_input == n_base + 8 and new_input == n_base + 11

    net = TDNetwork(
        hidden_sizes=old_hidden,
        input_size=new_input,
        activation=old_activation,
        output_mode="probability",
        encoder_name=encoder.name,
    )
    old_state = checkpoint.get("state_dict", checkpoint)
    new_state = net.state_dict()
    for key in new_state:
        if key == "hidden_layers.0.weight":
            w_old = old_state[key]
            w_new = new_state[key]
            w_new.zero_()
            # base board features + [persp 3-hot, is_cube_action]
            w_new[:, :n_base + 4] = w_old[:, :n_base + 4]
            # cube one-hot: level 2^k gets old scalar column * 2^k
            w_cv = w_old[:, n_base + 4]
            for k in range(4):
                w_new[:, n_base + 4 + k] = w_cv * float(2 ** k)
            # away, away, crawford shift by 3
            w_new[:, n_base + 8:n_base + 11] = w_old[:, n_base + 5:n_base + 8]
        elif key in old_state:
            new_state[key] = old_state[key]
    net.load_state_dict(new_state)
    print(f"Warm-started matchplay1h from: {path}")
    print(f"  Input: {old_input} -> {new_input} (cube scalar -> one-hot, "
          f"function-preserving w*2^k init)")
    print(f"  Hidden: {old_hidden}, encoder: {encoder.name}")
    return net


def _warm_start_matchplay_cubeful(path: str, encoder=None) -> TDNetwork:
    """Load a CUBEFUL MONEY net (200 inputs, equity head) and extend it
    to a matchplay layout.

    Why this source rather than DMP: the cubeful money encoder's four
    appended columns — [cube 3-hot, is_cube_action] — are the same four
    columns, in the same order, that the matchplay encoders append
    first, so they transfer bit-for-bit (asserted in
    tests/test_warm_start_cubeful.py). A DMP source has no cube columns
    at all, and is trained on a game where gammons are worthless — a
    poor prior for 3-point play where gammon pricing dominates.

    Everything the source cannot know (cube LEVEL, away scores,
    Crawford, and the MET block if present) is zero-initialised, so the
    warm-started net initially predicts the source's cube-context-aware
    value regardless of score. Zero-init is safe for INPUT columns:
    unlike the width_expand dead-unit bug, the gradient w.r.t. an input
    weight is (input value x upstream grad), which is non-zero as soon
    as the feature is.

    The equity head is reinterpreted through the probability head's
    sigmoid, exactly as _warm_start_matchplay does for cubeless-money
    sources: sigmoid(equity) is monotone in winning-ness and sends
    equity 0 to P 0.5.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 200)
    old_activation = checkpoint.get("activation", "relu")
    old_encoder = checkpoint.get("encoder_name", "cubeful_perspective196")
    old_output = checkpoint.get("output_mode", "equity")

    if old_encoder != "cubeful_perspective196":
        raise ValueError(
            f"--warm-start-matchplay-cubeful expects a "
            f"cubeful_perspective196 source (e.g. "
            f"best_models/cubeful_money_512_512_256_256.pt); got "
            f"encoder={old_encoder!r}."
        )
    if old_output not in ("equity", "probability"):
        raise ValueError(
            f"--warm-start-matchplay-cubeful expects an equity- or "
            f"probability-output source; got {old_output!r}."
        )

    if encoder is None:
        encoder = MatchplayCube1HEncoder()
    n_base = encoder._base.num_features          # 196
    n_shared = n_base + 4                        # base + [3-hot, is_cube_action]
    new_input = encoder.num_features
    if old_input != n_shared:
        raise ValueError(
            f"--warm-start-matchplay-cubeful expects a {n_shared}-input "
            f"source (196 board + 4 cube); got {old_input}."
        )
    if new_input < old_input:
        raise ValueError(
            f"target encoder {encoder.name} has {new_input} inputs, "
            f"fewer than the {old_input}-input source."
        )

    net = TDNetwork(
        hidden_sizes=old_hidden,
        input_size=new_input,
        activation=old_activation,
        output_mode="probability",
        encoder_name=encoder.name,
    )

    old_state = checkpoint.get("state_dict", checkpoint)
    new_state = net.state_dict()
    for key in new_state:
        if key == "hidden_layers.0.weight":
            new_state[key].zero_()
            new_state[key][:, :n_shared] = old_state[key][:, :n_shared]
        elif key in old_state:
            new_state[key] = old_state[key]
    net.load_state_dict(new_state)

    print(f"Warm-started matchplay from cubeful money: {path}")
    print(f"  Input: {old_input} -> {new_input} "
          f"({n_shared} copied incl. cube perspective, "
          f"{new_input - n_shared} zero-init)")
    print(f"  Output: {old_output} -> probability (via sigmoid)")
    print(f"  Hidden: {old_hidden}, encoder: {encoder.name}")
    return net


def _warm_start_matchplay_met(path: str) -> TDNetwork:
    """Extend a matchplay1h net (207) to the MET-feature layout (220).

    Function-preserving: the 13 new columns are zero-initialised, so
    the model is bit-identical to the source until training gives the
    MET block weight. That makes a Phase-2 champion the natural
    starting point for the feature arm, and any measured difference
    attributable to the features alone.
    """
    from encoding import MatchplayMETEncoder

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 207)
    old_activation = checkpoint.get("activation", "relu")
    old_encoder = checkpoint.get("encoder_name", "matchplay1h_perspective196")
    old_output = checkpoint.get("output_mode", "probability")

    if old_encoder != "matchplay1h_perspective196":
        raise ValueError(
            f"--warm-start-matchplay-met expects a "
            f"matchplay1h_perspective196 source; got {old_encoder!r}."
        )
    if old_output != "probability":
        raise ValueError(
            f"--warm-start-matchplay-met expects probability output; "
            f"got {old_output!r}."
        )

    encoder = MatchplayMETEncoder()
    new_input = encoder.num_features
    if new_input != old_input + 13:
        raise ValueError(
            f"expected source+13 == target; got {old_input}+13 != {new_input}"
        )

    net = TDNetwork(
        hidden_sizes=old_hidden,
        input_size=new_input,
        activation=old_activation,
        output_mode="probability",
        encoder_name=encoder.name,
    )
    old_state = checkpoint.get("state_dict", checkpoint)
    new_state = net.state_dict()
    for key in new_state:
        if key == "hidden_layers.0.weight":
            new_state[key].zero_()
            new_state[key][:, :old_input] = old_state[key]
        elif key in old_state:
            new_state[key] = old_state[key]
    net.load_state_dict(new_state)
    print(f"Warm-started matchplay MET features from: {path}")
    print(f"  Input: {old_input} -> {new_input} "
          f"(13 MET columns zero-init, function-preserving)")
    print(f"  Hidden: {old_hidden}, encoder: {encoder.name}")
    return net


def eval_vs_random(agent, n_games: int = 200, seed: int = 99) -> float:
    """Play `n_games` of `agent` (WHITE) vs RandomAgent. Returns win rate."""
    rng = random.Random(seed)
    rnd = RandomAgent()
    wins = 0
    for _ in range(n_games):
        s = BoardState.initial()
        while not s.is_game_over():
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            plays = get_legal_plays(s, (d1, d2))
            if plays:
                actor = agent if s.turn == WHITE else rnd
                _, s = actor.choose_checker_action(s, (d1, d2), plays)
            else:
                s = switch_turn(s)
        if s.winner() == WHITE:
            wins += 1
    return wins / n_games


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by train_online.py and train_batch.py."""
    g_arch = parser.add_argument_group("network / game")
    g_arch.add_argument("--game-mode",
                        choices=["dmp", "cubeless-money", "cubeful-money",
                                 "matchplay"],
                        required=True,
                        help="Game type: dmp (single game DMP), "
                             "cubeless-money (money play, no cube), "
                             "cubeful-money (money play with doubling cube), "
                             "or matchplay (N-point match; see "
                             "--match-length).")
    g_arch.add_argument("--match-length", type=int, default=2, metavar="N",
                        help="Matchplay only: match length in points "
                             "(episode = one full N-point match). "
                             "Default: 2.")
    g_arch.add_argument("--jacoby", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Jacoby rule for cubeful-money: gammons/bgs "
                             "don't count while the cube is centered. "
                             "Ignored for non-cubeful modes. Default: on.")
    g_arch.add_argument("--output-mode", choices=["probability", "equity"],
                        default="probability",
                        help="Network output: 'probability' (sigmoid P(win)) or "
                             "'equity' (linear ±3). Ignored when --resume is set.")
    g_arch.add_argument("--hidden", default="40",
                        help="Comma-separated hidden layer sizes, e.g. '40' or '80,40'. "
                             "Ignored when --resume is set.")
    g_arch.add_argument("--encoder", default="perspective196")

    g_train = parser.add_argument_group("training")
    g_train.add_argument("--num-episodes", type=int, required=True,
                         help="Number of self-play episodes to train for.")
    g_train.add_argument("--seed", type=int, default=None,
                         help="Seed for the game RNG (dice rolls, opening, shuffling).")
    g_train.add_argument("--torch-seed", type=int, default=None,
                         help="Seed for torch network init (for reproducibility).")
    g_train.add_argument("--device", default="cpu")

    g_io = parser.add_argument_group("persistence")
    g_io.add_argument("--resume", default=None,
                      help="Path to a saved TDNetwork (.pt) to continue training from. "
                           "Architectural flags are ignored when this is set.")
    g_io.add_argument("--expand", default=None, metavar="PATH",
                      help="Width-expand a smaller saved network to --hidden. "
                           "The new neurons start at zero so the function is "
                           "preserved at init.")
    g_io.add_argument("--expand-depth", default=None, metavar="PATH",
                      help="Depth-expand a saved network by appending one hidden "
                           "layer initialized as near-identity. "
                           "Function is preserved at init.")
    g_io.add_argument("--expand-depth-size", type=int, default=None, metavar="N",
                      help="Size of the new layer when using --expand-depth. "
                           "Defaults to the last hidden layer's size, and must "
                           "EQUAL it (square identity init). Grow other layers "
                           "with --expand instead.")
    g_io.add_argument("--expand-depth-eps", type=float, default=None,
                      metavar="EPS",
                      help="REQUIRED with --expand-depth. Symmetry-breaking "
                           "noise on the near-identity init. No default -- "
                           "0.0 is exact but symmetric, injected deviation "
                           "grows as eps*sqrt(width), and the equity cost is "
                           "net-specific. Measure it (see model.py, "
                           "DEPTH_EXPAND_EPS_SCALE).")
    g_io.add_argument("--warm-start-equity", default=None, metavar="PATH",
                      help="Initialize an equity-output model from a "
                           "probability-output (DMP) model. Copies hidden "
                           "layers, re-initializes output head (unless "
                           "--keep-output-layer is set).")
    g_io.add_argument("--keep-output-layer", action="store_true",
                      help="With --warm-start-equity: copy the output layer "
                           "weights instead of re-initializing them.")
    g_io.add_argument("--warm-start-cubeful", default=None, metavar="PATH",
                      help="Initialize a cubeful money model from any "
                           "196-input model (DMP or cubeless money). Extends "
                           "input 196->200 (adds cube features), sets output "
                           "to equity.")
    g_io.add_argument("--warm-start-cubeful-take", default=None, metavar="PATH",
                      help="Initialize a cubeful_take_* model from an "
                           "existing cubeful_* model. Extends input "
                           "200->201 (adds the is_cube_take_decision "
                           "bit, zero-initialized so the new model is "
                           "bit-identical to the source on chequer and "
                           "cube-offer states). Pair with "
                           "--take-decision-state in train_batch.")
    g_io.add_argument("--warm-start-matchplay1h", default=None, metavar="PATH",
                      help="Initialize a matchplay1h model (cube value "
                           "one-hot, 207 inputs) from an existing "
                           "matchplay_perspective196 model. "
                           "Function-preserving (w*2^k column init).")
    g_io.add_argument("--warm-start-matchplay-met", default=None,
                      metavar="PATH",
                      help="Extend a matchplay1h model (207 inputs) to the "
                           "MET-feature layout (220 inputs: +6 no-double, "
                           "+6 double-take, +1 double-drop outcome MWCs). "
                           "Function-preserving (13 zero-init columns).")
    g_io.add_argument("--met-features", action="store_true",
                      help="With --warm-start-matchplay-cubeful, target the "
                           "220-input MET-feature encoder instead of the "
                           "207-input matchplay1h encoder.")
    g_io.add_argument("--warm-start-matchplay-cubeful", default=None,
                      metavar="PATH",
                      help="Initialize a matchplay1h model (207 inputs) "
                           "from a CUBEFUL MONEY model "
                           "(cubeful_perspective196, 200 inputs, equity "
                           "head). The 4 cube-context columns transfer "
                           "bit-for-bit; cube level, away scores and "
                           "crawford are zero-initialized. Preferred over "
                           "--warm-start-matchplay (DMP), which starts "
                           "from a gammon-blind net with no cube columns.")
    g_io.add_argument("--warm-start-matchplay", default=None, metavar="PATH",
                      help="Initialize a matchplay model from a 196-input "
                           "probability model (e.g. a DMP net). Extends "
                           "input 196->204 (adds cube + away-score + "
                           "crawford features, zero-initialized so the "
                           "model starts as the source's P(win game) at "
                           "every score).")
    g_io.add_argument("--save", default=None,
                      help="Path to save the trained network at the end of the run.")

    g_log = parser.add_argument_group("logging / eval")
    g_log.add_argument("--eval-vs-random", type=int, default=0, metavar="N",
                       help="If N>0, play N games vs RandomAgent at the end and "
                            "report win rate.")
