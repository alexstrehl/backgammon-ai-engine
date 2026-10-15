"""
train_cli.py -- Shared helpers for train_online.py and train_batch.py:
common CLI args, network/mode construction (fresh, resume, expand,
warm-start), and a vs-Random eval.
"""

import argparse
import os
import random

import torch
from agents import RandomAgent
from backgammon_engine import BoardState, WHITE, get_legal_plays, switch_turn
from encoding import CubefulEncoder
from model import TDNetwork, suggested_depth_expand_eps
from modes import CubefulMoneyMode, CubelessMoneyMode, DMPMode


DEFAULT_HIDDEN = "40"
DEFAULT_OUTPUT_MODE = "probability"


def parse_hidden_sizes(s: str):
    """Parse '80,40' → [80, 40]."""
    return [int(x) for x in s.split(",") if x]


def resolve_save_path(save: str, game_mode: str, hidden_sizes) -> str:
    """Normalize --save: a directory (existing or trailing-separator)
    gets a filename from the game mode and hidden sizes, e.g.
    `models/dmp_512_512_256.pt`. Missing parent directories are
    created.
    """
    if save.endswith(("/", "\\")) or os.path.isdir(save):
        hidden_str = "_".join(str(h) for h in hidden_sizes)
        save = os.path.join(save, f"{game_mode}_{hidden_str}.pt")
    parent = os.path.dirname(save)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return save


def build_mode(name: str, jacoby: bool = True):
    if name == "dmp":
        return DMPMode()
    if name == "cubeless-money":
        return CubelessMoneyMode()
    if name == "cubeful-money":
        return CubefulMoneyMode(jacoby=jacoby)
    raise ValueError(f"Unknown game mode: {name}")


def apply_cubeful_money_upgrades(args) -> None:
    """Cubeful-money needs a cubeful_* encoder and equity output.
    Upgrades the defaults in place. Errors on conflicting flags. No-op
    unless game_mode == "cubeful-money" and not resuming.
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
    """Construct, resume, expand, or warm-start a TDNetwork from CLI args.

    Mutually-exclusive sources (in priority order):
      --resume             continue with the saved architecture
      --expand             width-expand to --hidden
      --expand-depth       append one hidden layer
      --warm-start-equity  probability -> equity head
      --warm-start-cubeful 196 -> 200 inputs, equity head
      (none)               fresh --hidden / --output-mode / --encoder

    Resuming or expanding inherits --output-mode and --encoder from
    the source (CLI overrides are ignored with a warning).
    """
    warm_start_eq = getattr(args, "warm_start_equity", None)
    warm_start_cf = getattr(args, "warm_start_cubeful", None)
    sources = [args.resume, args.expand, args.expand_depth,
               warm_start_eq, warm_start_cf]
    if sum(1 for s in sources if s) > 1:
        raise ValueError(
            "--resume, --expand, --expand-depth, --warm-start-equity, "
            "and --warm-start-cubeful are mutually exclusive"
        )

    if args.resume:
        net = TDNetwork.load(args.resume)
        print(
            f"Resumed from {args.resume}: hidden={net.hidden_sizes} "
            f"output_mode={net.output_mode} encoder={net.encoder_name}"
        )
        # Warn only on values the user set: the parser defaults are not None.
        cli_hidden = (parse_hidden_sizes(args.hidden)
                      if args.hidden and args.hidden != DEFAULT_HIDDEN else None)
        if cli_hidden and cli_hidden != net.hidden_sizes:
            print(
                f"  WARN: --hidden {args.hidden} ignored "
                f"(saved network has {net.hidden_sizes})"
            )
        if (args.output_mode and args.output_mode != DEFAULT_OUTPUT_MODE
                and args.output_mode != net.output_mode):
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
                "(DEPTH_EXPAND_EPS_SCALE in model.py is a starting point).\n"
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
        net = _warm_start_cubeful(warm_start_cf, args.encoder)
        return net

    return TDNetwork(
        hidden_sizes=parse_hidden_sizes(args.hidden),
        output_mode=args.output_mode,
        encoder_name=args.encoder,
    )


def _warm_start_equity(path: str, reinit_output: bool = True) -> TDNetwork:
    """Load a probability-output model as an equity-output net. Hidden
    layers are copied. The output layer is re-initialized unless
    reinit_output=False.
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


def _warm_start_cubeful(path: str,
                        encoder_name: str = "cubeful_perspective196",
                        ) -> TDNetwork:
    """Load a base-input model as a cubeful equity net of the encoder's
    width (200 for cubeful_perspective196). The base input columns are
    copied. The new cube columns keep their fresh init. A non-cubeful
    `encoder_name` (the CLI default) selects cubeful_perspective196.
    """
    from encoding import get_encoder

    if not encoder_name.startswith("cubeful_"):
        encoder_name = "cubeful_perspective196"
    encoder = get_encoder(encoder_name)
    if not isinstance(encoder, CubefulEncoder):
        raise ValueError(
            f"--warm-start-cubeful requires a cubeful encoder; got "
            f"--encoder {encoder_name!r}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 196)
    old_activation = checkpoint.get("activation", "relu")

    n_base = encoder._base.num_features
    if old_input != n_base:
        raise ValueError(
            f"--warm-start-cubeful expects a {n_base}-input base model "
            f"(encoder {encoder.base_name}); {path} has {old_input} "
            f"inputs. A cubeful checkpoint resumes with --resume."
        )
    new_input = encoder.num_features  # base + encoder.cube_features

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
    print(f"  Input: {old_input} -> {new_input} "
          f"(+{encoder.cube_features} cube columns), output -> equity")
    print(f"  Hidden: {old_hidden}, encoder: {encoder.name}")
    return net


def eval_vs_random(agent, n_games: int = 200, seed: int = 99) -> float:
    """Win rate of `agent` (WHITE) over `n_games` vs RandomAgent."""
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
    """Flags shared by train_online.py and train_batch.py."""
    g_arch = parser.add_argument_group("network / game")
    g_arch.add_argument("--game-mode",
                        choices=["dmp", "cubeless-money", "cubeful-money"],
                        required=True,
                        help="Game type: dmp (single game DMP), "
                             "cubeless-money (money play, no cube), or "
                             "cubeful-money (money play with doubling cube).")
    g_arch.add_argument("--jacoby", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Jacoby rule for cubeful-money: gammons/bgs "
                             "don't count while the cube is centered. "
                             "Ignored for non-cubeful modes. Default: on.")
    g_arch.add_argument("--output-mode", choices=["probability", "equity"],
                        default=DEFAULT_OUTPUT_MODE,
                        help="Network output: 'probability' (sigmoid P(win)) or "
                             "'equity' (linear ±3). Ignored when --resume is set.")
    g_arch.add_argument("--hidden", default=DEFAULT_HIDDEN,
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
                           "the input to the --encoder's width (200 for "
                           "cubeful_perspective196), sets output to equity.")
    g_io.add_argument("--save", default=None,
                      help="Path to save the trained network at the end of the run.")

    g_log = parser.add_argument_group("logging / eval")
    g_log.add_argument("--eval-vs-random", type=int, default=0, metavar="N",
                       help="If N>0, play N games vs RandomAgent at the end and "
                            "report win rate.")
