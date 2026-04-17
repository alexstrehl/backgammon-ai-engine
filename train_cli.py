"""
train_cli.py -- Shared helpers for train_online.py and train_batch.py.

Keeps the per-script files small by factoring out network/mode
construction, the resume path, and a vs-Random eval helper.
"""

import argparse
import random

import torch
from agents import RandomAgent
from backgammon_engine import BoardState, WHITE, get_legal_plays, switch_turn
from encoding import CubefulEncoder
from model import TDNetwork
from modes import CubefulMoneyMode, CubelessMoneyMode, DMPMode


def parse_hidden_sizes(s: str):
    """Parse '80,40' → [80, 40]."""
    return [int(x) for x in s.split(",") if x]


def build_mode(name: str, jacoby: bool = True):
    if name == "dmp":
        return DMPMode()
    if name == "cubeless-money":
        return CubelessMoneyMode()
    if name == "cubeful-money":
        return CubefulMoneyMode(jacoby=jacoby)
    raise ValueError(f"Unknown game mode: {name}")


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
        net = TDNetwork.depth_expand(source, new_layer_size=depth_size)
        print(
            f"Depth-expanded from {args.expand_depth}: "
            f"{source.hidden_sizes} -> {net.hidden_sizes} "
            f"(output_mode={net.output_mode}, encoder={net.encoder_name})"
        )
        return net

    if warm_start_eq:
        keep_output = getattr(args, "keep_output_layer", False)
        net = _warm_start_equity(warm_start_eq, reinit_output=not keep_output)
        return net

    if warm_start_cf:
        net = _warm_start_cubeful(warm_start_cf)
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
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
    """Load a 196-input model and extend to cubeful (199-input).

    Copies the first 196 columns of the input layer from the pretrained
    model. The 3 new columns (cube features) get Kaiming initialization.
    Output mode is set to equity.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old_hidden = checkpoint["hidden_sizes"]
    old_input = checkpoint.get("input_size", 196)
    old_activation = checkpoint.get("activation", "relu")

    encoder = CubefulEncoder()
    new_input = encoder.num_features  # 199

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
                           "Defaults to the last hidden layer's size. Must be "
                           "<= the last hidden size (for identity init).")
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
                           "input 196->199 (adds cube features), sets output "
                           "to equity.")
    g_io.add_argument("--save", default=None,
                      help="Path to save the trained network at the end of the run.")

    g_log = parser.add_argument_group("logging / eval")
    g_log.add_argument("--eval-vs-random", type=int, default=0, metavar="N",
                       help="If N>0, play N games vs RandomAgent at the end and "
                            "report win rate.")
