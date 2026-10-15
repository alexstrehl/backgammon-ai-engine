"""The BGNN export's encoder name and the C API encoder table.

  - export_weights writes format 2 ("BGNV", version, 64-byte encoder
    name); read_header and nn_load (through bgc_open) read it back, and
    the weights payload is format 1's, byte for byte
  - a legacy format-1 file ("BGNN", no name) opens by its width (196 /
    200) with the kind and the outputs (bitwise) of its format-2
    twin; another width fails naming the legacy layout
  - an unknown encoder, an unknown base encoder, a width that is not the
    encoder's and an output mode the encoder has no kind for fail
    bgc_open / Engine() with a message naming the encoder; a newer BGNV
    version and a bad name field fail nn_load
  - the C table agrees with encoding.get_encoder (width, cube block) on
    every name the registry builds
  - a library built before format 2 is refused by load_lib with the
    rebuild command (it would reject every new export at the magic)
  - bench_forward's export cache never reuses an older-format entry

Backends as tests/test_c_api.py (built on demand, CAPI_BACKENDS).

Match play: the registry here also builds the matchplay encoders. The
C table knows two of them (matchplay1h64met_ and matchplay1h64met6_, kind
MATCH; the match core is tested in tests/test_c_match.py); the others
stay unknown (test_matchplay_encoders_in_table), and the prod nets'
exports open by their name, never by width (test_matchplay_export_opens).
"""
import ctypes
import os
import random
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_c_api import (P, backend, _playout_states, _random_ms,  # noqa: F401
                        FULL, PROD_CUBEFUL, PROD_PROB5, REPO)

import export_weights as EW  # noqa: E402
from encoding import ENCODERS, CubefulEncoder, get_encoder  # noqa: E402
from model import ProbNetwork, TDNetwork, load_model  # noqa: E402

BASE = "perspective196"
KINDS = {  # name -> (encoder, input size, output mode)
    "prob": (BASE, 196, "probability"),
    "equity": (BASE, 196, "equity"),
    "prob5": (BASE, 196, "prob5"),
    "cubeful": ("cubeful_" + BASE, 200, "equity"),
}


def _net(enc, n_in, mode, seed=0, hidden=(24, 12)):
    torch.manual_seed(seed)
    if mode == "prob5":
        net = ProbNetwork(hidden_sizes=list(hidden), input_size=n_in, encoder_name=enc)
    else:
        net = TDNetwork(hidden_sizes=list(hidden), encoder_name=enc, input_size=n_in,
                        output_mode=mode)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(3.0)
    return net.eval()


def _export(net, path):
    EW.export_live(net, str(path))
    return str(path)


def _payload(net):
    """Format 1's header fields from num_hidden on + the weights, built from
    the net alone (independent of export_weights' writer)."""
    if isinstance(net, ProbNetwork):
        layers = [m for m in net.trunk if isinstance(m, torch.nn.Linear)] + [net.head]
        mode = EW.OUTPUT_MODE_PROB5
    else:
        layers = list(net.hidden_layers) + [net.fc_output]
        mode = {"probability": 0, "equity": 1}[net.output_mode]
    out = struct.pack("<iiii", len(net.hidden_sizes), net.input_size,
                      EW.ACTIVATION_MAP[net.activation], mode)
    out += struct.pack(f"<{len(net.hidden_sizes)}i", *net.hidden_sizes)
    for L in layers:
        out += L.weight.detach().numpy().astype(np.float32).tobytes()
        out += L.bias.detach().numpy().astype(np.float32).tobytes()
    return out


def _legacy(net, path):
    """The format-1 file of `net` (no encoder name)."""
    with open(path, "wb") as f:
        f.write(b"BGNN" + _payload(net))
    return str(path)


def _open_error(P, backend, path):
    """bgc_open_ex's NULL and message on `path`."""
    lib = P.load_lib(backend)
    err = ctypes.create_string_buffer(1024)
    h = lib.bgc_open_ex(str(path).encode(), err, len(err))
    if h:
        lib.bgc_close(h)
    return h, err.value.decode()


def _values(e, states, cubeful, plies=(0,)):
    """Bytes of bgc_value and score_moves at `plies` over the states."""
    rng = random.Random(5)
    buf = []
    for s in states:
        ms = _random_ms(rng) if cubeful else None
        for ica in ((0, 1) if cubeful else (0,)):
            buf.append(np.float32(e.value(s, ms, ica)).tobytes())
        d = (rng.randint(1, 6), rng.randint(1, 6))
        for p in plies:
            buf.append(e.score_moves(s, d, ms, p)[0].tobytes())
    return b"".join(buf)


# ── header round trip ───────────────────────────────────────────────

@pytest.mark.parametrize("kind", list(KINDS))
def test_header_round_trip(P, backend, tmp_path, kind):
    enc, n_in, mode = KINDS[kind]
    net = _net(enc, n_in, mode)
    path = _export(net, tmp_path / "m.bin")
    raw = open(path, "rb").read()
    assert raw[:4] == b"BGNV" and struct.unpack("<i", raw[4:8]) == (EW.FORMAT_VERSION,) == (2,)
    assert raw[8:72] == enc.encode().ljust(64, b"\0")
    assert raw[72:] == _payload(net)              # format 1's fields and weights, unchanged
    h = EW.read_header(path)
    assert (h["format_version"], h["encoder_name"], h["input_size"], h["hidden_sizes"],
            h["data_offset"]) == (2, enc, n_in, [24, 12], 72 + 16 + 8)
    e = P.Engine(path, backend)
    want = {"prob": P.KIND_PROB, "equity": P.KIND_EQUITY, "prob5": P.KIND_PROB5,
            "cubeful": P.KIND_CUBEFUL}[kind]
    assert (e.kind, e.input_size, e.encoder, e.format_version) == (want, n_in, enc, 2)
    e.close()


def test_encoder_name_field():
    assert EW.encoder_name_field("x") == b"x" + b"\0" * 63
    assert len(EW.encoder_name_field("a" * 63)) == 64
    for bad in ("", None, "a" * 64, "has space", "tab\there", "née"):
        with pytest.raises(ValueError):
            EW.encoder_name_field(bad)
    with pytest.raises(ValueError, match="encoder_name"):
        EW.export_live(type("N", (), {"activation": "relu", "encoder_name": "",
                                      "output_mode": "equity", "hidden_layers": [],
                                      "fc_output": None, "hidden_sizes": []})(), os.devnull)


# ── legacy (format 1) files ─────────────────────────────────────────

@pytest.mark.parametrize("kind", list(KINDS))
def test_legacy_file_opens_by_width(P, backend, tmp_path, kind):
    """A format-1 file opens through the legacy path (encoder by width)
    with its format-2 twin's kind, encoder and outputs, bitwise."""
    enc, n_in, mode = KINDS[kind]
    net = _net(enc, n_in, mode, seed=3)
    new = _export(net, tmp_path / "new.bin")
    old = _legacy(net, tmp_path / "old.bin")
    assert open(old, "rb").read() == b"BGNN" + open(new, "rb").read()[72:]
    assert EW.read_header(old)["format_version"] == 1
    assert EW.read_header(old)["encoder_name"] is None
    e2, e1 = P.Engine(new, backend), P.Engine(old, backend)
    assert e1.format_version == 1 and e2.format_version == 2
    assert (e1.kind, e1.input_size, e1.encoder) == (e2.kind, e2.input_size, e2.encoder)
    assert e1.encoder == enc
    states = _playout_states(60, 11)
    cubeful = e1.kind == P.KIND_CUBEFUL
    assert _values(e1, states, cubeful) == _values(e2, states, cubeful)
    assert _values(e1, states[:4], cubeful, (1,)) == _values(e2, states[:4], cubeful, (1,))


@pytest.mark.parametrize("model", [PROD_CUBEFUL, PROD_PROB5], ids=["cubeful", "prob5"])
def test_legacy_prod_export_bitwise(P, backend, tmp_path, model):
    """The prod nets' format-1 files (a stale cache) give the outputs of
    their format-2 exports, bitwise."""
    if backend == "ref" and not FULL:
        pytest.skip("scalar ref: the property is backend-independent (CAPI_FULL=1 runs it)")
    net = load_model(model)
    e2 = P.Engine(model, backend)
    e1 = P.Engine(_legacy(net, tmp_path / "old.bin"), backend)
    assert (e1.format_version, e1.encoder, e1.kind, e1.input_size) == \
        (1, net.encoder_name, e2.kind, net.input_size)
    states = _playout_states(40, 13)
    cubeful = e1.kind == P.KIND_CUBEFUL
    assert _values(e1, states, cubeful) == _values(e2, states, cubeful)
    assert _values(e1, states[:2], cubeful, (1,)) == _values(e2, states[:2], cubeful, (1,))


def test_legacy_unknown_width_fails(P, backend, tmp_path, capfd):
    net = _net("cubeful_x", 198, "equity")
    path = _legacy(net, tmp_path / "old.bin")
    h, msg = _open_error(P, backend, path)
    assert not h and "legacy" in msg and "198" in msg, msg
    with pytest.raises(RuntimeError, match="legacy BGNN file"):
        P.Engine(path, backend)
    lib = P.load_lib(backend)
    assert not lib.bgc_open(path.encode())      # bgc_open: the message on stderr
    assert "198 inputs" in capfd.readouterr().err


# ── what bgc_open refuses ───────────────────────────────────────────

@pytest.mark.parametrize("enc,needles", [
    ("fancy196", ["unknown encoder 'fancy196'", "perspective196"]),
    ("cubeful_fancy196", ["encoder 'cubeful_fancy196'", "unknown base encoder 'fancy196'"]),
])
def test_unknown_encoder_fails(P, backend, tmp_path, enc, needles):
    """A checkpoint whose encoder_name the C table does not know: bgc_open
    fails naming it, from the .pt (Engine exports it) and from the .bin."""
    net = _net(enc, 196, "equity")
    pt = str(tmp_path / "m.pt")
    net.save(pt)
    assert load_model(pt).encoder_name == enc
    for src in (pt, net, _export(net, tmp_path / "m.bin")):
        with pytest.raises(RuntimeError) as ex:
            P.Engine(src, backend)
        for n in needles:
            assert n in str(ex.value), str(ex.value)
    h, msg = _open_error(P, backend, tmp_path / "m.bin")
    assert not h and all(n in msg for n in needles), msg
    assert P.encoder_info(enc, backend) is None


@pytest.mark.parametrize("enc,n_in", [("cubeful_perspective196", 199),
                                      ("cubeful_perspective196", 196),
                                      ("perspective196", 200),
                                      ("perspective196", 199)])
def test_width_mismatch_fails(P, backend, tmp_path, enc, n_in):
    """input_size != the encoder's width: no width-based guess, a loud
    failure naming the encoder and both widths."""
    path = _export(_net(enc, n_in, "equity"), tmp_path / "m.bin")
    want = P.encoder_info(enc, backend)["input_size"]
    with pytest.raises(RuntimeError) as ex:
        P.Engine(path, backend)
    msg = str(ex.value)
    assert f"encoder '{enc}' rows are {want} wide" in msg and f"the net has {n_in}" in msg, msg


@pytest.mark.parametrize("enc,n_in,mode", [("cubeful_perspective196", 200, "probability"),
                                           ("cubeful_perspective196", 200, "prob5")])
def test_kind_mismatch_fails(P, backend, tmp_path, enc, n_in, mode):
    """A cubeful encoder needs an equity output (CUBEFUL)."""
    path = _export(_net(enc, n_in, mode), tmp_path / "m.bin")
    with pytest.raises(RuntimeError, match=f"encoder '{enc}' \\(cubeful\\)"):
        P.Engine(path, backend)


def test_bad_headers_fail(P, backend, tmp_path, capfd):
    """nn_load: a newer BGNV version, a bad name field, a bad magic."""
    good = open(_export(_net(BASE, 196, "equity"), tmp_path / "g.bin"), "rb").read()
    cases = {
        "v3": good[:4] + struct.pack("<i", 3) + good[8:],
        "v1": good[:4] + struct.pack("<i", 1) + good[8:],
        "padding": good[:8] + b"perspective196\0x" + good[24:],
        "empty": good[:8] + b"\0" * 64 + good[72:],
        "space": good[:8] + b"perspective 96".ljust(64, b"\0") + good[72:],
        "unterminated": good[:8] + b"p" * 64 + good[72:],
        "magic": b"BGNX" + good[4:],
        "truncated": good[:40],
    }
    for name, raw in cases.items():
        p = tmp_path / f"{name}.bin"
        p.write_bytes(raw)
        with pytest.raises(RuntimeError, match="cannot load BGNN model"):
            P.Engine(str(p), backend)
        if name not in ("truncated", "magic"):
            with pytest.raises(ValueError):
                EW.read_header(str(p))
    err = capfd.readouterr().err
    assert "BGNV format version 3" in err and "bad encoder name field" in err
    assert "is not a BGNN/BGNV export (magic BGNX)" in err


# ── the table and the Python registry ───────────────────────────────

def _registry_names():
    for base in ENCODERS:
        yield base
        yield f"cubeful_{base}"


def test_encoder_table_matches_registry(P, backend):
    """Every name encoding.get_encoder builds is in the C table with the
    Python encoder's width and cube block; names
    get_encoder refuses are refused too."""
    for name in _registry_names():
        enc = get_encoder(name)
        info = P.encoder_info(name, backend)
        assert info is not None, name
        cubeful = isinstance(enc, CubefulEncoder)
        assert info["input_size"] == enc.num_features
        assert info["ncube"] == (enc.cube_features if cubeful else 0)
        assert info["n_base"] + info["ncube"] == info["input_size"]
    for name in ("", "cubeful_", "perspective",
                 "cubeful_cubeful_perspective196", "Perspective196", "perspective196 "):
        with pytest.raises((ValueError, KeyError)):
            get_encoder(name)
        assert P.encoder_info(name, backend) is None, name


def _matchplay_names():
    """Every matchplay encoder name this branch's registry builds over the
    base encoders."""
    pre = ("matchplay_", "matchplay1h_", "matchplay1h64_", "matchplay1h64met_",
           "matchplay1h64met4_", "matchplay1h64met5_", "matchplay1h64met6_",
           "matchplaymet_")
    for base in ENCODERS:
        for p in pre:
            try:
                get_encoder(p + base)
            except (ValueError, KeyError):
                continue
            yield p + base


C_MATCH = {"matchplay1h64met_perspective196": 13, "matchplay1h64met6_perspective196": 17}


def test_matchplay_encoders_in_table(P, backend):
    """Exactly matchplay1h64met_ and matchplay1h64met6_ over perspective196
    are in the C table (match block = the encoder's width past the base,
    the MET block's width); every other matchplay name the registry
    builds is not."""
    names = list(_matchplay_names())
    assert set(C_MATCH) <= set(names)
    for name in names:
        info = P.encoder_info(name, backend)
        if name not in C_MATCH:
            assert info is None, name
            continue
        enc = get_encoder(name)
        assert info["match"] and info["nmet"] == C_MATCH[name]
        assert info["input_size"] == enc.num_features
        assert info["n_base"] == enc._base.num_features
        assert info["ncube"] == enc.num_features - enc._base.num_features


@pytest.mark.parametrize("model,enc,n_in", [
    ("matchplay_1h64met_512_512_256_256.pt", "matchplay1h64met_perspective196", 350)],
    ids=["met"])
def test_matchplay_export_opens(P, backend, tmp_path, model, enc, n_in):
    """The matchplay prod nets export their encoder name (format 2) and
    open by it as KIND_MATCH; the legacy twin (no name) still fails
    naming its width (no width-based guess: these widths have no legacy
    layout), and an equity output on a match encoder fails naming it."""
    path = os.path.join(REPO, "best_models", model)
    net = load_model(path)
    assert (net.encoder_name, net.input_size) == (enc, n_in)
    b = _export(net, tmp_path / "m.bin")
    hd = EW.read_header(b)
    assert (hd["format_version"], hd["encoder_name"], hd["input_size"]) == (2, enc, n_in)
    for src in (path, b):
        e = P.Engine(src, backend)
        assert (e.kind, e.encoder, e.input_size, e.format_version) == (P.KIND_MATCH, enc,
                                                                       n_in, 2)
        e.close()
    h, msg = _open_error(P, backend, _legacy(net, tmp_path / "old.bin"))
    assert not h and "legacy" in msg and str(n_in) in msg, msg
    small = _net(enc, n_in, "equity")
    with pytest.raises(RuntimeError, match=f"encoder '{enc}' \\(match\\)"):
        P.Engine(_export(small, tmp_path / "eq.bin"), backend)
    with pytest.raises(RuntimeError, match=f"encoder '{enc}' rows are {n_in} wide"):
        P.Engine(_export(_net(enc, n_in + 1, "probability"), tmp_path / "w.bin"), backend)


# ── stale libraries and caches ──────────────────────────────────────

def test_load_lib_refuses_pre_name_library(P, tmp_path, monkeypatch):
    """A libpuretd built before format 2 (no bgc_open_ex) would reject
    every new export at the magic: load_lib refuses it (no ABI version),
    naming the rebuild."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    src = tmp_path / "old.c"
    src.write_text("void *bgc_open(const char *p) { (void)p; return 0; }\n")
    so = tmp_path / "libpuretd_pre_name_test.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="built before the C API ABI check.*"
                                           "bash c_api/build.sh pre_name_test"):
        P.load_lib("pre_name_test")
    assert "pre_name_test" not in P._libs


@pytest.fixture
def bench_forward(monkeypatch, tmp_path):
    """c_api/bench_forward with its export cache in tmp_path."""
    monkeypatch.setenv("BENCH_NO_PIN", "1")
    sys.path.insert(0, os.path.join(REPO, "c_api"))
    # Snapshot before the import: bench_common turns autograd off and torch
    # to one thread at import; both are restored after the test.
    grad, threads = torch.is_grad_enabled(), torch.get_num_threads()
    import bench_forward as BF
    monkeypatch.setattr(BF, "DATA_BINS", str(tmp_path / "bins"))
    yield BF
    torch.set_grad_enabled(grad)
    torch.set_num_threads(threads)


def test_bench_forward_cache_never_reuses_older_format(P, backend, bench_forward):
    """bench_forward.export_bin: an older exporter's cache entry (format 1
    under the old key `<stem>_<md5>.bin`) is ignored; a file at the new key
    that is not this format, this encoder and this width is re-exported;
    a good one is reused."""
    BF = bench_forward
    import hashlib
    model = PROD_CUBEFUL
    net = load_model(model)
    os.makedirs(BF.DATA_BINS)
    h = hashlib.md5(open(model, "rb").read()).hexdigest()[:12]
    stem = os.path.basename(model)[:-3]
    old_key = os.path.join(BF.DATA_BINS, f"{stem}_{h}.bin")
    _legacy(net, old_key)
    old_bytes = open(old_key, "rb").read()
    new_key = BF.cache_path(model)
    assert new_key != old_key and new_key.endswith(f"_{h}_bgnv2.bin")

    out = BF.export_bin(model)
    assert out == new_key and open(old_key, "rb").read() == old_bytes
    hd = EW.read_header(out)
    assert (hd["format_version"], hd["encoder_name"], hd["input_size"]) == \
        (2, net.encoder_name, net.input_size)
    ino = os.stat(out).st_ino
    assert BF.export_bin(model) == out and os.stat(out).st_ino == ino     # reused

    # planted at the new key: a format-1 file, another encoder's, truncated
    other = _net(BASE, 196, "equity")
    for plant in (lambda p: _legacy(net, p), lambda p: _export(other, p),
                  lambda p: open(p, "wb").write(b"BGNV")):
        plant(new_key)
        assert BF.export_bin(model) == new_key
        hd = EW.read_header(new_key)
        assert (hd["format_version"], hd["encoder_name"], hd["input_size"]) == \
            (2, net.encoder_name, net.input_size)
        e = P.Engine(new_key, backend)
        assert (e.encoder, e.format_version, e.input_size) == \
            (net.encoder_name, 2, net.input_size)
        e.close()
    assert not [f for f in os.listdir(BF.DATA_BINS) if f.endswith(".tmp")]
