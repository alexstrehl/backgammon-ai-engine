"""tools/distill.py: prob5 and scalar students fit their teacher's
labels, the output resumes in the trainers' loaders, and mismatched
inputs are refused."""
import hashlib
import os

import pytest
import torch

from model import ProbNetwork, TDNetwork, load_model
from tools import collect_positions as cp
from tools import distill

SEED = 3


def _save(tmp, name, net):
    path = str(tmp / f"{name}.pt")
    net.save(path)
    return path


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    """Tiny teacher nets and one 2-game bank per kind."""
    tmp = tmp_path_factory.mktemp("distill")
    torch.manual_seed(SEED)
    nets = {"p5": ProbNetwork(hidden_sizes=[8]),
            "cbl": TDNetwork(hidden_sizes=[8], output_mode="equity"),
            "cf": TDNetwork(hidden_sizes=[8], output_mode="equity",
                            encoder_name="cubeful_perspective196")}
    out = {"tmp": tmp}
    for k, net in nets.items():
        out[k] = _save(tmp, k, net)
        out[k + "_rows"] = str(tmp / f"{k}.npz")
        cp.main(["--model", out[k], "--games", "2", "--workers", "1", "--seed", str(SEED),
                 "--out", out[k + "_rows"]])
    return out


def _distill(banks, teacher, out, *extra):
    return distill.main(["--teacher", banks[teacher], "--rows", banks[teacher + "_rows"],
                         "--epochs", "4", "--batch", "16", "--lr", "3e-3", "--val-frac", "0.2",
                         "--device", "cpu", "--seed", str(SEED), "--out", out, *extra])


def test_prob5_end_to_end(banks):
    out = str(banks["tmp"] / "p5_student.pt")
    r = _distill(banks, "p5", out, "--hidden", "8,4", "--init-from", banks["p5"])
    assert r["val_loss"] < r["val_loss0"]
    stu = load_model(out)
    assert isinstance(stu, ProbNetwork) and stu.hidden_sizes == [8, 4]
    ProbNetwork.load(out)  # train_prob5 --resume
    tp = stu._train_params
    assert tp["teacher_md5"] == hashlib.md5(open(banks["p5"], "rb").read()).hexdigest()
    assert tp["rows"] == ["p5.npz"] and len(tp["rows_md5"]) == 1
    assert tp["init_from"] == "p5.pt" and tp["argv"].startswith("tools/distill.py ")
    assert tp["val_loss"] == r["val_loss"]
    with pytest.raises(SystemExit):  # never overwrites
        _distill(banks, "p5", out, "--hidden", "8")


def test_scalar_end_to_end_and_resume(banks):
    out = str(banks["tmp"] / "cbl_student.pt")
    r = _distill(banks, "cbl", out, "--hidden", "6", "--rows-on-cpu")
    assert r["val_loss"] < r["val_loss0"]
    stu = load_model(out)
    assert isinstance(stu, TDNetwork) and stu.hidden_sizes == [6] and stu.output_mode == "equity"
    TDNetwork.load(out)  # train_batch --resume
    out2 = str(banks["tmp"] / "cbl_student2.pt")
    r2 = _distill(banks, "cbl", out2, "--resume", out)
    assert r2["val_loss0"] == pytest.approx(r["val_loss"])
    assert load_model(out2)._train_params["resume"] == "cbl_student.pt"


def test_cubeful_end_to_end(banks):
    out = str(banks["tmp"] / "cf_student.pt")
    r = _distill(banks, "cf", out, "--hidden", "8", "--init-from", banks["cf"])
    assert r["val_loss"] < r["val_loss0"]
    assert load_model(out).input_size == 200


def test_init_from_copies_leading_layers(banks):
    teacher = load_model(banks["p5"])
    stu = distill.build_student(teacher, [8, 4], banks["p5"])
    assert torch.equal(stu.trunk[0].weight, teacher.trunk[0].weight)
    with pytest.raises(SystemExit):  # no shape-matching layer
        distill.build_student(teacher, [16], banks["p5"])
    with pytest.raises(SystemExit):  # wrong class
        distill.build_student(teacher, [8], banks["cbl"])


def test_rejects_mismatched_inputs(banks):
    tmp = banks["tmp"]
    with pytest.raises(SystemExit):  # 200-feature bank, 196-feature teacher
        distill.main(["--teacher", banks["cbl"], "--rows", banks["cf_rows"], "--hidden", "4",
                      "--device", "cpu", "--out", str(tmp / "x1.pt")])
    with pytest.raises(SystemExit):  # same width, but a prob5 teacher cannot play cubeless-money
        distill.main(["--teacher", banks["p5"], "--rows", banks["cbl_rows"], "--hidden", "4",
                      "--device", "cpu", "--out", str(tmp / "x2.pt")])
    dmp = _save(tmp, "dmp", TDNetwork(hidden_sizes=[4], output_mode="probability"))
    with pytest.raises(SystemExit):  # resumed student has another output mode
        distill.main(["--teacher", banks["cbl"], "--rows", banks["cbl_rows"], "--resume", dmp,
                      "--device", "cpu", "--out", str(tmp / "x3.pt")])
    with pytest.raises(SystemExit):  # --hidden and --resume are exclusive
        distill.main(["--teacher", banks["cbl"], "--rows", banks["cbl_rows"], "--hidden", "4",
                      "--resume", banks["cbl"], "--device", "cpu", "--out", str(tmp / "x4.pt")])
    assert not any(os.path.exists(str(tmp / f"x{i}.pt")) for i in range(1, 5))
