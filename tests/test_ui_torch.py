"""The assistive UI detector, the parts that need PyTorch (run in the `detector` CI job): network, loss, a short real training run, ONNX export."""  # noqa: E501

import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataopen.runtime.frames import Frame  # noqa: E402
from dataopen.ui.data import SynthUi, collate  # noqa: E402
from dataopen.ui.evaluate import evaluate  # noqa: E402
from dataopen.ui.model import UiConfig, UiNet  # noqa: E402
from dataopen.ui.taxonomy import NAMES, STRIDES, NotAUiModel  # noqa: E402

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


def test_the_exporter_refuses_a_model_that_is_not_a_ui_model(tmp_path):
    from dataopen.ui.export import export_onnx

    net = UiNet(UiConfig(width=0.25, neck=16, classes=("player_ct", "player_t"), n_cls=2))
    with pytest.raises(NotAUiModel):
        export_onnx(net, tmp_path / "x.onnx")


def test_the_network_has_three_scales_no_keypoints_and_a_small_size():
    net = UiNet(UiConfig(width=0.5, neck=48)).eval()
    outs = net(torch.rand(1, 3, 128, 128))
    assert [tuple(o.shape) for o in outs] == [(1, len(NAMES) + 4, 128 // s, 128 // s) for s in STRIDES]
    assert net.n_params() < 700_000


def test_boxes_are_assigned_to_the_scale_that_fits_them_and_a_tiny_box_still_gets_a_cell():
    from dataopen.ui.loss import assign

    boxes = torch.tensor([[100.0, 100, 120, 114], [200, 300, 330, 340], [10, 10, 12.5, 12.5]])
    cls = torch.tensor([0, 2, 7])
    for lv, stride in enumerate(STRIDES):
        ct, bt = assign(boxes, cls, 640 // stride, 640 // stride, stride, lv, len(NAMES))
        found = set(ct[ct >= 0].tolist())
        if lv == 0:
            assert 0 in found and 7 in found and 2 not in found  # small things on the fine level
        if lv == 2:
            assert 2 in found and 0 not in found
    ct, _ = assign(boxes, cls, 80, 80, 8, 0, len(NAMES))
    n7 = int((ct == 7).sum())
    assert 1 <= n7 <= 9 and ct[1, 1] == 7  # the 2.5 px box: the cells around its centre, including the one holding it


def test_the_loss_is_finite_has_gradients_and_handles_empty_screens():
    from dataopen.ui.loss import ui_loss

    net = UiNet(UiConfig(width=0.25, neck=16))
    x = torch.rand(2, 3, 256, 256)
    gts = [
        (torch.tensor([[40.0, 50, 120, 90], [200, 10, 214, 24]]), torch.tensor([0, 7])),
        (torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)),
    ]
    loss, parts = ui_loss(net(x), gts, len(NAMES))
    loss.backward()
    assert torch.isfinite(loss) and parts["pos"] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_the_training_machinery_learns_a_handful_of_screens():
    """Not a quality claim: a tiny net on 3 fixed screens must drive the loss down and find most of the elements again."""
    from dataopen.ui.loss import ui_loss
    from dataopen.ui.train import predict

    torch.manual_seed(0)
    ds = SynthUi(3, seed=7, train=False)
    x, gts, _, _ = collate([ds[i] for i in range(3)])
    x = x.float() / 255
    net = UiNet(UiConfig(width=0.5, neck=32))
    opt = torch.optim.AdamW(net.parameters(), 3e-3)
    first = None
    for _ in range(120):
        loss, _ = ui_loss(net(x), gts, len(NAMES))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(loss)
    assert float(loss) < 0.5 * first
    r = evaluate(*predict(net, ds, 3, conf=0.2))
    assert r.map50 > 0.2 and r.recall_targets > 0.3


# ---------------------------------------------------------------- tracker and scene
def test_onnx_export_is_plain_ops_matches_torch_and_the_detector_checks_the_classes(tmp_path):
    from dataopen.ui.export import export_onnx
    from dataopen.ui.service import OrtUiDetector

    net = UiNet(UiConfig(width=0.25, neck=16))
    r = export_onnx(net, tmp_path / "ui.onnx")
    assert r["max_rel_diff_vs_torch"] < 1e-4 and set(r["ops"]) <= {
        "Conv",
        "Relu",
        "Add",
        "Resize",
        "Concat",
        "Constant",
        "Cast",
        "Transpose",
        "Mul",
        "Identity",
    }
    det = OrtUiDetector(tmp_path / "ui.onnx", conf=0.01)
    assert det.classes == NAMES
    out = det.detect(Frame(np.random.default_rng(0).integers(0, 256, (640, 640, 3), dtype=np.uint8), 0, 0))
    assert all(d.cls in NAMES for d in out)
    with pytest.raises(ValueError):
        det.detect(Frame(np.zeros((480, 640, 3), np.uint8), 0, 0))
    import onnx

    m = onnx.load(str(tmp_path / "ui.onnx"))
    m.metadata_props[0].value = m.metadata_props[0].value.replace("button", "player_ct")
    onnx.save(m, str(tmp_path / "bad.onnx"))
    with pytest.raises(NotAUiModel):
        OrtUiDetector(tmp_path / "bad.onnx")
    m2 = onnx.load(str(tmp_path / "ui.onnx"))
    del m2.metadata_props[:]
    onnx.save(m2, str(tmp_path / "plain.onnx"))
    with pytest.raises(ValueError):
        OrtUiDetector(tmp_path / "plain.onnx")


# ---------------------------------------------------------------- the whole assistive chain with a detector-derived scene
