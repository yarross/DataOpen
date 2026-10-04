"""PyTorch side of the detector: model structure and reparameterization, ONNX export, losses, augmentation correctness, a real
(tiny) training run that learns, and the same checkpoint pushed through export -> closed-loop evaluator -> INT8 quantization."""
import collections
import json
from pathlib import Path

import numpy as np
import pytest

from dataopen.detector.layout import HeadLayout

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("PIL")

from dataopen.core.schema_io import load_schema  # noqa: E402
from dataopen.detector.config import ConfigError, DetectorConfig, TrainConfig, load_config, model_config  # noqa: E402
from dataopen.detector.data import (PoseDataset, affine_matrix, collate, hflip, read_coco, transform_labels,  # noqa: E402
                                    load_schema_from_dataset)
from dataopen.detector.loss import DetectionLoss, box_iou_pairwise, ciou, task_aligned_assign  # noqa: E402
from dataopen.detector.model import RepConv, build_model, count_macs, count_params  # noqa: E402
from tests.helpers.toydata import make_toy_dataset  # noqa: E402

SCHEMA = load_schema("shooter12").schema
ROOT = Path(__file__).resolve().parents[1]


def test_variants_fit_their_compute_budgets_and_have_the_documented_head_layout():
    got = {}
    for v, (lo, hi) in {"n": (0.9, 1.5), "s": (1.9, 2.8)}.items():
        cfg = model_config(v)
        m = build_model(cfg, primary=(1,))
        macs = count_macs(m, (640, 640))
        got[v] = macs["total"] / 1e9
        assert lo < got[v] < hi, (v, got[v])
        assert count_params(m) < 3e6
    assert got["n"] < got["s"]
    m = build_model(model_config("t"), primary=(1,)).eval()
    outs = m(torch.zeros(2, 3, 128, 128))
    assert [o.shape[1] for o in outs[:3]] == [2 + 4 + 48] * 3                       # cls | box | kp offsets | kp score | kp vis
    assert [tuple(o.shape[2:]) for o in outs] == [(16, 16), (8, 8), (4, 4), (32, 32)]   # strides 8/16/32 and the stride-4 aim map
    assert outs[3].shape[1] == 3                                                  # heatmap + offset x + offset y for one primary point
    lay = m.layout(tuple(SCHEMA.keypoints), tuple(SCHEMA.classes), tuple(SCHEMA.flip_idx()), "shooter12")
    assert lay.refine_stride == 4 and lay.primary == (1,) and lay.flip_idx == tuple(SCHEMA.flip_idx())


def test_reparameterization_is_exact_and_removes_everything_the_npu_does_not_need():
    torch.manual_seed(0)
    m = build_model(model_config("t"), primary=(1,))
    for mod in m.modules():                                  # non-trivial BN statistics, otherwise the fusion test proves nothing
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.3)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.normal_(0, 0.2)
    m.eval()
    x = torch.randint(0, 256, (2, 3, 128, 128)).float()
    ref = [t.clone() for t in m(x)]
    n_rep = sum(isinstance(k, RepConv) for k in m.modules())
    m.fuse()
    got = m(x)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-3 * float(b.abs().max()) + 1e-4, rtol=1e-3)
    assert n_rep > 20 and not any(isinstance(k, torch.nn.BatchNorm2d) for k in m.modules())    # BN folded into the convs
    assert not hasattr(m, "head_o2m") and m.deployed and float(m.input_scale) == 1.0       # one-to-many head and 1/255 gone
    assert all(k.deploy for k in m.modules() if isinstance(k, RepConv))
    with pytest.raises(AttributeError):
        m.head_o2m                                           # noqa: B018


def test_box_helpers_and_the_task_aligned_assigner_behave():
    a = torch.tensor([[0.0, 0, 10, 10]])
    assert float(box_iou_pairwise(a, a)) == pytest.approx(1.0) and float(box_iou_pairwise(a, a + 20)) == 0.0
    assert float(ciou(a, a)[0]) == pytest.approx(1.0, abs=1e-4) and float(ciou(a, a + 3)[0]) < float(box_iou_pairwise(a, a + 3))
    gy, gx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    anchors = torch.stack([(gx.reshape(-1) + 0.5) * 8, (gy.reshape(-1) + 0.5) * 8], 1).float()
    stride = torch.full((256,), 8.0)
    gt = torch.tensor([[[24.0, 24, 64, 80], [0, 0, 0, 0]]])
    cls = torch.tensor([[1, -1]])
    valid = torch.tensor([[True, False]])
    scores = torch.full((1, 256, 2), 0.2)
    boxes = torch.tensor([[24.0, 24, 64, 80]]).repeat(256, 1)[None]                 # every anchor predicts the GT perfectly
    fg3, idx3, t3 = task_aligned_assign(scores, boxes, anchors, stride, gt, cls, valid, 3, 0.5, 6.0)
    fg1, _, t1 = task_aligned_assign(scores, boxes, anchors, stride, gt, cls, valid, 1, 0.5, 6.0)
    assert int(fg3.sum()) == 3 and int(fg1.sum()) == 1                                # top-k = how many anchors one object owns
    pos = fg3[0].nonzero().flatten()
    assert (anchors[pos, 0] > 24).all() and (anchors[pos, 0] < 64).all()               # only anchors inside the box
    assert (t3[0][fg3[0]].argmax(-1) == 1).all() and float(t3[0, :, 0].sum()) == 0.0   # target mass goes to the GT class only
    none_fg, _, none_t = task_aligned_assign(scores, boxes, anchors, stride, gt[:, :0], cls[:, :0], valid[:, :0], 3, 0.5, 6.0)
    assert not none_fg.any() and float(none_t.sum()) == 0.0                            # frames without people are legal


def make_batch(n=2, size=128, K=12):
    boxes = torch.tensor([[[30.0, 20, 62, 100], [70, 30, 100, 110]]] * n)
    cls = torch.tensor([[0, 1]] * n)
    kp = torch.zeros(n, 2, K, 3)
    for b in range(n):
        for m in range(2):
            x0, y0, x1, y1 = boxes[b, m]
            for k in range(K):
                kp[b, m, k] = torch.tensor([x0 + (x1 - x0) * (0.2 + 0.05 * k), y0 + (y1 - y0) * (0.05 + 0.07 * k), 2.0])
    return {"boxes": boxes, "cls": cls, "kpts": kp, "valid": torch.ones(n, 2, dtype=torch.bool)}


def test_loss_is_finite_has_every_term_gets_gradients_everywhere_and_weights_the_aim_point():
    cfg = DetectorConfig(model_config("t"), TrainConfig(batch_size=2, imgsz=128))
    m = build_model(cfg.model, primary=(1,)).train()
    lay = m.layout(tuple(SCHEMA.keypoints), tuple(SCHEMA.classes), tuple(SCHEMA.flip_idx()))
    lf = DetectionLoss(cfg.train, lay, SCHEMA.oks_sigmas(), SCHEMA.oks_weights(), (1,))
    tg = make_batch()
    out = m(torch.rand(2, 3, 128, 128) * 255)
    loss, items = lf(out, tg)
    assert torch.isfinite(loss) and {"m_cls", "m_box", "m_kp", "m_kp_score", "m_vis", "o_cls", "o_box", "heat", "total"} <= set(items)
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not dead, dead[:5]                                 # both heads, the neck, the backbone and the aim head all learn
    # the same pixel error on the aim point costs more than on a derived point (schema weights and sigmas)
    w = torch.tensor(SCHEMA.oks_weights())
    assert w[1] == w.max() and w[11] == w.min()
    empty = {"boxes": torch.zeros(2, 0, 4), "cls": torch.zeros(2, 0, dtype=torch.long), "kpts": torch.zeros(2, 0, 12, 3),
             "valid": torch.zeros(2, 0, dtype=torch.bool)}
    l0, _ = lf(m(torch.rand(2, 3, 128, 128) * 255), empty)   # negative frames: only the "nothing here" terms
    assert torch.isfinite(l0)


def test_flip_swaps_left_and_right_through_the_schema_and_is_an_involution():
    img = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    boxes = np.array([[4.0, 2, 14, 30]], np.float32)
    kp = np.zeros((1, 12, 3), np.float32)
    kp[0, :, 2] = 2
    kp[0, SCHEMA.index("l_shoulder"), :2] = (5, 10)
    kp[0, SCHEMA.index("r_shoulder"), :2] = (13, 10)
    kp[0, SCHEMA.index("head_center"), :2] = (9, 4)
    kp[0, 3, 2] = 0                                                               # an unlabeled point stays unlabeled
    f_img, f_box, f_kp = hflip(img, boxes, kp, SCHEMA.flip_idx())
    assert f_box[0].tolist() == [18.0, 2, 28, 30]
    assert f_kp[0, SCHEMA.index("l_shoulder"), :2].tolist() == [32 - 13, 10]       # the old RIGHT shoulder, mirrored, is now LEFT
    assert f_kp[0, SCHEMA.index("r_shoulder"), :2].tolist() == [32 - 5, 10]
    assert f_kp[0, SCHEMA.index("head_center"), :2].tolist() == [23, 4] and f_kp[0, 3].tolist() == [0, 0, 0]
    b_img, b_box, b_kp = hflip(f_img, f_box, f_kp, SCHEMA.flip_idx())
    assert (b_img == img).all() and np.allclose(b_box, boxes) and np.allclose(b_kp, kp)


def test_affine_transform_moves_boxes_and_keypoints_together_and_drops_what_leaves_the_frame():
    boxes = np.array([[10.0, 10, 40, 90], [100, 100, 127, 127]], np.float32)
    kp = np.zeros((2, 12, 3), np.float32)
    kp[:, :, 2] = 2
    kp[:, :, :2] = boxes[:, None, :2] + 5
    m = affine_matrix(128, 128, 128, scale=1.0, deg=0.0, tx=20.0, ty=0.0)           # shift right by 20
    nb, nk, keep = transform_labels(boxes, kp, m, 128)
    assert keep.tolist() == [True, True] and nb[0, 0] == 30 and nk[0, 0, 0] == 35
    m2 = affine_matrix(128, 128, 128, scale=1.0, deg=0.0, tx=90.0, ty=0.0)           # second person leaves the frame
    nb2, nk2, keep2 = transform_labels(boxes, kp, m2, 128)
    assert keep2.tolist() == [True, False] and (nk2[0, :, 2] >= 0).all()
    assert (nk2[0][nk2[0, :, 0] == 0][:, 2] == 0).all()                              # keypoints that left have v = 0 and x = y = 0


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return make_toy_dataset(tmp_path_factory.mktemp("toy") / "ds", 32, 8)


def test_dataset_yields_consistent_labeled_batches(toy):
    items = read_coco(toy, "train", 12)
    cfg = TrainConfig(imgsz=128, mosaic=1.0, mixup=0.5, flip=0.5, smoke_flash=0.5, workers=0)
    ds = PoseDataset(items, cfg, SCHEMA.flip_idx(), 12, train=True)
    assert len(items) == 32 and abs(ds.sample_weights().sum() - 1.0) < 1e-9
    w = ds.sample_weights()
    assert w.max() > w.min()                                                           # hard frames (weight 2) are sampled more
    batch = collate([ds[i] for i in range(6)])
    assert batch["images"].dtype == torch.uint8 and tuple(batch["images"].shape) == (6, 3, 128, 128)
    v = batch["valid"]
    assert v.any()
    b = batch["boxes"][v]
    assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all() and (b >= 0).all() and (b <= 128 + 1e-3).all()
    kp = batch["kpts"][v]
    lab = kp[..., 2] > 0
    assert (kp[..., :2][lab] >= 0).all() and (kp[..., :2][lab] <= 128).all() and (kp[..., :2][~lab] == 0).all()
    assert batch["cls"][v].min() >= 0 and batch["cls"][v].max() <= 1
    ds.set_epoch(3, close_mosaic=True)
    a1, a2 = ds[2], ds[2]
    assert (a1[0] == a2[0]).all()                                                      # deterministic for (seed, epoch, index)
    val = PoseDataset(read_coco(toy, "val", 12), cfg, SCHEMA.flip_idx(), 12, train=False)
    img, lb, it = val[0]
    assert img.shape == (128, 128, 3) and lb.scale == 1.0


def test_config_files_load_and_bad_keys_are_reported():
    for name in ("apollo_n.toml", "apollo_s.toml"):
        cfg = load_config(ROOT / "configs" / name)
        assert cfg.train.epochs == 120 and cfg.train.hard_sampling == 1.0 and cfg.model.variant == name[7]
    with pytest.raises(ConfigError):
        model_config("xl")
    bad = ROOT / "configs" / "apollo_s.toml"
    text = bad.read_text().replace("mosaic = 1.0", "mosaik = 1.0")
    tmp = Path(__file__).parent / "_bad.toml"
    tmp.write_text(text)
    try:
        with pytest.raises(ConfigError, match="mosaik"):
            load_config(tmp)
    finally:
        tmp.unlink()


@pytest.fixture(scope="module")
def trained(toy, tmp_path_factory):
    """A real (tiny) training run: 40 epochs of the toy data on CPU, no augmentation beyond jitter. ~30 s."""
    from dataopen.detector.train import train
    out = tmp_path_factory.mktemp("run")
    cfg = DetectorConfig(model_config("t"), TrainConfig(
        epochs=40, batch_size=8, imgsz=128, workers=0, amp=False, warmup_epochs=1, eval_every=20, mosaic=0.0, mixup=0.0, flip=0.0,
        smoke_flash=0.0, scale=0.1, translate=0.05, degrees=0.0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.2, lr=3e-3, close_mosaic_epochs=0,
        ema_ramp=100.0, ema_decay=0.99))
    res = train(cfg, [toy], out, device="cpu", log=lambda s: None)
    return out, res


def test_training_really_learns_and_writes_resumable_checkpoints(trained, toy):
    out, res = trained
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert len(log) == 40 and res["skipped_steps"] == 0
    assert log[-1]["total"] < 0.5 * log[0]["total"]                                      # the loss halves
    v = res["val"]
    assert v["map50"] > 0.3 and v["kp_ap_oks50"] > 0.3 and v["recall_oks50"] > 0.3, v        # a tiny model, 160 steps
    assert v["class_acc"] == 1.0                                                           # the team colour is learned
    # The all-points OKS cannot tell whether the HEAD was found (it is 1 point of 12): the aim point is checked on its own.
    # (A regression that left it ~50% of the person's height off still passed every assertion above.)
    assert v["aim_ap50"] > 0.5 and v["aim_err_med"] < 0.1 and v["aim_hit_rate"] > 0.15, v
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    from dataopen.detector.train import train
    cfg2 = DetectorConfig(model_config("t"), TrainConfig(epochs=41, batch_size=8, imgsz=128, workers=0, amp=False, eval_every=100,
                                                         mosaic=0.0, mixup=0.0, flip=0.0, smoke_flash=0.0, close_mosaic_epochs=0))
    r2 = train(cfg2, [toy], out, device="cpu", resume=True, log=lambda s: None)
    assert r2["steps"] == res["steps"] + 4                                                   # continued, did not restart


ALLOWED_OPS = {"Conv", "Relu", "Add", "Mul", "Concat", "Resize", "Constant", "Transpose", "Cast"}


def test_export_is_exact_uses_only_npu_friendly_ops_and_carries_its_layout(trained, tmp_path):
    from dataopen.detector.export import export_onnx, load_checkpoint
    import onnx
    model, lay, meta, _ = load_checkpoint(trained[0] / "best.pt")
    r = export_onnx(model, lay, tmp_path / "f.onnx", "float", schema_meta=meta)
    assert r["max_rel_diff_vs_torch"] < 1e-5 and r["outputs"] == ["p3", "p4", "p5", "aim"]
    g = onnx.load(str(tmp_path / "f.onnx"))
    ops = collections.Counter(n.op_type for n in g.graph.node)
    assert set(ops) <= ALLOWED_OPS, set(ops) - ALLOWED_OPS                                  # no softmax / LayerNorm / Gather / ...
    assert ops["Conv"] >= 20 and ops["Resize"] == 3
    assert all(next(a.s for a in n.attribute if a.name == "mode") == b"nearest" for n in g.graph.node if n.op_type == "Resize")
    md = {p.key: p.value for p in g.metadata_props}
    assert json.loads(md["apollo"])["refine_stride"] == 4 and json.loads(md["schema"])["classes"] == list(SCHEMA.classes)
    assert g.ir_version <= 8
    u = export_onnx(model, lay, tmp_path / "u.onnx", "uint8", schema_meta=meta)
    gu = onnx.load(str(tmp_path / "u.onnx"))
    assert gu.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.UINT8 and u["max_rel_diff_vs_torch"] < 1e-5
    with pytest.raises(ValueError):
        export_onnx(model, lay, tmp_path / "x.onnx", "int4")


def test_the_exported_model_runs_in_the_closed_loop_evaluator_with_identical_results(trained, toy, tmp_path):
    from dataopen.detector.evaluate import evaluate_evaluator
    from dataopen.detector.export import export_onnx, load_checkpoint
    from dataopen.detector.infer import TorchEvaluator
    from dataopen.quality.evaluators.onnx import OnnxEvaluator
    model, lay, meta, _ = load_checkpoint(trained[0] / "best.pt")
    export_onnx(model, lay, tmp_path / "u.onnx", "uint8", schema_meta=meta)
    items = read_coco(toy, "val", 12)
    schema = load_schema_from_dataset(toy)
    m_t = evaluate_evaluator(TorchEvaluator(model, lay, "cpu"), items, schema)
    ev = OnnxEvaluator(str(tmp_path / "u.onnx"), "apollo", (128, 128), conf_thr=0.25)
    m_o = evaluate_evaluator(ev, items, schema)
    for k in ("map50", "kp_ap_oks50", "recall_oks50", "aim_hit_rate", "class_acc"):
        assert m_o[k] == pytest.approx(m_t[k], abs=0.02), k
    assert ev.has_keypoints
    from dataopen.quality.metrics import DefaultMetricCalculator
    from dataopen.core.models import Annotation
    from dataopen.detector.data import load_image
    it = items[0]
    preds = ev.predict([load_image(it.path)])[0]
    gt = [Annotation(i, it.kpts[i], (float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])), {}, int(it.cls[i]))
          for i, b in enumerate(it.boxes)]
    met = DefaultMetricCalculator(schema).compute(gt, preds, "apollo")             # the quality loop's own metrics accept it
    assert met.evaluated and met.focus_evaluated and met.mean_oks > 0.3
    from onnx import TensorProto, helper
    plain = helper.make_model(helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                                                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 128, 128])],
                                                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 128, 128])]),
                              opset_imports=[helper.make_opsetid("", 13)])
    plain.ir_version = 8
    import onnx as _onnx
    _onnx.save(plain, str(tmp_path / "plain.onnx"))
    with pytest.raises(ValueError, match="metadata"):                                  # a model without a stored layout is refused
        OnnxEvaluator(str(tmp_path / "plain.onnx"), "apollo")


def test_the_trained_model_runs_through_the_production_runtime_with_the_same_quality(trained, toy, tmp_path):
    """Export -> `OrtBackend` (C post-processor, packed Q12.4 KeypointArray) -> `RuntimeEvaluator` -> the closed loop's metrics:
    the path that ships scores like the research path, and the layout sidecar for the .rknn conversion is written."""
    from dataopen.detector.evaluate import evaluate_evaluator
    from dataopen.detector.export import export_onnx, load_checkpoint
    from dataopen.detector.infer import TorchEvaluator
    from dataopen.runtime.backends import OrtBackend
    from dataopen.runtime.evaluator import RuntimeEvaluator
    model, lay, meta, _ = load_checkpoint(trained[0] / "best.pt")
    export_onnx(model, lay, tmp_path / "u.onnx", "uint8", schema_meta=meta)
    assert HeadLayout.from_json((tmp_path / "u.layout.json").read_text()) == lay
    items = read_coco(toy, "val", 12)
    schema = load_schema_from_dataset(toy)
    m_t = evaluate_evaluator(TorchEvaluator(model, lay, "cpu"), items, schema)
    ev = RuntimeEvaluator(OrtBackend(tmp_path / "u.onnx", "cpu", conf_thr=0.25), window=4)
    m_r = evaluate_evaluator(ev, items, schema)
    snap = ev.runtime_metrics()
    ev.close()
    for k in ("map50", "kp_ap_oks50", "recall_oks50", "aim_hit_rate", "class_acc"):
        assert m_r[k] == pytest.approx(m_t[k], abs=0.03), k
    assert snap["published"] == len(items) and snap["dropped_total"] == 0 and snap["errors"]["total"] == 0


def test_int8_quantization_keeps_the_outputs_close_and_reports_how_close(trained, toy, tmp_path):
    from dataopen.detector.evaluate import evaluate_evaluator
    from dataopen.detector.export import export_onnx, load_checkpoint
    from dataopen.detector.quantize import calibration_files, compare_outputs, quantize_ort, rknn_convert
    from dataopen.quality.evaluators.onnx import OnnxEvaluator
    model, lay, meta, _ = load_checkpoint(trained[0] / "best.pt")
    export_onnx(model, lay, tmp_path / "f.onnx", "float", schema_meta=meta)
    files = sorted((toy / "images" / "train").glob("*.png"))
    rep = quantize_ort(tmp_path / "f.onnx", tmp_path / "q.onnx", files, "percentile", "uint8", min_images=500)
    assert "warning" in rep and rep["weights"] == "int8 per-channel" and rep["activations"] == "uint8 per-tensor"
    assert rep["size_mb"] < 0.6 * (tmp_path / "f.onnx").stat().st_size / 1e6 * 1.0 + 0.2                  # ~4x smaller weights
    import onnx
    ops = collections.Counter(n.op_type for n in onnx.load(str(tmp_path / "q.onnx")).graph.node)
    assert ops["QuantizeLinear"] > 10 and ops["DequantizeLinear"] > 40                                       # real QDQ, not a copy
    sim = compare_outputs(tmp_path / "f.onnx", tmp_path / "q.onnx", files, 8)
    assert all(v["cosine"] > 0.99 for v in sim.values()), sim
    schema = load_schema_from_dataset(toy)
    items = read_coco(toy, "val", 12)
    m32 = evaluate_evaluator(OnnxEvaluator(str(tmp_path / "f.onnx"), "apollo", (128, 128)), items, schema)
    m8 = evaluate_evaluator(OnnxEvaluator(str(tmp_path / "q.onnx"), "apollo", (128, 128)), items, schema)
    assert m8["map50"] >= m32["map50"] - 0.25 and m8["class_acc"] >= 0.9                                    # INT8 stays usable
    with pytest.raises(ValueError):
        quantize_ort(tmp_path / "f.onnx", tmp_path / "x.onnx", [], "percentile")
    with pytest.raises(ValueError):
        quantize_ort(tmp_path / "f.onnx", tmp_path / "x.onnx", files[:2], "bogus")
    script = rknn_convert("f.onnx", "calib/dataset.txt", "m.rknn", tmp_path / "convert.py", "rk3588", "mmse", (128, 128))
    text = script.read_text()
    compile(text, "convert.py", "exec")                                                                     # valid Python
    assert "asymmetric_quantized-8" in text and 'quantized_method="channel"' in text and "dataset.txt" in text
    assert calibration_files(tmp_path / "nonexistent") == []


def test_the_bench_budget_is_arithmetic_and_the_device_script_is_valid_python(tmp_path):
    from dataopen.detector.bench import latency_budget, meets_targets, rknn_bench_script
    b = latency_budget(1.2e9)
    assert b["gmacs"] == 1.2 and b["all_cores_ms@50%"] < b["all_cores_ms@20%"] and b["one_core_ms@35%"] > b["all_cores_ms@35%"]
    assert b["all_cores_ms@35%"] == pytest.approx(2 * 1.2e9 / (6e12 * 0.35) * 1e3, abs=0.01)
    assert meets_targets({"all_cores": {"p99_ms": 3.2}, "throughput_fps_3_cores": 260}) == []
    assert len(meets_targets({"all_cores": {"p99_ms": 5.0}, "throughput_fps_3_cores": 100})) == 2 and meets_targets({}) is None
    s = rknn_bench_script(tmp_path / "bench.py", "m.rknn", 640)
    compile(s.read_text(), "bench.py", "exec")
    assert "NPU_CORE_0_1_2" in s.read_text() and "4.0" in s.read_text() and "240" in s.read_text()


def test_cli_end_to_end_train_eval_export_calib_quantize_bench(toy, tmp_path, capsys):
    from dataopen.cli import main

    def run(*argv):
        with pytest.raises(SystemExit) as e:
            main(list(argv))
        return e.value.code

    out = tmp_path / "run"
    assert run("detector", "train", "--variant", "t", "--data", str(toy), "--out", str(out), "--device", "cpu", "--epochs", "1",
               "--batch-size", "8", "--imgsz", "128", "--workers", "0", "--max-steps", "3") == 0
    assert (out / "best.pt").exists()
    assert run("detector", "eval", "--model", str(out / "best.pt"), "--data", str(toy), "--limit", "4") == 0
    assert run("detector", "eval", "--model", str(out / "best.pt"), "--data", str(toy), "--limit", "4", "--require") == 1   # 3 steps: no
    assert run("detector", "export", "--ckpt", str(out / "best.pt"), "--out", str(tmp_path / "m.onnx")) == 0
    assert run("detector", "eval", "--model", str(tmp_path / "m.onnx"), "--data", str(toy), "--limit", "4") == 0
    assert run("detector", "bench", "--ckpt", str(out / "best.pt"), "--onnx", str(tmp_path / "m.onnx"), "--runs", "3",
               "--rknn-script", str(tmp_path / "b.py")) == 0
    assert (tmp_path / "b.py").exists()
    assert run("detector", "header", "--out", str(tmp_path / "h.h")) == 0 and "APOLLO_MAGIC" in (tmp_path / "h.h").read_text()
    assert run("detector", "train", "--variant", "t", "--data", str(tmp_path / "nothing"), "--out", str(tmp_path / "x")) == 4
    # QAT and mixed precision from the command line
    assert run("detector", "qat", "--ckpt", str(out / "best.pt"), "--data", str(toy), "--out", str(tmp_path / "qat.onnx"), "--epochs", "1",
               "--batch-size", "8", "--workers", "0", "--calib-batches", "1", "--device", "cpu") == 0
    assert (tmp_path / "qat.onnx").exists()
    calib = tmp_path / "calib"
    assert run("detector", "calib", "--data", str(toy), "--out", str(calib), "--n", "6", "--size", "128") == 0
    assert run("detector", "sensitivity", "--onnx", str(tmp_path / "m.onnx"), "--calib", str(calib), "--work", str(tmp_path / "sw"),
               "--n-images", "2", "--top", "3") == 0
    assert run("detector", "quantize", "--onnx", str(tmp_path / "m.onnx"), "--calib", str(calib), "--out", str(tmp_path / "mx.onnx"),
               "--min-images", "0", "--method", "minmax", "--exclude-worst", "2") == 0
    assert (tmp_path / "mx.onnx").exists() and (tmp_path / "m.layout.json").exists()
    capsys.readouterr()


def test_a_trained_detector_runs_inside_the_closed_loop_as_the_baseline_model(trained, tmp_path):
    """The loop that produced the training data can now be driven by the model trained on it (and refuses a mismatched one)."""
    from dataopen.cli import main
    from dataopen.detector.export import export_onnx, load_checkpoint
    model, lay, meta, _ = load_checkpoint(trained[0] / "best.pt")
    export_onnx(model, lay, tmp_path / "m.onnx", "uint8", schema_meta=meta)
    out = tmp_path / "ds"
    with pytest.raises(SystemExit) as e:
        main(["collect", "--game", "mock_shooter", "--frames", "20", "--out", str(out), "--no-doctor", "--adaptive",
              "--quality-model", str(tmp_path / "m.onnx"), "--quality-format", "apollo"])
    assert e.value.code == 0
    recs = [json.loads(x) for p in (out / "annotations").glob("*.jsonl") for x in p.read_text().splitlines()]
    q = [r["meta"]["quality"] for r in recs]
    assert len(recs) == 20 and all(x["metrics"]["evaluated"] and x["metrics"]["backend"].startswith("onnx:apollo") for x in q)
    assert any(x["metrics"]["focus_evaluated"] for x in q)                      # the model's aim point was judged against the labels
    bad = tmp_path / "other"
    with pytest.raises(SystemExit) as e2:                                      # a 13-point dataset cannot use a 12-point model
        main(["collect", "--game", "mock", "--frames", "5", "--out", str(bad), "--no-doctor", "--quality-model",
              str(tmp_path / "m.onnx"), "--quality-format", "apollo"])
    assert e2.value.code == 4


def test_aim_point_regression_has_a_gradient_even_when_it_starts_far_away():
    """Regression for two real bugs: (1) the OKS-only keypoint loss vanishes a few pixels from the target when the sigma is tight
    (the aim point's), so a point that starts far off never converged; (2) the heatmap's positive cell was never marked."""
    cfg = TrainConfig(w_kp=0.0, w_kp_l1=6.0, w_cls=0.0, w_box=0.0, w_kp_score=0.0, w_vis=0.0, w_heat=0.0, tal_topk_o2m=3)
    lay = build_model(model_config("t"), primary=(1,)).layout(tuple(SCHEMA.keypoints), tuple(SCHEMA.classes))
    lf = DetectionLoss(cfg, lay, SCHEMA.oks_sigmas(), SCHEMA.oks_weights(), (1,))
    tg = make_batch(1)
    shapes = [(16, 16), (8, 8), (4, 4)]
    outs = [torch.zeros(1, lay.channels, h, w, requires_grad=True) for h, w in shapes]
    with torch.no_grad():
        for o in outs:
            o[:, 2:6] = 1.0                                                      # boxes of a plausible size so anchors get assigned
            o[:, :2] = 2.0
    loss, items = lf.head_loss(outs, tg, 3)
    loss.backward()
    kp_grad = sum(float(o.grad[:, 6:30].abs().sum()) for o in outs)
    assert float(items["kp_l1"]) > 0 and kp_grad > 0, "the keypoint offsets must receive gradient from the distance term"
    # the heatmap: the loss is low when the logit peaks on the object's centre cell and high when it peaks elsewhere
    cfg2 = TrainConfig(w_heat=1.0)
    lf2 = DetectionLoss(cfg2, build_model(model_config("t"), primary=(1,)).layout(tuple(SCHEMA.keypoints), tuple(SCHEMA.classes)),
                        SCHEMA.oks_sigmas(), SCHEMA.oks_weights(), (1,))
    tg2 = make_batch(1)
    tg2["valid"][0, 1] = False                                                   # one object: one positive cell
    stride = 4
    x, y = float(tg2["kpts"][0, 0, 1, 0]), float(tg2["kpts"][0, 0, 1, 1])
    right = torch.full((1, 3, 32, 32), -8.0)
    right[0, 0, int(round(y / stride - 0.5)), int(round(x / stride - 0.5))] = 8.0
    wrong = torch.full((1, 3, 32, 32), -8.0)
    wrong[0, 0, 0, 0] = 8.0
    assert float(lf2.heat_loss(right, tg2)) < 0.5 * float(lf2.heat_loss(wrong, tg2))


def _ddp_worker(rank, world, root, out, port, q):
    import os
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    try:
        from dataopen.detector.train import train
        cfg = DetectorConfig(model_config("t"), TrainConfig(epochs=2, batch_size=8, imgsz=128, workers=0, amp=False, eval_every=1,
                                                            mosaic=0.0, mixup=0.0, flip=0.0, smoke_flash=0.0, close_mosaic_epochs=0))
        res = train(cfg, [Path(root)], Path(out), device="cpu", log=lambda s: None)
        q.put((rank, res["param_checksum"], res["steps"], res["world_size"]))
    except BaseException as e:                                                      # report instead of hanging the parent
        q.put((rank, repr(e), -1, -1))


def test_data_parallel_training_keeps_the_ranks_in_sync_and_only_rank_zero_writes(toy, tmp_path):
    """Two real processes (gloo, CPU): DDP must leave identical parameters on both ranks; one set of checkpoints."""
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, str(toy), str(tmp_path / "ddp"), port, q)) for r in range(2)]
    for p in procs:
        p.start()
    got = sorted([q.get(timeout=240) for _ in procs])
    for p in procs:
        p.join(30)
    assert all(isinstance(g[1], float) for g in got), got
    assert got[0][1] == pytest.approx(got[1][1], rel=1e-9) and got[0][2] == got[1][2] > 0 and got[0][3] == 2
    assert (tmp_path / "ddp" / "best.pt").exists() and (tmp_path / "ddp" / "last.pt").exists()
    lines = (tmp_path / "ddp" / "train_log.jsonl").read_text().splitlines()
    assert len(lines) == 2                                                          # one log, written by rank 0 only


def test_the_global_batch_must_split_evenly_across_ranks(toy, tmp_path, monkeypatch):
    from dataopen.detector.data import DataError
    from dataopen.detector.train import train
    monkeypatch.setenv("WORLD_SIZE", "3")
    cfg = DetectorConfig(model_config("t"), TrainConfig(epochs=1, batch_size=8, imgsz=128, workers=0))
    with pytest.raises(DataError, match="divisible"):
        train(cfg, [toy], tmp_path / "x", device="cpu")


def test_qat_finetunes_a_fused_model_exports_qdq_and_stays_usable(trained, toy, tmp_path):
    from torch.utils.data import DataLoader

    from dataopen.detector.evaluate import evaluate_evaluator
    from dataopen.detector.export import load_checkpoint
    from dataopen.detector.qat import FakeQuantConv, export_qat_onnx, prepare_qat, qat_finetune
    from dataopen.quality.evaluators.onnx import OnnxEvaluator
    model, lay, meta, _ = load_checkpoint(trained[0] / "best.pt")
    with pytest.raises(ValueError, match="fuse"):
        prepare_qat(model)                                                           # only the deploy graph can be quantized
    model.fuse()
    q = prepare_qat(model)
    n_conv = sum(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    wrapped = [m for m in q.modules() if isinstance(m, FakeQuantConv)]
    assert len(wrapped) == n_conv and sum(m.quant_output for m in wrapped) == 10      # 3 head convs x 3 levels + the aim map
    tc = TrainConfig(imgsz=128, mosaic=0.0, mixup=0.0, flip=0.0, smoke_flash=0.0, workers=0)
    ds = PoseDataset(read_coco(toy, "train", 12), tc, SCHEMA.flip_idx(), 12, train=True)
    lf = DetectionLoss(tc, lay, SCHEMA.oks_sigmas(), SCHEMA.oks_weights(), model.primary)
    net, info = qat_finetune(model, lf, DataLoader(ds, batch_size=8, collate_fn=collate), epochs=1, lr=1e-4, calib_batches=2,
                             log=lambda s: None, max_steps=3)
    assert info["steps"] == 3 and all(np.isfinite(info["loss"]))
    assert all(float(m.in_seen) == 1.0 and float(m.in_max) > float(m.in_min) for m in net.modules() if isinstance(m, FakeQuantConv))
    r = export_qat_onnx(net, lay, tmp_path / "qat.onnx", meta)
    assert r["max_rel_diff_vs_torch"] < 0.15, r                                      # ORT QDQ == the fake-quant forward (rounding aside)
    import onnx
    ops = collections.Counter(n.op_type for n in onnx.load(str(tmp_path / "qat.onnx")).graph.node)
    assert ops["QuantizeLinear"] > 30 and ops["DequantizeLinear"] > 30
    # weights sit exactly on a per-channel int8 grid, so a later per-channel int8 quantizer (RKNN) reproduces them
    w = next(m for m in net.modules() if isinstance(m, FakeQuantConv)).conv.weight.detach()
    sc = w.abs().amax(dim=(1, 2, 3)) / 127.0
    grid = w / sc[:, None, None, None]
    assert float((grid - grid.round()).abs().max()) < 1e-3
    items = read_coco(toy, "val", 12)
    m = evaluate_evaluator(OnnxEvaluator(str(tmp_path / "qat.onnx"), "apollo", (128, 128)), items, load_schema_from_dataset(toy))
    assert m["class_acc"] >= 0.9 and m["map50"] > 0.2


def test_layer_sensitivity_ranks_convolutions_and_mixed_precision_keeps_them_in_float(trained, toy, tmp_path):
    from dataopen.detector.export import export_onnx, load_checkpoint
    from dataopen.detector.quantize import layer_sensitivity, quantize_ort
    import onnx
    model, lay, meta, _ = load_checkpoint(trained[0] / "best.pt")
    export_onnx(model, lay, tmp_path / "f.onnx", "float", schema_meta=meta)
    files = sorted((toy / "images" / "train").glob("*.png"))[:4]
    rows = layer_sensitivity(tmp_path / "f.onnx", files, tmp_path / "w", n_images=3, top=0)
    n_conv = sum(n.op_type == "Conv" for n in onnx.load(str(tmp_path / "f.onnx")).graph.node)
    assert len(rows) == n_conv and rows == sorted(rows, key=lambda r: r["min_cosine"])
    assert all(0.0 <= r["min_cosine"] <= 1.0 + 1e-6 for r in rows)
    worst = [r["node"] for r in rows[:3]]
    rep = quantize_ort(tmp_path / "f.onnx", tmp_path / "mixed.onnx", files, "minmax", "uint8", min_images=0, nodes_to_exclude=worst)
    assert rep["nodes_kept_in_float"] == worst
    g = onnx.load(str(tmp_path / "mixed.onnx"))
    q_in = {i for n in g.graph.node if n.op_type == "DequantizeLinear" for i in n.output}
    kept = {n.name: n for n in g.graph.node if n.name in worst}
    assert all(n.op_type == "Conv" and not any(i in q_in for i in n.input[:2]) for n in kept.values())    # float conv: no DQ inputs
