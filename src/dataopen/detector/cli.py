"""`dataopen detector ...` and `dataopen agreement`: the training / export / quantization / benchmark commands.
Handlers import torch lazily so `dataopen` itself stays importable without it."""
from __future__ import annotations

import json
from pathlib import Path

EXIT_OK, EXIT_FAILED_CHECK, EXIT_USAGE = 0, 1, 4


def _need_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise SystemExit("this command needs PyTorch: pip install 'dataopen[train]'  (CPU wheels: "
                         "pip install torch --index-url https://download.pytorch.org/whl/cpu)") from e


def _print(d) -> None:
    print(json.dumps(d, indent=2, default=str))


def _cmd_train(a) -> int:
    _need_torch()
    from .config import ConfigError, load_config, model_config
    from .data import DataError
    from .train import train
    try:
        cfg = load_config(a.config) if a.config else None
        if cfg is None:
            from .config import DetectorConfig
            cfg = DetectorConfig(model_config(a.variant))
        if a.variant and not a.config:
            cfg.model = model_config(a.variant)
        for k in ("epochs", "batch_size", "imgsz", "lr", "workers"):
            if getattr(a, k) is not None:
                setattr(cfg.train, k, getattr(a, k))
        if a.imgsz:
            cfg.model.input_size = (a.imgsz, a.imgsz)
        res = train(cfg, [Path(d) for d in a.data], Path(a.out), a.device, a.resume, a.max_steps)
    except (ConfigError, DataError, FloatingPointError) as e:
        print(f"train failed: {e}")
        return EXIT_USAGE
    _print(res)
    return EXIT_OK


def _load_eval_model(a):
    """-> (IModelEvaluator, SkeletonSchema) from a .pt checkpoint or an exported .onnx."""
    from ..core.card import schema_from_card_dict
    from .data import load_schema_from_dataset
    if str(a.model).endswith(".pt"):
        from .export import load_checkpoint
        from .infer import TorchEvaluator
        model, lay, meta, _ = load_checkpoint(a.model, device=a.device)
        return TorchEvaluator(model, lay, a.device, a.conf), schema_from_card_dict(meta)
    from ..quality.evaluators.onnx import OnnxEvaluator
    ev = OnnxEvaluator(a.model, "apollo", (640, 640), a.device if a.device in ("cpu", "cuda", "tensorrt", "directml") else "cpu",
                       conf_thr=a.conf)
    return ev, load_schema_from_dataset(Path(a.data[0]))


def _cmd_eval(a) -> int:
    _need_torch()
    from .data import read_coco
    from .evaluate import check_targets, evaluate_evaluator
    ev, schema = _load_eval_model(a)
    items = [it for d in a.data for it in read_coco(Path(d), a.split, schema.num_keypoints)]
    if a.limit:
        items = items[:a.limit]
    m = evaluate_evaluator(ev, items, schema, a.conf)
    _print(m)
    if a.require:
        bad = check_targets(m)
        for b in bad:
            print("TARGET MISSED:", b)
        return EXIT_FAILED_CHECK if bad else EXIT_OK
    return EXIT_OK


def _cmd_export(a) -> int:
    _need_torch()
    from .export import export_onnx, load_checkpoint
    model, lay, meta, _ = load_checkpoint(a.ckpt, device="cpu")
    _print(export_onnx(model, lay, a.out, a.input, a.opset, meta))
    return EXIT_OK


def _cmd_qat(a) -> int:
    """QAT: fine-tune the fused deploy model with fake quantization on the training split, export QDQ ONNX (the fallback when PTQ
    loses more than the budget; docs/DETECTOR.md section 6)."""
    _need_torch()
    import torch
    from torch.utils.data import DataLoader

    from .config import TrainConfig
    from .data import PoseDataset, collate, load_schema_from_dataset, read_coco
    from .export import load_checkpoint
    from .loss import DetectionLoss
    from .qat import export_qat_onnx, qat_finetune
    dev = "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    model, lay, meta, _ = load_checkpoint(a.ckpt, device=dev)
    model.fuse()
    schema = load_schema_from_dataset(Path(a.data[0]))
    items = [it for d in a.data for it in read_coco(Path(d), "train", schema.num_keypoints)]
    tc = TrainConfig(imgsz=lay.input_size[0], mosaic=0.0, mixup=0.0, workers=a.workers, batch_size=a.batch_size)
    ds = PoseDataset(items, tc, schema.flip_idx(), schema.num_keypoints, train=True, imgsz=tc.imgsz)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate,
                        drop_last=len(ds) >= a.batch_size)
    loss_fn = DetectionLoss(tc, lay, schema.oks_sigmas(), schema.oks_weights(), model.primary)
    net, info = qat_finetune(model, loss_fn, loader, a.epochs, a.lr, dev, a.calib_batches)
    _print({"qat": info, "export": export_qat_onnx(net, lay, a.out, meta)})
    return EXIT_OK


def _cmd_sensitivity(a) -> int:
    """Which convolutions lose the most when quantized: the candidates for float / INT16 (mixed precision)."""
    from .quantize import calibration_files, layer_sensitivity
    rows = layer_sensitivity(a.onnx, calibration_files(a.calib, a.n_images), Path(a.work), a.n_images, a.top)
    _print(rows)
    return EXIT_OK


def _cmd_calib(a) -> int:
    from .calibration_set import build_calibration_set
    try:
        rep = build_calibration_set([Path(d) for d in a.data], Path(a.out), a.n, a.size, a.seed, a.hard_share)
    except ValueError as e:
        print(f"calibration set failed: {e}")
        return EXIT_USAGE
    _print(rep)
    return EXIT_OK


def _cmd_quantize(a) -> int:
    from .quantize import calibration_files, compare_outputs, quantize_ort, rknn_convert
    files = calibration_files(a.calib)
    rep = {}
    if a.out:
        keep_float = list(a.exclude or [])
        if a.exclude_worst:                                             # mixed precision: the N most sensitive convs stay float
            from .quantize import layer_sensitivity
            rows = layer_sensitivity(a.onnx, files, Path(a.out).parent / "sensitivity_work", 8, 0)
            keep_float += [r["node"] for r in rows[:a.exclude_worst]]
        rep["ort"] = quantize_ort(a.onnx, a.out, files, a.method, a.activation, a.min_images, nodes_to_exclude=keep_float or None)
        rep["output_similarity"] = compare_outputs(a.onnx, a.out, files)
    if a.rknn_script:
        rep["rknn_script"] = str(rknn_convert(a.onnx, Path(a.calib) / "dataset.txt", a.rknn_out or "apollo_int8.rknn", a.rknn_script,
                                              a.platform, a.rknn_algorithm))
        rep["note"] = "the RKNN script must be run with rknn-toolkit2 on an x86 Linux host; it was not run here"
    _print(rep)
    return EXIT_OK


def _cmd_bench(a) -> int:
    from .bench import bench_onnx, latency_budget, rknn_bench_script
    res = {}
    if a.ckpt:
        _need_torch()
        from .export import load_checkpoint
        from .model import count_macs, count_params
        model, lay, _, cfg = load_checkpoint(a.ckpt)
        macs = count_macs(model, lay.input_size)
        res["model"] = {"params_m": round(count_params(model) / 1e6, 3), "gmacs": round(macs["total"] / 1e9, 3),
                        "by_block_gmacs": {k: round(v / 1e9, 3) for k, v in macs.items() if k != "total"}}
        res["napkin_npu_budget"] = latency_budget(macs["total"])
    if a.onnx:
        res["host_cpu"] = bench_onnx(a.onnx, a.runs)
    if a.rknn_script:
        res["rknn_script"] = str(rknn_bench_script(a.rknn_script, a.rknn_model))
    _print(res)
    return EXIT_OK


def _cmd_header(a) -> int:
    from .structs import header_text
    Path(a.out).write_text(header_text())
    print(f"written: {a.out}")
    return EXIT_OK


def _cmd_agreement(a) -> int:
    from .agreement import compare_coco
    rep = compare_coco(a.a, a.b, a.thr)
    _print(rep if a.full else {k: v for k, v in rep.items() if k != "images_to_review"})
    return EXIT_OK if rep["passes"] else EXIT_FAILED_CHECK


def register(sub) -> None:
    d = sub.add_parser("detector", help="NPU keypoint detector: train / eval / export / quantize / bench (docs/DETECTOR.md)")
    ds = d.add_subparsers(dest="det_cmd", required=True)

    t = ds.add_parser("train", help="train on one or more `dataopen collect` datasets")
    t.add_argument("--config", help="TOML (configs/apollo_s.toml)")
    t.add_argument("--variant", default="s", choices=["t", "n", "s", "m"])
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="auto")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--epochs", type=int)
    t.add_argument("--batch-size", dest="batch_size", type=int)
    t.add_argument("--imgsz", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--workers", type=int)
    t.add_argument("--max-steps", dest="max_steps", type=int, help="stop after N optimizer steps (smoke tests)")
    t.set_defaults(fn=_cmd_train)

    e = ds.add_parser("eval", help="mAP@0.5, keypoint AP@OKS0.5, aim hit rate (all frames and the hard subset)")
    e.add_argument("--model", required=True, help=".pt checkpoint or exported .onnx (FP32 or INT8)")
    e.add_argument("--data", nargs="+", required=True)
    e.add_argument("--split", default="val")
    e.add_argument("--conf", type=float, default=0.25)
    e.add_argument("--device", default="cpu")
    e.add_argument("--limit", type=int)
    e.add_argument("--require", action="store_true", help="exit 1 unless map50 > 0.90, kp_ap_oks50 > 0.85 and aim_ap50 > 0.85")
    e.set_defaults(fn=_cmd_eval)

    x = ds.add_parser("export", help="checkpoint -> deploy ONNX (reparameterized, parity-checked against PyTorch)")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--input", default="float", choices=["float", "uint8"])
    x.add_argument("--opset", type=int, default=13)
    x.set_defaults(fn=_cmd_export)

    c = ds.add_parser("calib", help="build the INT8 calibration set (>= 500 stratified closed-loop frames)")
    c.add_argument("--data", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=500)
    c.add_argument("--size", type=int, default=640)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--hard-share", dest="hard_share", type=float, default=0.4)
    c.set_defaults(fn=_cmd_calib)

    q = ds.add_parser("quantize", help="INT8: ONNX Runtime QDQ proxy and/or the rknn-toolkit2 script")
    q.add_argument("--onnx", required=True)
    q.add_argument("--calib", required=True, help="directory made by `detector calib`")
    q.add_argument("--out", help="INT8 QDQ ONNX to write")
    q.add_argument("--method", default="percentile", choices=["minmax", "entropy", "percentile"])
    q.add_argument("--activation", default="uint8", choices=["uint8", "int8"])
    q.add_argument("--min-images", dest="min_images", type=int, default=500)
    q.add_argument("--exclude", nargs="*", help="Conv node names kept in float (mixed precision)")
    q.add_argument("--exclude-worst", dest="exclude_worst", type=int, default=0,
                   help="keep the N most quantization-sensitive convolutions in float (runs `detector sensitivity` first)")
    q.add_argument("--rknn-script", dest="rknn_script", help="write the rknn-toolkit2 conversion script here")
    q.add_argument("--rknn-out", dest="rknn_out")
    q.add_argument("--platform", default="rk3588")
    q.add_argument("--rknn-algorithm", dest="rknn_algorithm", default="mmse", choices=["normal", "mmse", "kl_divergence"])
    q.set_defaults(fn=_cmd_quantize)

    sn = ds.add_parser("sensitivity", help="rank convolutions by how much INT8 hurts them (mixed-precision candidates)")
    sn.add_argument("--onnx", required=True)
    sn.add_argument("--calib", required=True)
    sn.add_argument("--work", default="sensitivity_work")
    sn.add_argument("--n-images", dest="n_images", type=int, default=8)
    sn.add_argument("--top", type=int, default=15)
    sn.set_defaults(fn=_cmd_sensitivity)

    qa = ds.add_parser("qat", help="quantization-aware fine-tuning of a trained checkpoint -> QDQ ONNX")
    qa.add_argument("--ckpt", required=True)
    qa.add_argument("--data", nargs="+", required=True)
    qa.add_argument("--out", required=True)
    qa.add_argument("--epochs", type=int, default=5)
    qa.add_argument("--lr", type=float, default=1e-4)
    qa.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    qa.add_argument("--calib-batches", dest="calib_batches", type=int, default=8)
    qa.add_argument("--workers", type=int, default=2)
    qa.add_argument("--device", default="auto")
    qa.set_defaults(fn=_cmd_qat)

    b = ds.add_parser("bench", help="MACs, napkin NPU budget, host CPU latency, on-device RKNN benchmark script")
    b.add_argument("--ckpt")
    b.add_argument("--onnx")
    b.add_argument("--runs", type=int, default=100)
    b.add_argument("--rknn-script", dest="rknn_script")
    b.add_argument("--rknn-model", dest="rknn_model", default="apollo_int8.rknn")
    b.set_defaults(fn=_cmd_bench)

    h = ds.add_parser("header", help="write the C header of KeypointArray / ApolloDetection")
    h.add_argument("--out", default="apollo_detection.h")
    h.set_defaults(fn=_cmd_header)

    g = sub.add_parser("agreement", help="IoU agreement between two COCO annotation files (target: > 0.85)")
    g.add_argument("a")
    g.add_argument("b")
    g.add_argument("--thr", type=float, default=0.85)
    g.add_argument("--full", action="store_true", help="include the per-image review list")
    g.set_defaults(fn=_cmd_agreement)
