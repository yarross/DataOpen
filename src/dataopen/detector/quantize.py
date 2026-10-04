"""INT8 post-training quantization: per-channel weights, per-tensor activations, calibrated on >= 500 closed-loop frames.

Two paths (docs/DETECTOR.md, section 6):
  quantize_ort   ONNX Runtime static quantization (QDQ). Runs anywhere, so accuracy loss can be measured on any machine and
                 the INT8 model can be dropped into the closed loop. It is a PROXY for the NPU, not the NPU's own numerics.
  rknn_convert   writes the rknn-toolkit2 script for the real thing (asymmetric 8-bit, per-channel, MMSE). Needs the Rockchip
                 toolkit (x86 Linux) and was NOT run in this repository's environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..quality.evaluators.decode import letterbox


def calibration_files(directory: str | Path, limit: Optional[int] = None) -> list[Path]:
    d = Path(directory)
    listing = d / "dataset.txt"
    if listing.exists():
        files = [Path(x.strip()) for x in listing.read_text().splitlines() if x.strip()]
        files = [f if f.is_absolute() else d / f for f in files]
    else:
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return files[:limit] if limit else files


class _Reader:
    """onnxruntime CalibrationDataReader over image files, in the model's own input convention."""

    def __init__(self, session_path: str, files: Sequence[Path]) -> None:
        import onnxruntime as ort
        from .data import load_image
        sess = ort.InferenceSession(session_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        self.name = inp.name
        self.uint8 = "uint8" in inp.type
        shape = list(inp.shape)
        self.nhwc = len(shape) == 4 and shape[-1] == 3 and shape[1] != 3
        h, w = (shape[1], shape[2]) if self.nhwc else (shape[2], shape[3])
        self.size, self.files, self.load = (int(w), int(h)), list(files), load_image
        self._it = iter(self.files)

    def get_next(self):
        for f in self._it:
            img, _ = letterbox(self.load(f), self.size[0], self.size[1])
            t = img if self.nhwc else img.transpose(2, 0, 1)
            t = np.ascontiguousarray(t, dtype=np.uint8 if self.uint8 else np.float32)
            return {self.name: t[None]}
        return None

    def rewind(self) -> None:
        self._it = iter(self.files)


def quantize_ort(fp32: str | Path, out: str | Path, calib: Iterable[Path], method: str = "percentile",
                 activation: str = "uint8", min_images: int = 500, nodes_to_exclude: Optional[Sequence[str]] = None,
                 nodes_to_quantize: Optional[Sequence[str]] = None) -> dict:
    """Static QDQ quantization. `method`: minmax | entropy | percentile. `nodes_to_exclude` keeps those nodes in float (mixed
    precision: see `layer_sensitivity`). Returns a report (and warns when < min_images)."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    files = list(calib)
    if not files:
        raise ValueError("empty calibration set")
    methods = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy, "percentile": CalibrationMethod.Percentile}
    if method not in methods or activation not in ("uint8", "int8"):
        raise ValueError("method: minmax|entropy|percentile; activation: uint8|int8")
    reader = _Reader(str(fp32), files)
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
    quantize_static(str(fp32), str(out), reader, quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8 if activation == "uint8" else QuantType.QInt8,
                    calibrate_method=methods[method], extra_options=extra,
                    nodes_to_exclude=list(nodes_to_exclude or []), nodes_to_quantize=list(nodes_to_quantize) if nodes_to_quantize else None)
    rep = {"int8": str(out), "nodes_kept_in_float": list(nodes_to_exclude or []), "calibration_images": len(files),
           "method": method, "activation": activation,
           "weights": "int8 per-channel", "activations": f"{activation} per-tensor",
           "size_mb": round(Path(out).stat().st_size / 1e6, 3)}
    if len(files) < min_images:
        rep["warning"] = f"only {len(files)} calibration images; use >= {min_images} representative closed-loop frames"
    return rep


def compare_outputs(fp32: str | Path, int8: str | Path, files: Sequence[Path], n: int = 32) -> dict:
    """Per-output cosine similarity and relative error between FP32 and INT8 on real frames: the first thing to read when
    accuracy drops (the outputs with the lowest similarity point at the layers to keep in higher precision)."""
    import onnxruntime as ort
    reader = _Reader(str(fp32), files[:n])
    a = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    names = [o.name for o in a.get_outputs()]
    cos = {k: [] for k in names}
    rel = {k: [] for k in names}
    while (feed := reader.get_next()) is not None:
        for k, x, y in zip(names, a.run(None, feed), b.run(None, feed)):
            x, y = x.reshape(-1).astype(np.float64), y.reshape(-1).astype(np.float64)
            cos[k].append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
            rel[k].append(float(np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-12)))
    return {k: {"cosine": round(float(np.mean(cos[k])), 5), "rel_error": round(float(np.mean(rel[k])), 5)} for k in names}


RKNN_SCRIPT = '''"""Generated by `dataopen detector quantize --rknn`: float ONNX -> INT8 .rknn for {platform}.
Run on an x86 Linux host with rknn-toolkit2 installed (NOT run in the dataopen repository).
    python {name}
"""
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(
    target_platform="{platform}",
    mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],     # the 1/255 scale is folded into the first conv
    quantized_dtype="asymmetric_quantized-8",           # int8 weights and activations
    quantized_method="channel",                         # PER-CHANNEL weights; activations are per-tensor
    quantized_algorithm="{algorithm}",                  # mmse is slower and usually more accurate than normal
    optimization_level=3,
    output_tensor_type=None,
)
assert rknn.load_onnx(model="{onnx}") == 0
# dataset.txt: one calibration image path per line (>= 500 closed-loop frames, letterboxed to {w}x{h})
assert rknn.build(do_quantization=True, dataset="{dataset}") == 0
# layer-wise cosine similarity vs the float model: look for layers below ~0.98
rknn.accuracy_analysis(inputs=["{sample}"], output_dir="rknn_accuracy")
assert rknn.export_rknn("{out}") == 0
rknn.release()
'''


def rknn_convert(onnx_path: str | Path, dataset_txt: str | Path, out_rknn: str | Path, script: str | Path,
                 platform: str = "rk3588", algorithm: str = "mmse", size: tuple[int, int] = (640, 640),
                 sample: str = "sample.png") -> Path:
    """Write the conversion script (it cannot run here: rknn-toolkit2 is Rockchip's, x86 Linux only)."""
    script = Path(script)
    script.write_text(RKNN_SCRIPT.format(platform=platform, algorithm=algorithm, onnx=onnx_path, dataset=dataset_txt, out=out_rknn,
                                         name=script.name, w=size[0], h=size[1], sample=sample))
    return script


def layer_sensitivity(fp32: str | Path, files: Sequence[Path], work_dir: str | Path, n_images: int = 8, top: int = 0) -> list[dict]:
    """Which convolutions hurt when quantized: quantize ONE Conv at a time (min-max calibration, `n_images` frames) and measure the
    cosine similarity of the model outputs against FP32. Sorted worst first; the worst few are the candidates for
    `quantize_ort(nodes_to_exclude=...)` (and for INT16/FP16 in RKNN's hybrid quantization). Costs one quantization per Conv."""
    import onnx
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    convs = [n.name for n in onnx.load(str(fp32)).graph.node if n.op_type == "Conv"]
    files = list(files)[:n_images]
    rows = []
    for name in convs:
        q = work / "one.onnx"
        quantize_ort(fp32, q, files, "minmax", "uint8", min_images=0, nodes_to_quantize=[name])
        sim = compare_outputs(fp32, q, files, n_images)
        rows.append({"node": name, "min_cosine": min(v["cosine"] for v in sim.values()),
                     "mean_rel_error": round(float(np.mean([v["rel_error"] for v in sim.values()])), 5)})
    rows.sort(key=lambda r: r["min_cosine"])
    return rows[:top] if top else rows
