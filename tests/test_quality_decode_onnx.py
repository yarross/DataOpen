import numpy as np
import pytest

from dataopen.quality.evaluators.decode import (KeypointMap, decode_dfine, decode_yolov8_pose, letterbox, nms,
                                                resize_bilinear)

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from dataopen.quality.evaluators.onnx import OnnxEvaluator  # noqa: E402

W, H = 1280, 720                          # original image
IN = 640                                  # model input


def coco_person(x, y, w, h):
    """17 COCO keypoints for a person in box (x, y, w, h), original pixels."""
    rel = [(.5, .03), (.46, .02), (.54, .02), (.42, .04), (.58, .04), (.3, .2), (.7, .2), (.2, .38), (.8, .38), (.15, .5),
           (.85, .5), (.38, .52), (.62, .52), (.38, .74), (.62, .74), (.38, .97), (.62, .97)]
    return np.array([[x + rx * w, y + ry * h, 0.9] for rx, ry in rel])


def to_input(xy, scale=0.5, pad_x=0.0, pad_y=140.0):
    return (xy[0] * scale + pad_x, xy[1] * scale + pad_y)


def yolo_tensor(dets):
    """dets: list of (bbox_xywh_original, score, kp17_original) -> (1, 56, N) in letterboxed input coordinates."""
    cols = []
    for (x, y, w, h), score, kp in dets:
        cx, cy = to_input((x + w / 2, y + h / 2))
        col = [cx, cy, w * 0.5, h * 0.5, score]
        for kx, ky, kc in kp:
            ix, iy = to_input((kx, ky))
            col += [ix, iy, kc]
        cols.append(col)
    return np.array(cols, dtype=np.float32).T[None]


def test_letterbox_geometry_and_decode_roundtrip_with_nms_and_threshold():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    lb_img, lb = letterbox(img, IN, IN)
    assert lb_img.shape == (IN, IN, 3) and lb.scale == 0.5 and (lb.pad_x, lb.pad_y) == (0, 140)
    box = (400.0, 200.0, 100.0, 300.0)
    kp = coco_person(*box)
    raw = yolo_tensor([(box, 0.9, kp), ((402.0, 202.0, 100.0, 298.0), 0.6, kp),      # duplicate: NMS removes it
                       ((900.0, 100.0, 80.0, 200.0), 0.01, coco_person(900, 100, 80, 200))])   # below threshold
    preds = decode_yolov8_pose(raw, lb, conf_thr=0.05, iou_thr=0.7)
    assert len(preds) == 1 and preds[0].score == pytest.approx(0.9, abs=1e-6)
    assert preds[0].bbox == pytest.approx(box, abs=1e-3)
    assert preds[0].keypoints.shape == (17, 3) and np.allclose(preds[0].keypoints[:, :2], kp[:, :2], atol=1e-3)
    both = decode_yolov8_pose(raw, lb, conf_thr=0.05, iou_thr=0.99)                   # lenient NMS keeps both
    assert len(both) == 2


def test_transposed_output_layout_and_bad_channel_count():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    _, lb = letterbox(img, IN, IN)
    box = (400.0, 200.0, 100.0, 300.0)
    raw = yolo_tensor([(box, 0.9, coco_person(*box))])
    assert len(decode_yolov8_pose(raw.transpose(0, 2, 1), lb, 0.05, 0.7)) == 1          # (1, N, C)
    with pytest.raises(ValueError, match="unexpected YOLOv8-pose output"):
        decode_yolov8_pose(np.zeros((1, 10, 5), dtype=np.float32), lb, 0.05, 0.7)


def test_coco17_to_human13_mapping_geometry_and_confidence():
    box = (400.0, 200.0, 100.0, 300.0)
    kp = coco_person(*box)
    kp[11, 2] = 0.2                                                    # one hip is uncertain
    m = KeypointMap.coco17_to_human13().apply(kp)
    assert m.shape == (13, 3)
    assert np.allclose(m[0, :2], kp[[3, 4], :2].mean(axis=0))          # head = ears
    assert np.allclose(m[1, :2], kp[[5, 6], :2].mean(axis=0))          # neck = mid-shoulders
    assert np.allclose(m[8, :2], kp[[11, 12], :2].mean(axis=0)) and m[8, 2] == pytest.approx(0.2)   # pelvis conf = min
    assert np.allclose(m[2, :2], kp[5, :2]) and np.allclose(m[12, :2], kp[16, :2])


def test_nms_and_resize_fallback_agree_with_expectations():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]
    img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    assert resize_bilinear(img, 30, 20).shape == (20, 30, 3)
    assert np.array_equal(resize_bilinear(np.full((8, 8, 3), 77, np.uint8), 16, 16), np.full((16, 16, 3), 77, np.uint8))


def const_model(path, outputs, inputs):
    """ONNX model returning fixed tensors; inputs are declared (and consumed by a zero-multiply) like a real model."""
    nodes, inits, outs = [], [], []
    ins = [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs]
    first = inputs[0][0]
    nodes.append(helper.make_node("ReduceMean", [first], ["m"], keepdims=0))
    inits.append(numpy_helper.from_array(np.zeros((1,), np.float32), "zero"))
    nodes.append(helper.make_node("Mul", ["m", "zero"], ["z"]))
    for name, arr in outputs:
        inits.append(numpy_helper.from_array(arr, name + "_c"))
        if arr.dtype == np.float32:
            nodes.append(helper.make_node("Add", [name + "_c", "z"], [name]))
        else:
            nodes.append(helper.make_node("Identity", [name + "_c"], [name]))
        outs.append(helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(arr.dtype), list(arr.shape)))
    g = helper.make_graph(nodes, "g", ins, outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_onnx_yolov8_pose_end_to_end_with_a_real_session(tmp_path):
    box = (400.0, 200.0, 100.0, 300.0)
    raw = yolo_tensor([(box, 0.9, coco_person(*box))])
    p = tmp_path / "yolo.onnx"
    const_model(p, [("output0", raw)], [("images", TensorProto.FLOAT, [1, 3, IN, IN])])
    ev = OnnxEvaluator(str(p), "yolov8_pose", (IN, IN), keypoint_map=KeypointMap.coco17_to_human13())
    ev.warmup()
    (preds,) = ev.predict([np.zeros((H, W, 3), dtype=np.uint8)])
    assert len(preds) == 1 and preds[0].keypoints.shape == (13, 3) and preds[0].bbox == pytest.approx(box, abs=1e-2)
    assert ev.has_keypoints and ev.name.startswith("onnx:yolov8_pose")
    # an image of another size maps back to ITS OWN coordinates (letterbox is per image)
    (small,) = ev.predict([np.zeros((360, 640, 3), dtype=np.uint8)])
    assert small[0].bbox[0] != preds[0].bbox[0]


def test_onnx_dfine_two_inputs_and_label_filter(tmp_path):
    labels = np.array([[0, 0, 5]], dtype=np.int64)
    boxes = np.array([[[400, 200, 500, 500], [10, 10, 50, 60], [0, 0, 5, 5]]], dtype=np.float32)
    scores = np.array([[0.9, 0.2, 0.99]], dtype=np.float32)
    p = tmp_path / "dfine.onnx"
    const_model(p, [("labels", labels), ("boxes", boxes), ("scores", scores)],
                [("images", TensorProto.FLOAT, [1, 3, IN, IN]), ("orig_target_sizes", TensorProto.INT64, [1, 2])])
    ev = OnnxEvaluator(str(p), "dfine", (IN, IN), conf_thr=0.5)
    (preds,) = ev.predict([np.zeros((H, W, 3), dtype=np.uint8)])
    assert len(preds) == 1 and preds[0].bbox == (400.0, 200.0, 100.0, 300.0) and preds[0].keypoints is None
    assert not ev.has_keypoints
    assert decode_dfine(labels, boxes, scores, 0.1) and len(decode_dfine(labels, boxes, scores, 0.1)) == 2


def test_bad_arguments_are_rejected_early(tmp_path):
    with pytest.raises(ValueError, match="fmt"):
        OnnxEvaluator("x.onnx", "bogus")
    p = tmp_path / "m.onnx"
    const_model(p, [("output0", np.zeros((1, 56, 3), np.float32))], [("images", TensorProto.FLOAT, [1, 3, IN, IN])])
    with pytest.raises(ValueError, match="device"):
        OnnxEvaluator(str(p), "yolov8_pose", device="tpu")
    ev = OnnxEvaluator(str(p), "yolov8_pose", device="cuda")        # not available here: falls back to CPU with a warning
    assert ev.predict([np.zeros((H, W, 3), np.uint8)]) == [[]]
