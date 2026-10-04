import json

import numpy as np

from dataopen.core.export import coco_dict, yolo_pose_lines
from dataopen.core.models import Annotation, FrameKind, FrameRecord
from dataopen.core.schema import HUMAN_13


def record():
    kp = np.zeros((13, 3))
    kp[0] = [100, 50, 2]
    kp[1] = [100, 80, 1]          # occluded: coords kept
    a = Annotation(0, kp, (80.0, 40.0, 40.0, 100.0))
    return FrameRecord("s000000_f0000", 0, 0, "train", "images/train/x.png", 200, 200,
                       FrameKind.POSITIVE, [a])


def test_coco_keypoints_structure():
    d = coco_dict([record(), FrameRecord("n", 0, 1, "train", "n.png", 200, 200, FrameKind.NEGATIVE, [])], HUMAN_13)
    ann = d["annotations"][0]
    assert len(ann["keypoints"]) == 39 and ann["num_keypoints"] == 2
    assert ann["keypoints"][:6] == [100.0, 50.0, 2, 100.0, 80.0, 1]
    assert ann["keypoints"][6:9] == [0, 0, 0]               # v == 0 -> x = y = 0
    assert ann["bbox"] == [80.0, 40.0, 40.0, 100.0] and ann["area"] == 4000.0
    assert len(d["images"]) == 2 and len(d["annotations"]) == 1   # negative image has no annotations
    cat = d["categories"][0]
    assert len(cat["keypoints"]) == 13 and all(1 <= i <= 13 for e in cat["skeleton"] for i in e)
    json.dumps(d)


def test_yolo_pose_line_is_normalized_and_has_xcycwh_then_triplets():
    (line,) = yolo_pose_lines(record())
    v = line.split()
    assert len(v) == 5 + 13 * 3 and v[0] == "0"
    xc, yc, w, h = map(float, v[1:5])
    assert (xc, yc, w, h) == (0.5, 0.45, 0.2, 0.5)
    assert v[5:8] == ["0.500000", "0.250000", "2"] and v[8:11] == ["0.500000", "0.400000", "1"]
    assert v[11:14] == ["0.000000", "0.000000", "0"]


def test_annotation_dict_roundtrip():
    r = record()
    r2 = FrameRecord.from_dict(json.loads(json.dumps(r.to_dict())))
    assert np.allclose(r2.annotations[0].keypoints, r.annotations[0].keypoints) and r2.kind is FrameKind.POSITIVE
