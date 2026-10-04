"""ApolloNet-Pose: an NPU-first, NMS-free keypoint detector and its training/export/quantization pipeline (docs/DETECTOR.md).

`layout`, `postprocess`, `structs`, `agreement` and `calibration_set` need only numpy (they run on the host next to the NPU
and inside the closed loop); `model`, `loss`, `data`, `train`, `evaluate`, `export`, `quantize` and `bench` need PyTorch
(`pip install 'dataopen[train]'`)."""
