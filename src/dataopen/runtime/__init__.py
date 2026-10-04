"""Inference runtime for the target SoC (docs/RUNTIME.md): frames in from a shared / DMA buffer, non-blocking inference on a model
backend (ONNX Runtime CPU/CUDA/TensorRT today, RKNN on the board), `KeypointArray` out on an IPC channel, metrics throughout.
The same `IModelEvaluator`s the closed loop uses (mock included) plug in as backends, and the runtime itself is an
`IModelEvaluator` (`RuntimeEvaluator`), so the closed loop can evaluate exactly the path that ships."""
