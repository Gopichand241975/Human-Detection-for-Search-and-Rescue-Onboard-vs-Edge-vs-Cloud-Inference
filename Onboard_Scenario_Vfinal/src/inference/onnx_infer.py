"""
onnx_infer.py
─────────────
Generator-based ONNX inference runner.

The model drives the loop — it signals readiness for the next image
by yielding after each inference, rather than the caller pushing
tensors in a for-loop.

Usage:
    runner = ONNXInfer()
    gen    = runner.run_generator()
    next(gen)                        # prime: model is now waiting for first tensor
    outputs = gen.send(input_tensor) # feed tensor → receive outputs
    outputs = gen.send(input_tensor) # feed next tensor → receive outputs
    gen.close()                      # done
"""

import onnxruntime as ort

from src.utils.config import MODEL_PATH


class ONNXInfer:
    def __init__(self):
        self.session    = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def run_generator(self):
        """
        Generator that yields after each inference, signalling it is ready
        for the next input tensor.

        Send a tensor → receive model outputs.
        Send None     → generator stops cleanly.
        """
        input_tensor = yield  # prime: wait for first tensor

        while input_tensor is not None:
            outputs      = self.session.run(None, {self.input_name: input_tensor})
            input_tensor = yield outputs  # return outputs, wait for next tensor
