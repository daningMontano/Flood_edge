# src/inference/detector.py
import time
import onnxruntime as ort
import numpy as np

class FloodDetectorEdge:
    def __init__(self, model_path, use_gpu=False):
        providers = ['CPUExecutionProvider']
        if use_gpu:
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor: np.ndarray):
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0]

    def predict_with_timing(self, input_tensor: np.ndarray):
        t0 = time.perf_counter()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        return outputs[0], inference_ms
