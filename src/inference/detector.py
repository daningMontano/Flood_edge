# src/inference/detector.py
import os
import onnxruntime as ort
import numpy as np

class FloodDetectorEdge:
    def __init__(self, model_path: str):
        # Silenciar warnings (opcional)
        # ort.set_default_logger_severity(3)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        n = os.cpu_count() or 1  # núcleos lógicos disponibles
        # Paralelismo: usa todos los núcleos
        so.intra_op_num_threads = n
        so.inter_op_num_threads = max(1, n // 2)
        # Mejor para cargas paralelas en CPU
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        # Fuerza CPU
        self.session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: input_tensor})[0]