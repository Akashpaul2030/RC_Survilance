"""
ONNX Runtime inference wrapper for military audio classification.
Target: <200ms inference latency on edge devices.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import librosa
import onnxruntime as ort
from transformers import AutoFeatureExtractor


LABELS = [
    "Communication",
    "Footsteps",
    "Gunshot",
    "Shelling",
    "Vehicle",
    "Helicopter",
    "Fighter",
]

MODEL_ID = "Akashpaul123/tiny-ast-mad-military-audio-classifier"
DEFAULT_MODEL_DIR = "model_onnx_int8"
LATENCY_BUDGET_MS = 200.0
SAMPLE_RATE = 16000


@dataclass
class PredictionResult:
    label: str
    confidence: float
    all_scores: dict[str, float]
    latency_ms: float
    within_budget: bool


class AudioClassifier:
    """ONNX Runtime inference wrapper with latency tracking."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        latency_budget_ms: float = LATENCY_BUDGET_MS,
        num_threads: int = 4,
    ):
        self.latency_budget_ms = latency_budget_ms
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

        # Find the ONNX model file
        model_path = os.path.join(model_dir, "model_quantized.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No ONNX model found in {model_dir}. Run export_onnx.py first."
            )

        # Configure session for edge performance (tuned for Raspberry Pi 5)
        sess_options = ort.SessionOptions()
        # intra_op: threads used within a single op (e.g. matrix multiply); use all cores
        sess_options.intra_op_num_threads = num_threads
        # inter_op: threads used to run independent ops in parallel; keep at 1 to
        # avoid thread-spawn overhead on constrained hardware — sequential is faster here
        sess_options.inter_op_num_threads = 1
        # ORT_ENABLE_ALL applies constant folding, operator fusion, and memory layout
        # optimizations at load time so each inference call is as fast as possible
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Sequential execution avoids the scheduler overhead of the parallel executor;
        # beneficial on single-board computers where context-switching is expensive
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Warmup run
        self._warmup()

    def _warmup(self):
        """Run a dummy inference to warm up the model."""
        dummy = np.random.randn(SAMPLE_RATE).astype(np.float32)
        self.predict(dummy)

    def preprocess(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Preprocess audio to model input features."""
        # Resample if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
        )
        return inputs["input_values"]

    def predict(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE
    ) -> PredictionResult:
        """Run inference on audio data. Returns prediction with latency info."""
        t_start = time.perf_counter()

        # Preprocess
        input_values = self.preprocess(audio, sr)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_values})
        logits = outputs[0][0]

        # Numerically stable softmax: subtracting max(logits) before exp prevents
        # overflow when any logit is large, without changing the output distribution
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        predicted_idx = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - t_start) * 1000

        return PredictionResult(
            label=LABELS[predicted_idx],
            confidence=float(probs[predicted_idx]),
            all_scores={LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
            latency_ms=round(latency_ms, 2),
            within_budget=latency_ms <= self.latency_budget_ms,
        )

    def predict_file(self, file_path: str) -> PredictionResult:
        """Load audio file and run prediction."""
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return self.predict(audio, sr)

    def predict_bytes(self, audio_bytes: bytes) -> PredictionResult:
        """Run prediction on raw audio bytes (16-bit PCM, 16kHz, mono).

        Divides by 32768.0 (2^15) to convert int16 range [-32768, 32767]
        to float32 range [-1.0, ~1.0], matching the training data normalisation.
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.predict(audio, SAMPLE_RATE)
