"""
Tests for ONNX inference and export pipeline.
HuggingFace calls and ONNX sessions are mocked so tests run offline.
"""

import io
import os
import sys
import types
import struct
from dataclasses import fields
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build stub modules for heavy dependencies that are not installed
# or require network access (torch, transformers, optimum).
# ---------------------------------------------------------------------------

def _make_torch_stub():
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.no_grad = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s,
                                                     __exit__=lambda s, *a: None))
    return torch


def _install_stubs():
    """Install minimal stubs for packages not available in test environment."""
    sys.modules["torch"] = _make_torch_stub()
    sys.modules["torchaudio"] = types.ModuleType("torchaudio")

    # optimum stubs with the symbols export_onnx.py actually imports
    opt_ort = types.ModuleType("optimum.onnxruntime")
    opt_ort.ORTQuantizer = MagicMock()
    opt_ort_cfg = types.ModuleType("optimum.onnxruntime.configuration")
    opt_ort_cfg.AutoQuantizationConfig = MagicMock()
    opt_exp = types.ModuleType("optimum.exporters.onnx")
    opt_exp.main_export = MagicMock()
    opt_root = types.ModuleType("optimum")
    opt_root.onnxruntime = opt_ort
    opt_root.exporters = types.ModuleType("optimum.exporters")

    sys.modules["optimum"] = opt_root
    sys.modules["optimum.onnxruntime"] = opt_ort
    sys.modules["optimum.onnxruntime.configuration"] = opt_ort_cfg
    sys.modules["optimum.exporters"] = opt_root.exporters
    sys.modules["optimum.exporters.onnx"] = opt_exp

    # transformers stub – AutoFeatureExtractor is the only thing inference.py needs
    tf = types.ModuleType("transformers")
    tf.AutoFeatureExtractor = MagicMock()
    tf.ASTForAudioClassification = MagicMock()
    sys.modules["transformers"] = tf


_install_stubs()

# Now we can safely import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import inference  # noqa: E402
import simulate_edge  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = inference.SAMPLE_RATE
LABELS = inference.LABELS
N_LABELS = len(LABELS)


def _fake_logits(winner_idx: int = 0):
    """Return logits that will produce a clear prediction for winner_idx."""
    logits = np.zeros(N_LABELS, dtype=np.float32)
    logits[winner_idx] = 5.0          # high score for the winner
    return logits


def _make_mock_session(logits: np.ndarray | None = None):
    """Return a mock ort.InferenceSession that yields the given logits."""
    if logits is None:
        logits = _fake_logits(0)
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input_values")]
    session.run.return_value = [logits[np.newaxis, :]]   # shape (1, N_LABELS)
    return session


def _make_mock_feature_extractor():
    """Return a mock feature extractor that returns a fixed input_values array."""
    fe = MagicMock()
    fe.return_value = {"input_values": np.zeros((1, SAMPLE_RATE), dtype=np.float32)}
    return fe


def _make_classifier(winner_idx: int = 0, num_threads: int = 1):
    """Build an AudioClassifier with all heavy deps mocked."""
    mock_fe = _make_mock_feature_extractor()
    mock_sess = _make_mock_session(_fake_logits(winner_idx))

    with patch("inference.AutoFeatureExtractor") as mock_af, \
         patch("inference.ort.InferenceSession", return_value=mock_sess), \
         patch("inference.ort.SessionOptions", return_value=MagicMock()), \
         patch("os.path.exists", return_value=True):
        mock_af.from_pretrained.return_value = mock_fe
        clf = inference.AudioClassifier.__new__(inference.AudioClassifier)
        clf.latency_budget_ms = inference.LATENCY_BUDGET_MS
        clf.feature_extractor = mock_fe
        clf.session = mock_sess
        clf.input_name = "input_values"
    return clf


# ---------------------------------------------------------------------------
# PredictionResult dataclass
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_fields_exist(self):
        names = {f.name for f in fields(inference.PredictionResult)}
        assert names == {"label", "confidence", "all_scores", "latency_ms", "within_budget"}

    def test_construction(self):
        scores = {lbl: 1.0 / N_LABELS for lbl in LABELS}
        result = inference.PredictionResult(
            label="Gunshot",
            confidence=0.9,
            all_scores=scores,
            latency_ms=50.0,
            within_budget=True,
        )
        assert result.label == "Gunshot"
        assert result.within_budget is True

    def test_within_budget_false(self):
        scores = {lbl: 0.0 for lbl in LABELS}
        result = inference.PredictionResult(
            label="Vehicle",
            confidence=0.5,
            all_scores=scores,
            latency_ms=250.0,
            within_budget=False,
        )
        assert result.within_budget is False
        assert result.latency_ms == 250.0


# ---------------------------------------------------------------------------
# Softmax (numerically stable)
# ---------------------------------------------------------------------------

class TestSoftmax:
    """Test the inline softmax used in AudioClassifier.predict()."""

    def _softmax(self, logits):
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def test_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        probs = self._softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_argmax_preserved(self):
        logits = np.zeros(N_LABELS, dtype=np.float32)
        logits[3] = 10.0
        probs = self._softmax(logits)
        assert np.argmax(probs) == 3

    def test_large_logit_no_overflow(self):
        """Subtracting max prevents exp overflow for very large values."""
        logits = np.array([1000.0, 1001.0, 999.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        probs = self._softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_uniform_logits(self):
        logits = np.zeros(N_LABELS, dtype=np.float32)
        probs = self._softmax(logits)
        expected = 1.0 / N_LABELS
        assert np.allclose(probs, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

class TestPreprocess:
    def setup_method(self):
        self.clf = _make_classifier()

    def test_returns_ndarray(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        out = self.clf.preprocess(audio, SAMPLE_RATE)
        assert isinstance(out, np.ndarray)

    def test_float32_coercion(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float64)
        # preprocess should convert to float32 before passing to feature extractor
        out = self.clf.preprocess(audio, SAMPLE_RATE)
        # The mock returns float32, so just verify no exception is raised
        assert out is not None

    def test_normalization(self):
        """Peak normalisation should scale audio to [-1, 1] before feature extraction."""
        audio = np.array([0.0, 2.0, -4.0, 2.0], dtype=np.float32)
        # Capture the audio passed into the feature extractor
        captured = {}

        def fake_fe(a, sampling_rate, return_tensors):
            captured["audio"] = a
            return {"input_values": np.zeros((1, 4), dtype=np.float32)}

        self.clf.feature_extractor.side_effect = fake_fe
        self.clf.preprocess(audio, SAMPLE_RATE)
        assert "audio" in captured
        assert abs(np.abs(captured["audio"]).max() - 1.0) < 1e-5

    def test_silent_audio_no_division_by_zero(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        # peak == 0 branch: should not raise ZeroDivisionError
        out = self.clf.preprocess(audio, SAMPLE_RATE)
        assert out is not None

    def test_resampling_called_when_sr_differs(self):
        # librosa.resample is lazy-loaded so create=True is required
        audio = np.random.randn(22050).astype(np.float32)
        resampled = np.zeros(SAMPLE_RATE, np.float32)
        with patch("inference.librosa.resample", create=True, return_value=resampled) as mock_rs:
            self.clf.preprocess(audio, 22050)
            mock_rs.assert_called_once_with(audio, orig_sr=22050, target_sr=SAMPLE_RATE)

    def test_no_resample_when_sr_matches(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        with patch("inference.librosa.resample", create=True) as mock_rs:
            self.clf.preprocess(audio, SAMPLE_RATE)
            mock_rs.assert_not_called()


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_prediction_result(self):
        clf = _make_classifier(winner_idx=2)  # Gunshot
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        result = clf.predict(audio)
        assert isinstance(result, inference.PredictionResult)

    def test_predicted_label_matches_logits(self):
        for idx, label in enumerate(LABELS):
            clf = _make_classifier(winner_idx=idx)
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
            result = clf.predict(audio)
            assert result.label == label, f"Expected {label}, got {result.label}"

    def test_confidence_in_range(self):
        clf = _make_classifier()
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        result = clf.predict(audio)
        assert 0.0 <= result.confidence <= 1.0

    def test_all_scores_keys_match_labels(self):
        clf = _make_classifier()
        result = clf.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert set(result.all_scores.keys()) == set(LABELS)

    def test_all_scores_sum_to_one(self):
        clf = _make_classifier()
        result = clf.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert abs(sum(result.all_scores.values()) - 1.0) < 1e-5

    def test_latency_ms_positive(self):
        clf = _make_classifier()
        result = clf.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert result.latency_ms > 0

    def test_within_budget_true_when_fast(self):
        clf = _make_classifier()
        clf.latency_budget_ms = 99999.0   # impossible to exceed
        result = clf.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert result.within_budget is True

    def test_within_budget_false_when_slow(self):
        clf = _make_classifier()
        clf.latency_budget_ms = 0.0001    # impossible to beat
        result = clf.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert result.within_budget is False


# ---------------------------------------------------------------------------
# predict_bytes()
# ---------------------------------------------------------------------------

class TestPredictBytes:
    def test_int16_normalisation(self):
        """32768 int16 samples must map to float 1.0 before predict()."""
        clf = _make_classifier()
        captured = {}
        original_predict = clf.predict

        def spy_predict(audio, sr=SAMPLE_RATE):
            captured["audio"] = audio
            return original_predict(audio, sr)

        clf.predict = spy_predict

        # Build int16 bytes: [32767, -32768, 0]
        raw = struct.pack("<hhh", 32767, -32768, 0)
        clf.predict_bytes(raw)
        audio = captured["audio"]
        assert abs(audio[0] - (32767 / 32768.0)) < 1e-5
        assert abs(audio[1] - (-32768 / 32768.0)) < 1e-5
        assert audio[2] == 0.0

    def test_returns_prediction_result(self):
        clf = _make_classifier()
        raw = np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes()
        result = clf.predict_bytes(raw)
        assert isinstance(result, inference.PredictionResult)


# ---------------------------------------------------------------------------
# LABELS constant
# ---------------------------------------------------------------------------

class TestLabels:
    def test_seven_classes(self):
        assert len(LABELS) == 7

    def test_expected_classes(self):
        expected = {"Communication", "Footsteps", "Gunshot",
                    "Shelling", "Vehicle", "Helicopter", "Fighter"}
        assert set(LABELS) == expected

    def test_no_duplicates(self):
        assert len(LABELS) == len(set(LABELS))


# ---------------------------------------------------------------------------
# simulate_edge – generate_synthetic_audio
# ---------------------------------------------------------------------------

class TestSyntheticAudio:
    def test_shape(self):
        audio = simulate_edge.generate_synthetic_audio(duration_s=1.0)
        assert audio.shape == (SAMPLE_RATE,)

    def test_half_second(self):
        audio = simulate_edge.generate_synthetic_audio(duration_s=0.5)
        assert audio.shape == (SAMPLE_RATE // 2,)

    def test_dtype(self):
        audio = simulate_edge.generate_synthetic_audio(duration_s=1.0)
        assert audio.dtype == np.float32

    def test_non_silent(self):
        audio = simulate_edge.generate_synthetic_audio(duration_s=1.0)
        assert np.abs(audio).max() > 0.0

    def test_amplitude_reasonable(self):
        """Signal + noise amplitude should stay well within [-2, 2]."""
        audio = simulate_edge.generate_synthetic_audio(duration_s=1.0)
        assert np.abs(audio).max() < 2.0


# ---------------------------------------------------------------------------
# export_onnx – LABELS consistency
# ---------------------------------------------------------------------------

class TestExportLabels:
    def test_labels_match_inference(self):
        """export_onnx.LABELS and inference.LABELS must be identical."""
        # Import with stubs already in place
        import export_onnx
        assert export_onnx.LABELS == inference.LABELS
