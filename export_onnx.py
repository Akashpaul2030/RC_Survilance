"""
Export HuggingFace AST model to ONNX with INT8 quantization.
Model: Akashpaul123/tiny-ast-mad-military-audio-classifier
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.exporters.onnx import main_export


MODEL_ID = "Akashpaul123/tiny-ast-mad-military-audio-classifier"
ONNX_DIR = "model_onnx"
QUANTIZED_DIR = "model_onnx_int8"

LABELS = [
    "Communication",
    "Footsteps",
    "Gunshot",
    "Shelling",
    "Vehicle",
    "Helicopter",
    "Fighter",
]


def export_to_onnx(output_dir: str = ONNX_DIR) -> str:
    """Export the HuggingFace model to ONNX format."""
    print(f"[1/3] Exporting {MODEL_ID} to ONNX...")
    # opset 17 is the minimum required for the AST attention operators;
    # higher opsets offer more fused kernels but may not be supported by
    # older ORT versions shipped on some edge distros
    main_export(
        model_name_or_path=MODEL_ID,
        output=output_dir,
        task="audio-classification",
        opset=17,
    )
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"  -> ONNX model saved to {onnx_path}")
    return output_dir


def quantize_int8(onnx_dir: str = ONNX_DIR, output_dir: str = QUANTIZED_DIR) -> str:
    """Apply INT8 dynamic quantization to the ONNX model."""
    print("[2/3] Applying INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    # avx2: targets AVX2 SIMD instructions; ORT falls back gracefully on ARM (Raspberry Pi)
    # is_static=False: dynamic quantization — weights are quantized offline, activations
    #   are quantized on-the-fly per inference; no calibration dataset required
    # per_channel=True: each output channel gets its own scale/zero-point, preserving
    #   accuracy better than a single per-tensor scale at a small memory cost
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
    print(f"  -> Quantized model saved to {output_dir}")
    return output_dir


def verify_model(quantized_dir: str = QUANTIZED_DIR):
    """Verify the quantized model runs correctly."""
    import onnxruntime as ort

    print("[3/3] Verifying quantized model...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

    # Create a dummy 1-second audio signal at 16kHz
    dummy_audio = np.random.randn(16000).astype(np.float32)
    inputs = feature_extractor(
        dummy_audio, sampling_rate=16000, return_tensors="np"
    )

    # optimum names the output "model_quantized.onnx"; fall back to "model.onnx"
    # when --skip-quantize was used and the FP32 model is being verified directly
    model_path = os.path.join(quantized_dir, "model_quantized.onnx")
    if not os.path.exists(model_path):
        model_path = os.path.join(quantized_dir, "model.onnx")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: inputs["input_values"]})
    logits = outputs[0]
    predicted = int(np.argmax(logits, axis=-1)[0])
    print(f"  -> Verification passed. Predicted class: {LABELS[predicted]}")

    # Print model size comparison
    fp32_path = os.path.join(ONNX_DIR, "model.onnx")
    if os.path.exists(fp32_path):
        fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
        int8_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  -> FP32 model size: {fp32_size:.1f} MB")
        print(f"  -> INT8 model size: {int8_size:.1f} MB")
        print(f"  -> Compression ratio: {fp32_size / int8_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Export AST model to ONNX with INT8 quantization")
    parser.add_argument("--onnx-dir", default=ONNX_DIR, help="Output directory for ONNX model")
    parser.add_argument("--quantized-dir", default=QUANTIZED_DIR, help="Output directory for quantized model")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip INT8 quantization")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification step")
    args = parser.parse_args()

    export_to_onnx(args.onnx_dir)

    if not args.skip_quantize:
        quantize_int8(args.onnx_dir, args.quantized_dir)

    if not args.skip_verify:
        target_dir = args.quantized_dir if not args.skip_quantize else args.onnx_dir
        verify_model(target_dir)

    print("\nDone! Model is ready for edge deployment.")


if __name__ == "__main__":
    main()
