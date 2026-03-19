"""
Batch test script simulating continuous surveillance loop.
Generates synthetic audio and measures inference performance under edge constraints.
"""

import argparse
import sys
import time
import statistics

import numpy as np

from inference import AudioClassifier, LABELS, SAMPLE_RATE


def generate_synthetic_audio(duration_s: float = 1.0) -> np.ndarray:
    """Generate synthetic audio signal for testing."""
    n_samples = int(SAMPLE_RATE * duration_s)
    # Mix of frequencies to simulate different audio events
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    freq = np.random.choice([200, 500, 1000, 2000, 4000])
    signal = np.sin(2 * np.pi * freq * t) * 0.5
    noise = np.random.randn(n_samples).astype(np.float32) * 0.1
    return signal + noise


def run_surveillance_loop(
    num_iterations: int = 100,
    audio_duration_s: float = 1.0,
    model_dir: str = "model_onnx_int8",
    num_threads: int = 4,
):
    """Simulate continuous surveillance with batch inference."""
    print("=" * 60)
    print("EDGE DEPLOYMENT SIMULATION")
    print(f"  Model dir:       {model_dir}")
    print(f"  ORT threads:     {num_threads}")
    print(f"  Iterations:      {num_iterations}")
    print(f"  Audio duration:  {audio_duration_s}s")
    print("=" * 60)

    # Initialize classifier
    print("\nLoading model...")
    t0 = time.perf_counter()
    clf = AudioClassifier(model_dir=model_dir, num_threads=num_threads)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_time:.0f}ms\n")

    latencies = []
    predictions = {label: 0 for label in LABELS}
    budget_violations = 0

    print(f"Running {num_iterations} inference cycles...\n")

    for i in range(num_iterations):
        audio = generate_synthetic_audio(audio_duration_s)
        result = clf.predict(audio)

        latencies.append(result.latency_ms)
        predictions[result.label] += 1
        if not result.within_budget:
            budget_violations += 1

        # Progress indicator every 10 iterations
        if (i + 1) % 10 == 0:
            avg = statistics.mean(latencies[-10:])
            print(f"  [{i+1:4d}/{num_iterations}] Last 10 avg: {avg:.1f}ms | "
                  f"Predicted: {result.label} ({result.confidence:.2%})")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nLatency Statistics:")
    print(f"  Mean:    {statistics.mean(latencies):.1f}ms")
    print(f"  Median:  {statistics.median(latencies):.1f}ms")
    print(f"  P95:     {sorted(latencies)[int(0.95 * len(latencies))]:.1f}ms")
    print(f"  P99:     {sorted(latencies)[int(0.99 * len(latencies))]:.1f}ms")
    print(f"  Min:     {min(latencies):.1f}ms")
    print(f"  Max:     {max(latencies):.1f}ms")
    print(f"  Std:     {statistics.stdev(latencies):.1f}ms")

    print(f"\nBudget Compliance (< 200ms):")
    compliance = (num_iterations - budget_violations) / num_iterations * 100
    print(f"  Passed:  {num_iterations - budget_violations}/{num_iterations} ({compliance:.1f}%)")
    print(f"  Failed:  {budget_violations}/{num_iterations}")

    print(f"\nPrediction Distribution:")
    for label, count in sorted(predictions.items(), key=lambda x: -x[1]):
        bar = "#" * (count * 40 // num_iterations)
        print(f"  {label:15s}: {count:4d} ({count/num_iterations*100:5.1f}%) {bar}")

    print(f"\nThroughput: {num_iterations / sum(latencies) * 1000:.1f} inferences/sec")

    # Exit code based on budget compliance
    if compliance < 95.0:
        print(f"\n[WARN] Budget compliance below 95% threshold!")
        return 1
    print(f"\n[OK] Edge deployment simulation passed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Edge deployment simulation")
    parser.add_argument("-n", "--num-iterations", type=int, default=100,
                        help="Number of inference iterations (default: 100)")
    parser.add_argument("-d", "--duration", type=float, default=1.0,
                        help="Audio duration in seconds (default: 1.0)")
    parser.add_argument("--model-dir", type=str, default="model_onnx_int8",
                        help="Path to ONNX model directory")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of ORT inference threads (default: 4)")
    args = parser.parse_args()

    rc = run_surveillance_loop(
        num_iterations=args.num_iterations,
        audio_duration_s=args.duration,
        model_dir=args.model_dir,
        num_threads=args.threads,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
