"""
VoiceGuard AI - Target Deployment Latency Benchmarking Suite
Profiles real-world CPU deployment constraints (e.g. Render free tier / 0.5-1.0 vCPU):
1. Cold-Start Latency (Process start, import overhead, weights deserialization from disk)
2. First-Inference Latency (Memory caching, PyTorch tensor graph instantiation)
3. Steady-State Warm Latency (Mean, P50, P95, P99 across repeated inferences)
4. Granular Component Breakdown (Decode -> Preprocess -> Spectrogram -> CNN -> Baseline -> Calibrator)
5. FP32 vs Dynamic INT8 Quantization Speedup Comparison
"""

import os
import sys
import time
import json
import numpy as np
import torch

# Ensure project root in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.audio_preprocessor import AudioPreprocessor
from core.feature_extractor import FeatureExtractor
from core.baseline_model import BaselineVoiceClassifier
from core.deep_learning_model import DeepLearningVoiceClassifier
from core.risk_engine import RiskEngine
from core.calibrator import LearnedRiskCalibrator


def run_latency_benchmark(num_iterations: int = 25) -> dict:
    print("=" * 75)
    print(">> VoiceGuard AI: Deployment Latency & CPU Profiling Benchmark")
    print(f">> PyTorch Version: {torch.__version__} | Hardware Device: CPU")
    print("=" * 75)

    models_dir = os.path.join(PROJECT_ROOT, "models")
    deep_path = os.path.join(models_dir, "deep_cnn.pt")
    deep_q_path = os.path.join(models_dir, "deep_cnn_quantized.pt")
    baseline_path = os.path.join(models_dir, "baseline_rf.pkl")
    calibrator_path = os.path.join(models_dir, "risk_calibrator.pkl")

    # 1. Measure Cold-Start Latency (Model loading from disk)
    t_load_start = time.time()
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()

    base_model = BaselineVoiceClassifier(baseline_path) if os.path.exists(baseline_path) else None
    deep_model = DeepLearningVoiceClassifier(deep_path, device="cpu") if os.path.exists(deep_path) else None
    calibrator = LearnedRiskCalibrator(calibrator_path) if os.path.exists(calibrator_path) else None
    risk_engine = RiskEngine(calibrator=calibrator)

    cold_start_load_ms = (time.time() - t_load_start) * 1000.0
    print(f"\n[1/5] Cold-Start Model Deserialization Latency: {cold_start_load_ms:.2f} ms")

    # Test Audio Signal (3.0s speech)
    sr = 16000
    duration = 3.0
    t_axis = np.linspace(0, duration, int(sr * duration), endpoint=False)
    test_audio = (0.6 * np.sin(2 * np.pi * 180 * t_axis) + 0.2 * np.sin(2 * np.pi * 540 * t_axis)).astype(np.float32)

    # 2. Measure First-Inference Latency (Cold Execution)
    t_first_start = time.time()
    proc_first = preprocessor.process(test_audio)
    if deep_model:
        _ = deep_model.predict(proc_first["audio"])
    if base_model:
        _ = base_model.predict(proc_first["audio"])
    first_inference_ms = (time.time() - t_first_start) * 1000.0
    print(f"[2/5] First-Inference Latency (Cold Cache): {first_inference_ms:.2f} ms")

    # 3. Component-by-Component Timing
    print(f"\n[3/5] Profiling Component Breakdown ({num_iterations} iterations)...")
    timings = {
        "preprocessing": [],
        "spectrogram_extraction": [],
        "tabular_features": [],
        "deep_cnn_inference": [],
        "baseline_ml_inference": [],
        "forensic_signals": [],
        "risk_calibration": [],
        "total_end_to_end": [],
    }

    for _ in range(num_iterations):
        t_total = time.time()

        # Step A: Preprocess
        t0 = time.time()
        proc = preprocessor.process(test_audio)
        audio = proc["audio"]
        raw_trimmed = proc["raw_trimmed"]
        dur = proc["duration_sec"]
        timings["preprocessing"].append((time.time() - t0) * 1000.0)

        # Step B: Spectrogram Extraction
        t0 = time.time()
        spec = feature_extractor.extract_mel_spectrogram(audio)
        timings["spectrogram_extraction"].append((time.time() - t0) * 1000.0)

        # Step C: Deep CNN Inference
        t0 = time.time()
        deep_res = deep_model.predict(audio) if deep_model else {"synthetic_probability": 0.5}
        timings["deep_cnn_inference"].append((time.time() - t0) * 1000.0)

        # Step D: Tabular Feature Extraction
        t0 = time.time()
        feats, _ = feature_extractor.extract_tabular_features(audio)
        timings["tabular_features"].append((time.time() - t0) * 1000.0)

        # Step E: Baseline ML Inference
        t0 = time.time()
        base_res = base_model.predict(audio) if base_model else {"synthetic_probability": 0.5}
        timings["baseline_ml_inference"].append((time.time() - t0) * 1000.0)

        # Step F: Forensic Signals
        t0 = time.time()
        sig = feature_extractor.compute_forensic_signals(raw_trimmed)
        timings["forensic_signals"].append((time.time() - t0) * 1000.0)

        # Step G: Calibrated Risk Fusion
        t0 = time.time()
        risk = risk_engine.evaluate(deep_res, base_res, sig, audio_duration=dur)
        timings["risk_calibration"].append((time.time() - t0) * 1000.0)

        timings["total_end_to_end"].append((time.time() - t_total) * 1000.0)

    # Compute statistics
    breakdown_summary = {}
    for stage, times in timings.items():
        breakdown_summary[stage] = {
            "mean_ms": round(float(np.mean(times)), 2),
            "median_ms": round(float(np.median(times)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "p99_ms": round(float(np.percentile(times, 99)), 2),
        }

    # 4. FP32 vs Dynamic INT8 Quantization Speedup Comparison
    quant_speedup = 1.0
    int8_mean_ms = None
    if deep_model and hasattr(deep_model, "quantize_for_cpu"):
        print("\n[4/5] Evaluating Dynamic INT8 Quantization Speedup...")
        try:
            q_model = deep_model.quantize_for_cpu()
            q_times = []
            spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to("cpu")
            with torch.no_grad():
                for _ in range(num_iterations):
                    t0 = time.time()
                    _ = q_model(spec_tensor)
                    q_times.append((time.time() - t0) * 1000.0)
            int8_mean_ms = round(float(np.mean(q_times)), 2)
            fp32_cnn_ms = breakdown_summary["deep_cnn_inference"]["mean_ms"]
            quant_speedup = round(fp32_cnn_ms / max(1e-3, int8_mean_ms), 2)
            print(f"FP32 CNN Latency: {fp32_cnn_ms:.2f} ms  -->  INT8 Quantized Latency: {int8_mean_ms:.2f} ms (Speedup: {quant_speedup}x)")
        except Exception as e:
            print(f"Quantization benchmark notice: {e}")

    # 5. Display Clean Benchmark Table
    print("\n" + "=" * 75)
    print(f"{'PIPELINE STAGE':<30} | {'MEAN (ms)':<10} | {'P95 (ms)':<10} | {'PCT TOTAL':<10}")
    print("-" * 75)
    total_mean = breakdown_summary["total_end_to_end"]["mean_ms"]
    for stage, metrics in breakdown_summary.items():
        if stage != "total_end_to_end":
            pct = round((metrics["mean_ms"] / max(1e-3, total_mean)) * 100.0, 1)
            print(f"{stage.replace('_', ' ').title():<30} | {metrics['mean_ms']:<10.2f} | {metrics['p95_ms']:<10.2f} | {pct:>8.1f}%")
    print("-" * 75)
    print(f"{'END-TO-END LATENCY':<30} | {total_mean:<10.2f} | {breakdown_summary['total_end_to_end']['p95_ms']:<10.2f} | {'100.0%':>9}")
    print("=" * 75)

    benchmark_data = {
        "device": "CPU",
        "torch_version": torch.__version__,
        "cold_start_load_ms": round(cold_start_load_ms, 2),
        "first_inference_ms": round(first_inference_ms, 2),
        "end_to_end_steady_state_ms": breakdown_summary["total_end_to_end"],
        "breakdown": breakdown_summary,
        "quantization": {
            "fp32_mean_ms": breakdown_summary["deep_cnn_inference"]["mean_ms"],
            "int8_mean_ms": int8_mean_ms,
            "speedup_factor": quant_speedup,
        },
        "target_recommendation": {
            "deployment_target": "Render Free / 1 vCPU Container",
            "practical_expected_latency_ms": round(total_mean * 1.3, 1),
            "meets_realtime_budget_50ms": (total_mean < 50.0),
        },
    }

    # Save benchmark report to models/benchmark_latency.json
    out_path = os.path.join(models_dir, "benchmark_latency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[Latency Benchmark] Saved report to: {out_path}\n")

    return benchmark_data


if __name__ == "__main__":
    run_latency_benchmark()
