"""
VoiceGuard AI - Attack Condition & Robustness Verification Runner
Executes comprehensive stress testing:
1. Grandma False Positive Test (Adversarial Robustness on genuine distressed speech)
2. Variable Length Speech Test (1.0s, 2.0s, 3.0s, 4.0s)
3. Out-of-Distribution (OOD) Telephony Network Conditions (G.711, AMR-NB, Noise)
"""

import os
import sys
import json
import time
import numpy as np

# Ensure project root in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.baseline_model import BaselineVoiceClassifier
from core.deep_learning_model import DeepLearningVoiceClassifier
from core.risk_engine import RiskEngine
from core.calibrator import LearnedRiskCalibrator
from core.stress_test import StressTestSuite
from core.telephony_degradation import OODBenchmarkSuite, TelephonyDegradationPipeline
from core.dataset_corpora import generate_benchmark_multi_generator_dataset


def run_full_stress_test():
    print("=" * 78)
    print(">> VoiceGuard AI: Adversarial Stress-Testing & Robustness Verification")
    print("=" * 78)

    models_dir = os.path.join(PROJECT_ROOT, "models")
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    eval_dir = os.path.join(dataset_dir, "benchmark_eval")

    # Load models
    baseline_path = os.path.join(models_dir, "baseline_rf.pkl")
    deep_path = os.path.join(models_dir, "deep_cnn.pt")
    calibrator_path = os.path.join(models_dir, "risk_calibrator.pkl")

    base_model = BaselineVoiceClassifier(baseline_path) if os.path.exists(baseline_path) else None
    deep_model = DeepLearningVoiceClassifier(deep_path) if os.path.exists(deep_path) else None
    calibrator = LearnedRiskCalibrator(calibrator_path) if os.path.exists(calibrator_path) else None
    risk_engine = RiskEngine(calibrator=calibrator)

    if not base_model or not deep_model:
        print("[Error] Models not found. Please run scripts/train_models.py first.")
        return

    # Generate benchmark evaluation dataset if not exists
    print("\n[Step 1/3] Preparing Multi-Generator Benchmark Audio Corpora...")
    splits = generate_benchmark_multi_generator_dataset(eval_dir, samples_per_category=6)

    # Prepare sample pairs (audio, is_spoof)
    from core.audio_preprocessor import AudioPreprocessor
    prep = AudioPreprocessor()
    test_samples = []
    authentic_samples = []
    # Also include curated authentic human speech samples
    curated_dir = os.path.join(dataset_dir, "curated_samples")
    curated_real_files = ["sample_1_real_human_voice.wav", "sample_2_real_female_voice.wav"]
    for crf in curated_real_files:
        crf_path = os.path.join(curated_dir, crf)
        if os.path.exists(crf_path):
            try:
                audio_c, _ = prep.load_audio(crf_path)
                authentic_samples.append(audio_c)
                test_samples.append((audio_c, False))
            except Exception:
                pass

    for r in splits["test_id"] + splits["test_ood_generator"]:
        try:
            audio, _ = prep.load_audio(r.filepath)
            is_spoof = (r.label != "bonafide")
            test_samples.append((audio, is_spoof))
            if not is_spoof:
                authentic_samples.append(audio)
        except Exception as e:
            pass

    stress_suite = StressTestSuite(base_model, deep_model, risk_engine)
    ood_suite = OODBenchmarkSuite(base_model, deep_model, risk_engine)

    # 1. Grandma False Positive Test
    print("\n[Step 2/3] Executing 'Grandma False Positive & Adversarial Robustness Test'...")
    grandma_results = stress_suite.run_grandma_false_positive_test(authentic_samples)

    print("\n" + "-" * 78)
    print(f"{'DISTORTION SCENARIO':<32} | {'FPR (%)':<10} | {'MEAN RISK':<10} | {'STATUS':<10}")
    print("-" * 78)
    for scen, res in grandma_results["scenarios"].items():
        status_str = "PASS [OK]" if res["passed_safety_threshold"] else "FAIL [WARN]"
        print(f"{scen.replace('_', ' '):<32} | {res['false_positive_rate_pct']:<10.1f} | {res['mean_risk_score']:<10.1f} | {status_str}")
    print("-" * 78)
    print(f">> GRANDMA TEST VERDICT: {grandma_results['summary']}")

    # 2. Variable Length Speech Evaluation
    print("\n[Step 3/3] Running Variable Clip Length Stress Test (1.0s to 4.0s)...")
    dur_results = stress_suite.run_variable_length_test(test_samples, durations=[1.0, 2.0, 3.0, 4.0])

    print("\n" + "-" * 78)
    print(f"{'CLIP DURATION':<20} | {'ACCURACY':<12} | {'FPR (%)':<10} | {'FNR (%)':<10} | {'LATENCY (ms)':<12}")
    print("-" * 78)
    for dur_name, res in dur_results.items():
        print(f"{dur_name:<20} | {res['accuracy_pct']:>9.1f}% | {res['false_positive_rate_pct']:>8.1f}% | {res['false_negative_rate_pct']:>8.1f}% | {res['mean_latency_ms']:>10.1f} ms")
    print("-" * 78)

    # 3. OOD Telephony Conditions
    print("\nEvaluating Out-of-Distribution Telephony Degradations...")
    ood_results = ood_suite.benchmark_conditions(test_samples)
    print("\n" + "-" * 78)
    print(f"{'NETWORK / CODEC CONDITION':<32} | {'ACCURACY':<12} | {'FPR (%)':<10} | {'FNR (%)':<10}")
    print("-" * 78)
    for cond_name, res in ood_results.items():
        print(f"{cond_name.replace('_', ' '):<32} | {res['accuracy_pct']:>9.1f}% | {res['false_positive_rate_pct']:>8.1f}% | {res['false_negative_rate_pct']:>8.1f}%")
    print("-" * 78)

    # Save complete report
    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "grandma_false_positive_test": grandma_results,
        "variable_clip_duration": dur_results,
        "out_of_distribution_telephony": ood_results,
    }

    report_file = os.path.join(models_dir, "stress_test_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[Stress Test] Full forensic report saved to: {report_file}\n")


if __name__ == "__main__":
    run_full_stress_test()
