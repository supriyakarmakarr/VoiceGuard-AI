"""
VoiceGuard AI - Model Training & Evaluation Pipeline
Trains both Baseline Ensemble ML and PyTorch Deep Spectrogram CNN models.
"""

import os
import sys
import json
import time

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dataset_generator import build_synthetic_dataset
from core.baseline_model import BaselineVoiceClassifier
from core.deep_learning_model import DeepLearningVoiceClassifier


def run_training_pipeline(num_samples_per_class: int = 40, epochs_deep: int = 25):
    print("=" * 70)
    print(">> SIH26104 -- VoiceGuard AI: Model Training & Benchmark Pipeline")
    print("=" * 70)

    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")

    # Step 1: Ensure Dataset is ready
    real_count = len(os.listdir(real_dir)) if os.path.exists(real_dir) else 0
    fake_count = len(os.listdir(fake_dir)) if os.path.exists(fake_dir) else 0

    if real_count < 10 or fake_count < 10:
        print("\n[Step 1/3] Generating balanced audio dataset (Human Real & Synthetic AI)...")
        build_synthetic_dataset(
            dataset_dir,
            num_real=num_samples_per_class,
            num_fake=num_samples_per_class,
            curated_demo_samples=True,
        )
    else:
        print(f"\n[Step 1/3] Found existing dataset with {real_count} real and {fake_count} fake samples.")

    # Step 2: Train Baseline ML Model
    print("\n[Step 2/3] Extracting tabular features & Training Baseline Classifier (RF + GB Ensemble)...")
    start_time = time.time()
    baseline_clf = BaselineVoiceClassifier()
    X, y, feature_names = baseline_clf.extract_dataset_features(real_dir, fake_dir)
    print(f"Extracted {X.shape[1]} tabular forensic features for {len(y)} audio samples.")

    baseline_metrics = baseline_clf.train(X, y, test_size=0.25, random_state=42)
    baseline_model_path = os.path.join(models_dir, "baseline_rf.pkl")
    baseline_clf.save(baseline_model_path)
    print(f"Baseline training completed in {time.time() - start_time:.2f}s")

    # Step 3: Train Deep Learning Spectrogram CNN
    print(f"\n[Step 3/3] Extracting Mel-Spectrograms & Training Deep 2D Spectrogram CNN ({epochs_deep} epochs)...")
    start_time = time.time()
    deep_clf = DeepLearningVoiceClassifier()
    specs, labels = deep_clf.prepare_dataset(real_dir, fake_dir)
    print(f"Prepared {len(specs)} Mel-Spectrogram tensors (Shape: 1x128x128).")

    deep_metrics = deep_clf.train(specs, labels, epochs=epochs_deep, batch_size=16, lr=0.001)
    deep_model_path = os.path.join(models_dir, "deep_cnn.pt")
    deep_clf.save(deep_model_path)
    print(f"Deep learning training completed in {time.time() - start_time:.2f}s")

    # Step 4: Save Model Metadata and Evaluation Summary
    metadata = {
        "project": "SIH26104 - VoiceGuard AI",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "num_real_samples": len([y_val for y_val in y if y_val == 0]),
            "num_fake_samples": len([y_val for y_val in y if y_val == 1]),
            "sample_rate_hz": 16000,
            "duration_sec": 4.0,
        },
        "baseline_model": {
            "type": "Random Forest + Gradient Boosting Soft-Voting Ensemble",
            "metrics": baseline_metrics,
            "weights_file": "baseline_rf.pkl",
        },
        "deep_learning_model": {
            "type": "ResNet-SE Spectrogram 2D-CNN (Channel Attention)",
            "metrics": deep_metrics,
            "weights_file": "deep_cnn.pt",
        },
        "risk_thresholds": {
            "low_risk_max": 30.0,
            "medium_risk_max": 70.0,
            "high_risk_min": 70.0,
        },
    }

    meta_path = os.path.join(models_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("🎯 Training Summary & Accuracy Report:")
    print(f"  • Baseline ML Accuracy:  {baseline_metrics['accuracy']*100:.2f}% (F1: {baseline_metrics['f1_score']*100:.2f}%)")
    print(f"  • Deep CNN Val Accuracy: {deep_metrics['validation_accuracy']*100:.2f}%")
    print(f"  • Model weights & metadata saved in: {models_dir}")
    print("=" * 70)
    return metadata


if __name__ == "__main__":
    run_training_pipeline(num_samples_per_class=40, epochs_deep=25)
