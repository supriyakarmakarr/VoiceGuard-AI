"""
VoiceGuard AI - Calibrated Risk Fusion & Logistic Meta-Learner
Replaces manual heuristic weights (50/30/20) with:
1. Learned Stacking Logistic Meta-Learner over deep, baseline, and forensic features
2. Probability Calibration via Platt Scaling (Logistic Sigmoid) and Isotonic Regression
3. Empirical Reliability Metrics: Brier Score & Expected Calibration Error (ECE)
Ensures that a 70% risk score corresponds to an empirical ~70% likelihood of synthesis.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, Tuple, Optional, List


class LearnedRiskCalibrator:
    """
    Learned meta-classifier and probability calibrator for multi-signal forensic risk fusion.
    """

    FEATURE_NAMES = [
        "deep_prob",
        "baseline_prob",
        "hf_anomaly",
        "prosody_anomaly",
        "spectral_cutoff",
        "spectral_flatness",
        "audio_duration_sec",
    ]

    def __init__(self, model_path: str = None, calibration_method: str = "sigmoid"):
        self.model_path = model_path
        self.calibration_method = calibration_method  # 'sigmoid' (Platt scaling) or 'isotonic'
        self.meta_learner: Optional[LogisticRegression] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.metrics: Dict[str, float] = {}
        self.is_fitted = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def extract_meta_features(
        self,
        deep_prob: float,
        baseline_prob: float,
        forensic_signals: dict,
        duration: float = 4.0,
    ) -> np.ndarray:
        """Converts raw detector outputs into a normalized meta-feature vector."""
        vec = np.array([
            float(deep_prob),
            float(baseline_prob),
            float(forensic_signals.get("hf_anomaly_score", 0.0)),
            float(forensic_signals.get("prosody_anomaly_score", 0.0)),
            float(forensic_signals.get("spectral_cutoff_score", 0.0)),
            float(forensic_signals.get("spectral_flatness_score", 0.0)),
            float(min(duration, 5.0) / 5.0),  # normalized duration
        ], dtype=np.float32)
        return vec.reshape(1, -1)

    def compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE):
        ECE = sum( (bin_size / N) * |bin_acc - bin_conf| )
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                bin_acc = np.mean(y_true[bin_mask])
                bin_conf = np.mean(y_prob[bin_mask])
                ece += (bin_size / n_total) * abs(bin_acc - bin_conf)

        return float(ece)

    def fit(
        self,
        X_meta: np.ndarray,
        y_true: np.ndarray,
        validation_split: float = 0.25,
        random_state: int = 42,
    ) -> Dict[str, any]:
        """
        Trains the meta-learner and fits Platt scaling probability calibration.
        """
        # Base Logistic Meta-Learner
        base_lr = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            random_state=random_state,
        )

        # Calibrated classifier (Platt Scaling)
        cv_folds = min(3, len(y_true) // 4)
        if cv_folds >= 2:
            self.calibrated_model = CalibratedClassifierCV(
                estimator=base_lr,
                method=self.calibration_method,
                cv=cv_folds,
            )
            self.calibrated_model.fit(X_meta, y_true)
            self.meta_learner = base_lr.fit(X_meta, y_true)
            calibrated_probs = self.calibrated_model.predict_proba(X_meta)[:, 1]
        else:
            base_lr.fit(X_meta, y_true)
            self.meta_learner = base_lr
            self.calibrated_model = None
            calibrated_probs = base_lr.predict_proba(X_meta)[:, 1]

        # Compute calibration metrics
        brier = float(brier_score_loss(y_true, calibrated_probs))
        ece = self.compute_ece(y_true, calibrated_probs, n_bins=8)
        uncalibrated_ece = self.compute_ece(y_true, X_meta[:, 0], n_bins=8)  # compared to raw cnn

        # Extract learned feature weights
        learned_weights = dict(zip(self.FEATURE_NAMES, [round(float(w), 4) for w in self.meta_learner.coef_[0]]))

        self.metrics = {
            "brier_score": round(brier, 4),
            "calibrated_ece": round(ece, 4),
            "uncalibrated_ece": round(uncalibrated_ece, 4),
            "ece_reduction_pct": round(max(0.0, (uncalibrated_ece - ece) / max(1e-4, uncalibrated_ece) * 100), 1),
            "learned_weights": learned_weights,
            "intercept": round(float(self.meta_learner.intercept_[0]), 4),
            "calibration_method": self.calibration_method,
            "train_samples": len(y_true),
        }

        self.is_fitted = True
        print(f"[Calibrator] Fitted Platt Scaler -> Brier: {brier:.4f}, ECE: {ece:.4f} (from {uncalibrated_ece:.4f})")
        print(f"[Calibrator] Learned Weights: {learned_weights}")
        return self.metrics

    def predict_calibrated_probability(
        self,
        deep_prob: float,
        baseline_prob: float,
        forensic_signals: dict,
        duration: float = 4.0,
    ) -> Tuple[float, float]:
        """
        Returns (calibrated_synthetic_probability, empirical_confidence).
        If not fitted, falls back gracefully to smoothed heuristic fusion.
        """
        if not self.is_fitted:
            # Fallback heuristic with conservative scaling
            raw = 0.50 * deep_prob + 0.30 * baseline_prob + 0.20 * forensic_signals.get("hf_anomaly_score", 0.0)
            return float(np.clip(raw, 0.02, 0.98)), 0.75

        meta_x = self.extract_meta_features(deep_prob, baseline_prob, forensic_signals, duration)

        if self.calibrated_model:
            prob = float(self.calibrated_model.predict_proba(meta_x)[0, 1])
        else:
            prob = float(self.meta_learner.predict_proba(meta_x)[0, 1])

        # Clamped calibrated probability
        calibrated_p = float(np.clip(prob, 0.01, 0.99))
        confidence = float(max(calibrated_p, 1.0 - calibrated_p))
        return calibrated_p, confidence

    def save(self, output_path: str):
        """Saves calibrator state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        joblib.dump({
            "meta_learner": self.meta_learner,
            "calibrated_model": self.calibrated_model,
            "metrics": self.metrics,
            "calibration_method": self.calibration_method,
            "is_fitted": self.is_fitted,
        }, output_path)
        self.model_path = output_path
        print(f"[Calibrator] Saved calibrated risk model to: {output_path}")

    def load(self, model_path: str):
        """Loads calibrator state from disk."""
        data = joblib.load(model_path)
        self.meta_learner = data.get("meta_learner")
        self.calibrated_model = data.get("calibrated_model")
        self.metrics = data.get("metrics", {})
        self.calibration_method = data.get("calibration_method", "sigmoid")
        self.is_fitted = data.get("is_fitted", True)
        self.model_path = model_path
        print(f"[Calibrator] Loaded calibrated risk model from: {model_path}")
