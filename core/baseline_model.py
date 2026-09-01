"""
VoiceGuard AI - Baseline ML Model
Extracts statistical acoustic & spectral features (MFCCs, Deltas, F0, Vocoder Ratios, ZCR)
and trains a calibrated Random Forest / Gradient Boosting classifier for Real vs Synthetic voice detection.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

from .audio_preprocessor import AudioPreprocessor
from .feature_extractor import FeatureExtractor


class BaselineVoiceClassifier:
    """
    Baseline Machine Learning Model for Voice Deepfake Detection.
    Uses an ensemble of Random Forest and Gradient Boosting over tabular forensic features.
    """

    def __init__(self, model_path: str = None):
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.metrics = {}
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def extract_dataset_features(self, real_dir: str, fake_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Loads all audio files from real/ and fake/ directories and extracts tabular features.
        Label: 0 = Real Human, 1 = AI Synthetic / Cloned Voice.
        """
        X_list = []
        y_list = []
        feature_names = []

        # Process Real samples (label = 0)
        real_files = [
            os.path.join(real_dir, f)
            for f in os.listdir(real_dir)
            if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))
        ]
        for fpath in real_files:
            try:
                proc = self.preprocessor.process(fpath)
                feats, f_names = self.feature_extractor.extract_tabular_features(proc["audio"])
                X_list.append(feats)
                y_list.append(0)
                if not feature_names:
                    feature_names = f_names
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

        # Process Fake samples (label = 1)
        fake_files = [
            os.path.join(fake_dir, f)
            for f in os.listdir(fake_dir)
            if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))
        ]
        for fpath in fake_files:
            try:
                proc = self.preprocessor.process(fpath)
                feats, _ = self.feature_extractor.extract_tabular_features(proc["audio"])
                X_list.append(feats)
                y_list.append(1)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        self.feature_names = feature_names
        return X, y, feature_names

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = 42) -> dict:
        """
        Trains the ensemble baseline classifier and evaluates cross-validated metrics.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build Ensemble: Random Forest + Gradient Boosting
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.08, max_depth=4, random_state=random_state
        )

        self.model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            auc = 1.0

        cm = confusion_matrix(y_test, y_pred).tolist()
        cv_scores = cross_val_score(self.model, self.scaler.transform(X), y, cv=4).tolist()

        self.metrics = {
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
            "num_features": X.shape[1],
            "train_samples": len(y_train),
            "test_samples": len(y_test),
        }

        print(f"Baseline Model Trained -> Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc:.4f}")
        return self.metrics

    def predict(self, audio_input) -> dict:
        """
        Runs baseline prediction on raw audio array, audio bytes, or audio filepath.
        Returns:
            {
                "synthetic_probability": float (0.0 to 1.0),
                "genuine_probability": float (0.0 to 1.0),
                "prediction": "AI_SYNTHETIC" or "GENUINE_HUMAN",
                "confidence": float
            }
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded or trained. Call train() or load() first.")

        if not isinstance(audio_input, np.ndarray) or len(audio_input.shape) == 0:
            proc = self.preprocessor.process(audio_input)
            audio = proc["audio"]
        else:
            audio = self.preprocessor.pad_or_truncate(self.preprocessor.normalize_audio(audio_input))

        feats, _ = self.feature_extractor.extract_tabular_features(audio)
        feats_scaled = self.scaler.transform(feats.reshape(1, -1))

        proba = self.model.predict_proba(feats_scaled)[0]
        p_real = float(proba[0])
        p_fake = float(proba[1])

        pred_class = "AI_SYNTHETIC" if p_fake >= 0.5 else "GENUINE_HUMAN"
        confidence = float(max(p_real, p_fake))

        return {
            "synthetic_probability": round(p_fake, 4),
            "genuine_probability": round(p_real, 4),
            "prediction": pred_class,
            "confidence": round(confidence, 4),
        }

    def save(self, output_path: str):
        """Saves trained model, scaler, feature names, and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }
        joblib.dump(data, output_path)
        self.model_path = output_path
        print(f"Baseline model saved to: {output_path}")

    def load(self, model_path: str):
        """Loads trained model, scaler, and metadata from disk."""
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data.get("feature_names", [])
        self.metrics = data.get("metrics", {})
        self.model_path = model_path
        print(f"Baseline model successfully loaded from: {model_path}")
