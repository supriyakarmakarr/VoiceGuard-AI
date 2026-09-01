"""
VoiceGuard AI - Deep Learning Model (PyTorch)
2D Spectrogram Convolutional Neural Network with Residual Connections and Squeeze-and-Excitation (SE) Attention
for Mel-Spectrogram based Synthetic / Voice-Clone Detection.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .audio_preprocessor import AudioPreprocessor
from .feature_extractor import FeatureExtractor


class SqueezeExcitation(nn.Module):
    """Channel-wise Attention block to emphasize critical acoustic frequency bands."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = x.view(b, c, -1).mean(dim=2)  # Global average pool
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """Residual Block with SE Attention."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.leaky_relu(out + res, negative_slope=0.1)
        return out


class VoiceGuardSpectrogramCNN(nn.Module):
    """
    Deep 2D Convolutional Neural Network for Audio Deepfake Forensics.
    Input: (Batch, 1, 128, 128) Mel-Spectrogram tensor.
    Output: Synthetic Probability logit / probability.
    """

    def __init__(self):
        super().__init__()
        # Initial Conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # -> (32, 64, 64)
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            ResBlock(32, 64),
            nn.MaxPool2d(2, 2),  # -> (64, 32, 32)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ResBlock(64, 128),
            nn.MaxPool2d(2, 2),  # -> (128, 16, 16)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ResBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (256, 1, 1)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Dropout(0.35),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),  # binary output logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        out = self.classifier(x)
        return out


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for audio mel-spectrograms."""

    def __init__(self, specs: list[np.ndarray], labels: list[int]):
        self.specs = specs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.specs[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return x, y


class DeepLearningVoiceClassifier:
    """
    Manager class for Deep Learning CNN Training, Evaluation, and Real-Time Inference.
    """

    def __init__(self, model_path: str = None, device: str = None):
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = VoiceGuardSpectrogramCNN().to(self.device)
        self.model_path = model_path
        self.metrics = {}

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def prepare_dataset(self, real_dir: str, fake_dir: str):
        """Loads and extracts Mel-spectrograms for training."""
        specs = []
        labels = []

        real_files = [
            os.path.join(real_dir, f)
            for f in os.listdir(real_dir)
            if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))
        ]
        for fpath in real_files:
            try:
                proc = self.preprocessor.process(fpath)
                spec = self.feature_extractor.extract_mel_spectrogram(proc["audio"])
                specs.append(spec)
                labels.append(0)  # 0 = Real
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

        fake_files = [
            os.path.join(fake_dir, f)
            for f in os.listdir(fake_dir)
            if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))
        ]
        for fpath in fake_files:
            try:
                proc = self.preprocessor.process(fpath)
                spec = self.feature_extractor.extract_mel_spectrogram(proc["audio"])
                specs.append(spec)
                labels.append(1)  # 1 = Fake / Synth
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

        return specs, labels

    def train(
        self,
        specs: list[np.ndarray],
        labels: list[int],
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 0.001,
        val_split: float = 0.2,
    ) -> dict:
        """
        Trains the Deep Spectrogram CNN on extracted spectrograms.
        """
        indices = np.arange(len(labels))
        np.random.seed(42)
        np.random.shuffle(indices)

        val_size = int(len(labels) * val_split)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        train_specs = [specs[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_specs = [specs[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_ds = SpectrogramDataset(train_specs, train_labels)
        val_ds = SpectrogramDataset(val_specs, val_labels)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_acc = 0.0

        self.model.train()
        for epoch in range(epochs):
            total_train_loss = 0.0
            correct = 0
            total = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

            scheduler.step()
            train_acc = correct / max(1, total)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    val_logits = self.model(x_val)
                    v_loss = criterion(val_logits, y_val)
                    val_loss += v_loss.item() * len(y_val)
                    v_preds = (torch.sigmoid(val_logits) >= 0.5).float()
                    val_correct += (v_preds == y_val).sum().item()
                    val_total += len(y_val)

            val_acc = val_correct / max(1, val_total)
            self.model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_acc = val_acc

        self.metrics = {
            "validation_accuracy": round(float(best_acc), 4),
            "final_train_accuracy": round(float(train_acc), 4),
            "epochs_trained": epochs,
            "architecture": "ResNet-SE Spectrogram 2D-CNN",
            "device": str(self.device),
        }
        print(f"Deep Learning CNN Trained -> Val Accuracy: {best_acc*100:.2f}%")
        return self.metrics

    def predict(self, audio_input) -> dict:
        """
        Runs deep learning inference on audio input (file, bytes, or numpy array).
        Returns synthetic probability and model confidence.
        """
        self.model.eval()
        if not isinstance(audio_input, np.ndarray) or len(audio_input.shape) == 0:
            proc = self.preprocessor.process(audio_input)
            audio = proc["audio"]
        else:
            audio = self.preprocessor.pad_or_truncate(self.preprocessor.normalize_audio(audio_input))

        spec = self.feature_extractor.extract_mel_spectrogram(audio)
        tensor_in = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor_in)
            prob_fake = float(torch.sigmoid(logit).squeeze().cpu().item())

        prob_real = 1.0 - prob_fake
        pred_class = "AI_SYNTHETIC" if prob_fake >= 0.5 else "GENUINE_HUMAN"
        confidence = float(max(prob_real, prob_fake))

        return {
            "synthetic_probability": round(prob_fake, 4),
            "genuine_probability": round(prob_real, 4),
            "prediction": pred_class,
            "confidence": round(confidence, 4),
        }

    def save(self, output_path: str):
        """Saves PyTorch model state dictionary and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "metrics": self.metrics,
        }, output_path)
        self.model_path = output_path
        print(f"Deep CNN model saved to: {output_path}")

    def load(self, model_path: str):
        """Loads PyTorch model weights from disk."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.metrics = checkpoint.get("metrics", {})
        self.model_path = model_path
        self.model.eval()
        print(f"Deep CNN model loaded from: {model_path}")
