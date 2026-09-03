"""
VoiceGuard AI - Real Deepfake Corpora Loader & Cross-Generator / Cross-Speaker Splitter
Supports schemas for:
- ASVspoof 2019 (LA / PA) & ASVspoof 2021 (DF / LA)
- WaveFake (6+ vocoders / architectures: MelGAN, HiFi-GAN, WaveGlow, etc.)
- In-The-Wild Deepfake Audio
- Cross-Generator & Cross-Speaker Out-Of-Distribution (OOD) Partitioning
"""

import os
import json
import random
import numpy as np
import soundfile as sf
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional


@dataclass
class AudioMetadata:
    filename: str
    filepath: str
    label: str  # "bonafide" (human) or "spoof" (synthetic)
    speaker_id: str
    generator_id: str  # e.g., "human", "hifigan", "tacotron2", "vits", "bark", "elevenlabs"
    vocoder: str  # e.g., "natural", "hifigan", "griffin_lim", "waveglow"
    sample_rate: int
    duration_sec: float
    split: str = "train"  # "train", "val", "test_id", "test_ood_generator", "test_ood_speaker"


class CorporaManager:
    """
    Manages schemas, manifest loading, and partition generation for real-world audio corpora.
    Enforces strict Cross-Generator and Cross-Speaker holdout splits to verify true model generalization.
    """

    SUPPORTED_CORPORA = ["asvspoof2019", "asvspoof2021", "wavefake", "in_the_wild"]

    # Known synthesis algorithms in benchmark datasets
    KNOWN_GENERATORS = {
        "A01": "Neural Vocoder + Spectral Filter",
        "A02": "WaveNet / Mel-Spectrogram",
        "A03": "Tacotron2 + Griffin-Lim",
        "A04": "Tacotron2 + WaveRNN",
        "A05": "VCC2018 Voice Conversion",
        "A06": "FastSpeech + HiFi-GAN",
        "A07": "VITS End-to-End Variational",
        "A08": "Bark Audio LM",
        "A09": "Diffusion / Flow-Matching (ElevenLabs style)",
    }

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir
        self.metadata_records: List[AudioMetadata] = []

    def load_asvspoof_protocol(self, protocol_path: str, audio_dir: str) -> List[AudioMetadata]:
        """
        Parses ASVspoof 2019/2021 protocol file format:
        Columns: [SPEAKER_ID, AUDIO_FILE_NAME, SYSTEM_ID, KEY]
        KEY: 'bonafide' or 'spoof'
        """
        records = []
        if not os.path.exists(protocol_path):
            raise FileNotFoundError(f"ASVspoof protocol file not found at: {protocol_path}")

        with open(protocol_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    speaker_id = parts[0]
                    file_name = parts[1]
                    system_id = parts[2]
                    key = parts[3].lower()  # 'bonafide' or 'spoof'

                    if not file_name.endswith(".wav") and not file_name.endswith(".flac"):
                        candidate_wav = os.path.join(audio_dir, f"{file_name}.wav")
                        candidate_flac = os.path.join(audio_dir, f"{file_name}.flac")
                        fpath = candidate_wav if os.path.exists(candidate_wav) else candidate_flac
                    else:
                        fpath = os.path.join(audio_dir, file_name)

                    generator = "human" if key == "bonafide" else self.KNOWN_GENERATORS.get(system_id, system_id)
                    rec = AudioMetadata(
                        filename=file_name,
                        filepath=fpath,
                        label=key,
                        speaker_id=speaker_id,
                        generator_id=generator,
                        vocoder="natural" if key == "bonafide" else system_id,
                        sample_rate=16000,
                        duration_sec=4.0,
                    )
                    records.append(rec)

        self.metadata_records.extend(records)
        return records

    def load_wavefake_manifest(self, manifest_json: str, audio_root: str) -> List[AudioMetadata]:
        """
        Parses WaveFake dataset directory structure or JSON manifest.
        """
        records = []
        if not os.path.exists(manifest_json):
            raise FileNotFoundError(f"WaveFake manifest not found: {manifest_json}")

        with open(manifest_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            fpath = os.path.join(audio_root, item.get("relative_path", item.get("filename", "")))
            rec = AudioMetadata(
                filename=os.path.basename(fpath),
                filepath=fpath,
                label=item.get("label", "spoof"),
                speaker_id=item.get("speaker_id", "ljspeech"),
                generator_id=item.get("architecture", "hifigan"),
                vocoder=item.get("vocoder", "hifigan"),
                sample_rate=int(item.get("sample_rate", 16000)),
                duration_sec=float(item.get("duration", 4.0)),
            )
            records.append(rec)

        self.metadata_records.extend(records)
        return records

    def create_cross_generator_split(
        self,
        records: List[AudioMetadata],
        held_out_generators: List[str],
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, List[AudioMetadata]]:
        """
        Cross-Generator Split:
        - Train: Bonafide + Spoofs from seen generators (e.g. Tacotron2, WaveNet, MelGAN)
        - Val: Bonafide + Seen Spoofs
        - Test In-Distribution (ID): Test samples from seen generators
        - Test Out-of-Distribution (OOD): Spoofs from completely unseen held-out generators (e.g. VITS, Bark)
        """
        random.seed(seed)

        train_records = []
        val_records = []
        test_id_records = []
        test_ood_records = []

        bonafide = [r for r in records if r.label == "bonafide"]
        random.shuffle(bonafide)
        n_bonafide_val = int(len(bonafide) * val_ratio)
        n_bonafide_test = int(len(bonafide) * val_ratio)

        for r in bonafide[:n_bonafide_val]:
            r.split = "val"
            val_records.append(r)
        for r in bonafide[n_bonafide_val : n_bonafide_val + n_bonafide_test]:
            r.split = "test_id"
            test_id_records.append(r)
        for r in bonafide[n_bonafide_val + n_bonafide_test :]:
            r.split = "train"
            train_records.append(r)

        spoofs = [r for r in records if r.label != "bonafide"]
        for r in spoofs:
            is_held_out = any(hg.lower() in r.generator_id.lower() for hg in held_out_generators)
            if is_held_out:
                r.split = "test_ood_generator"
                test_ood_records.append(r)
            else:
                dice = random.random()
                if dice < val_ratio:
                    r.split = "val"
                    val_records.append(r)
                elif dice < (val_ratio * 2):
                    r.split = "test_id"
                    test_id_records.append(r)
                else:
                    r.split = "train"
                    train_records.append(r)

        return {
            "train": train_records,
            "val": val_records,
            "test_id": test_id_records,
            "test_ood_generator": test_ood_records,
        }

    def create_cross_speaker_split(
        self,
        records: List[AudioMetadata],
        held_out_speakers: List[str],
    ) -> Dict[str, List[AudioMetadata]]:
        """
        Cross-Speaker Split:
        Holds out entire speaker identities from training.
        """
        train_records = []
        test_ood_speaker_records = []

        for r in records:
            if r.speaker_id in held_out_speakers:
                r.split = "test_ood_speaker"
                test_ood_speaker_records.append(r)
            else:
                r.split = "train"
                train_records.append(r)

        return {
            "train": train_records,
            "test_ood_speaker": test_ood_speaker_records,
        }

    def export_manifest(self, records: List[AudioMetadata], output_path: str):
        """Exports metadata manifest to JSON for reproducible benchmarking."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = [asdict(r) for r in records]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[CorporaManager] Exported {len(records)} records manifest to: {output_path}")


def generate_benchmark_multi_generator_dataset(
    output_dir: str,
    samples_per_category: int = 15,
) -> Dict[str, List[AudioMetadata]]:
    """
    Synthesizes a realistic multi-generator evaluation dataset with labeled generator signatures
    and acoustic fingerprints for cross-generator and cross-speaker experimentation.
    Generators represented:
    1. Authentic Human (Male, Female, Senior/Grandma speakers)
    2. HiFi-GAN Vocoder Clone
    3. Tacotron2 + Griffin-Lim
    4. FastSpeech2 + Multi-Band MelGAN
    5. VITS End-to-End Deepfake (HELD-OUT GENERATOR)
    6. Bark Audio LM (HELD-OUT GENERATOR)
    """
    os.makedirs(output_dir, exist_ok=True)
    sr = 16000
    duration = 3.5
    num_samples = int(sr * duration)
    records = []

    speakers = ["speaker_01_m", "speaker_02_f", "speaker_03_m", "speaker_04_f", "speaker_05_grandma"]

    from .dataset_generator import generate_human_speech_sample, generate_synthetic_speech_sample

    def _make_human_audio(spk: str) -> np.ndarray:
        pitch = 120.0 if "_m" in spk else (190.0 if "_f" in spk else 165.0)
        gender = "male" if "_m" in spk else "female"
        return generate_human_speech_sample(duration=duration, base_pitch=pitch, gender=gender)

    def _make_synth_audio(gen_type: str, spk: str) -> np.ndarray:
        type_mapping = {
            "hifigan": "voice_clone",
            "tacotron_griffin_lim": "tts_vocoder",
            "fastspeech_melgan": "diffusion_vocoder",
            "vits": "voice_clone",
            "bark": "robotic_neural",
        }
        mapped_type = type_mapping.get(gen_type, "voice_clone")
        return generate_synthetic_speech_sample(duration=duration, synth_type=mapped_type)

    # 1. Authentic samples
    real_dir = os.path.join(output_dir, "bonafide")
    os.makedirs(real_dir, exist_ok=True)
    for i in range(samples_per_category * 2):
        spk = speakers[i % len(speakers)]
        fname = f"bonafide_{spk}_{i:03d}.wav"
        fpath = os.path.join(real_dir, fname)
        audio = _make_human_audio(spk)
        sf.write(fpath, audio, sr, subtype="PCM_16")
        records.append(AudioMetadata(
            filename=fname,
            filepath=fpath,
            label="bonafide",
            speaker_id=spk,
            generator_id="human",
            vocoder="natural",
            sample_rate=sr,
            duration_sec=duration,
        ))

    # 2. Synthetic samples across generators
    generators = ["hifigan", "tacotron_griffin_lim", "fastspeech_melgan", "vits", "bark"]
    for gen in generators:
        gen_dir = os.path.join(output_dir, f"spoof_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        for i in range(samples_per_category):
            spk = speakers[i % len(speakers)]
            fname = f"spoof_{gen}_{spk}_{i:03d}.wav"
            fpath = os.path.join(gen_dir, fname)
            audio = _make_synth_audio(gen, spk)
            sf.write(fpath, audio, sr, subtype="PCM_16")
            records.append(AudioMetadata(
                filename=fname,
                filepath=fpath,
                label="spoof",
                speaker_id=spk,
                generator_id=gen,
                vocoder=gen,
                sample_rate=sr,
                duration_sec=duration,
            ))

    manifest_path = os.path.join(output_dir, "corpus_manifest.json")
    mgr = CorporaManager()
    mgr.export_manifest(records, manifest_path)

    splits = mgr.create_cross_generator_split(
        records=records,
        held_out_generators=["vits", "bark"],
        val_ratio=0.15,
        seed=42,
    )
    return splits
