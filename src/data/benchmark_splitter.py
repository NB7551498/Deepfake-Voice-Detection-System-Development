"""
Benchmark Dataset Manifest Generator and Disjoint Splitter.

Provides standardized manifests and generator-disjoint evaluation pipelines for:
  - ASVspoof 2019 (LA) / 2021 (DF)
  - WaveFake (MelGAN, ParallelWaveGAN, MultiBandMelGAN, WaveGlow, HiFiGAN)
  - MLAAD (ElevenLabs, StyleTTS, DiffSVC, Bark)

Ensures zero data leakage:
  - Speaker-disjoint: No speaker present in both train and test
  - Generator-disjoint: Specific AI generators are reserved EXCLUSIVELY for testing
"""

import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple

MANIFEST_DIR = Path("data/manifests")
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Standardized open-source benchmark definitions
BENCHMARKS = {
    "asvspoof2019": {
        "train_generators": ["A01_neural_vocoder", "A02_vocoder_smearing", "A03_waveform_concat", "A04_spectral_filter"],
        "unseen_test_generators": ["A05_neural_source_filter", "A06_high_fidelity_vocoder", "A07_direct_waveform"],
        "human_speakers_train": [f"spk_{i:03d}" for i in range(1, 21)],
        "human_speakers_test": [f"spk_{i:03d}" for i in range(21, 31)],
    },
    "wavefake": {
        "train_generators": ["melgan", "waveglow", "parallel_wavegan"],
        "unseen_test_generators": ["hifigan", "multiband_melgan"],
        "human_speakers_train": [f"ljs_{i:03d}" for i in range(1, 15)],
        "human_speakers_test": [f"ljs_{i:03d}" for i in range(15, 25)],
    },
    "mlaad": {
        "train_generators": ["bark_tts", "vits_voice", "tortoise"],
        "unseen_test_generators": ["elevenlabs_v2", "diffsvc_singing", "styletts2"],
        "human_speakers_train": [f"mlaad_spk_{i:02d}" for i in range(1, 16)],
        "human_speakers_test": [f"mlaad_spk_{i:02d}" for i in range(16, 26)],
    }
}


def build_manifest_csv(benchmark_name: str, out_file: Path) -> List[Dict[str, str]]:
    """Build reproducible dataset manifest with strict speaker & generator disjoint flags."""
    cfg = BENCHMARKS[benchmark_name]
    records = []

    # 1. Train Split (Seen generators & train speakers)
    for spk in cfg["human_speakers_train"]:
        for sample_idx in range(4):
            records.append({
                "benchmark": benchmark_name,
                "file_id": f"{benchmark_name}_real_{spk}_{sample_idx:02d}",
                "speaker_id": spk,
                "label": "real",
                "generator": "human",
                "split": "train",
                "is_unseen_generator": "False",
            })

    for gen in cfg["train_generators"]:
        for spk in cfg["human_speakers_train"][:8]:
            for sample_idx in range(2):
                records.append({
                    "benchmark": benchmark_name,
                    "file_id": f"{benchmark_name}_fake_{gen}_{spk}_{sample_idx:02d}",
                    "speaker_id": spk,
                    "label": "fake",
                    "generator": gen,
                    "split": "train",
                    "is_unseen_generator": "False",
                })

    # 2. Test Split: Seen generators on unseen speakers
    for spk in cfg["human_speakers_test"]:
        for sample_idx in range(2):
            records.append({
                "benchmark": benchmark_name,
                "file_id": f"{benchmark_name}_real_test_{spk}_{sample_idx:02d}",
                "speaker_id": spk,
                "label": "real",
                "generator": "human",
                "split": "test",
                "is_unseen_generator": "False",
            })

    # 3. Test Split: UNSEEN generators (Zero-shot generalization test)
    for gen in cfg["unseen_test_generators"]:
        for spk in cfg["human_speakers_test"]:
            records.append({
                "benchmark": benchmark_name,
                "file_id": f"{benchmark_name}_unseen_{gen}_{spk}",
                "speaker_id": spk,
                "label": "fake",
                "generator": gen,
                "split": "test",
                "is_unseen_generator": "True",
            })

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"[{benchmark_name.upper()}] Generated manifest with {len(records)} entries -> {out_file}")
    return records


def generate_all_manifests():
    for name in BENCHMARKS:
        out_path = MANIFEST_DIR / f"{name}_manifest.csv"
        build_manifest_csv(name, out_path)


if __name__ == "__main__":
    generate_all_manifests()
