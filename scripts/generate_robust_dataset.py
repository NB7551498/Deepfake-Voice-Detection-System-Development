"""
Robust Dataset Generator: Real Human Speech, Singing & Music vs Deepfake AI Audio.

Generates realistic audio samples covering:
  REAL:
    - Conversational human speech (natural formants, pitch glide, breath)
    - Human singing voices (sustained vowels, 5-7Hz vibrato, expressive dynamic range)
    - Studio songs with reverb, delay, and vocal compression
    - Modern songs with subtle pitch correction (Auto-tune) & background chords/beats
  
  FAKE (DEEPFAKE):
    - Neural vocoder artifacts (high-frequency phase mismatch, metallic buzzing)
    - AI Voice Cloning / Singing Voice Conversion (RVC/DiffSVC style harmonic smearing)
    - TTS acoustic synthesis (over-smoothed spectral envelopes, unnatural pitch rigidity)
    - Generative speech synthesis artifacts (spectral buzz, phase discontinuities)
"""

import os
import random
import numpy as np
import scipy.io.wavfile as wav_io
from pathlib import Path

SAMPLE_RATE = 16000
DURATION_S = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)
t = np.linspace(0, DURATION_S, N_SAMPLES, endpoint=False)

TRAIN_REAL = Path("data/processed/train/real")
TRAIN_FAKE = Path("data/processed/train/fake")
VAL_REAL   = Path("data/processed/validation/real")
VAL_FAKE   = Path("data/processed/validation/fake")

for p in [TRAIN_REAL, TRAIN_FAKE, VAL_REAL, VAL_FAKE]:
    p.mkdir(parents=True, exist_ok=True)


def simulate_vocal_formants(f0: float, time_arr: np.ndarray, is_singing: bool = False) -> np.ndarray:
    """Generate human vocal fold excitation with natural formants (F1, F2, F3)."""
    # Natural micro-jitter (human vocal cords never vibrate with 100% mathematical perfection)
    jitter = 1.0 + 0.008 * np.sin(2 * np.pi * 12.3 * time_arr) + np.random.normal(0, 0.003, len(time_arr))
    
    if is_singing:
        # Singing vibrato: 5.5 Hz pitch modulation with natural onset
        vibrato_depth = 0.03
        vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * 5.5 * time_arr)
        pitch = f0 * jitter * vibrato
    else:
        # Speech intonation glide
        intonation = 1.0 - 0.08 * (time_arr / DURATION_S)
        pitch = f0 * jitter * intonation

    # Glottal pulse train harmonics
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    glottal = (
        0.50 * np.sin(phase) +
        0.30 * np.sin(2 * phase) +
        0.18 * np.sin(3 * phase) +
        0.10 * np.sin(4 * phase) +
        0.05 * np.sin(5 * phase)
    )

    # Human Formants: F1 ~ 600Hz (vowel /a/), F2 ~ 1200Hz, F3 ~ 2500Hz
    f1 = 0.35 * np.sin(2 * np.pi * 600 * time_arr) * np.exp(-((time_arr % 0.05) * 40))
    f2 = 0.20 * np.sin(2 * np.pi * 1200 * time_arr) * np.exp(-((time_arr % 0.05) * 60))
    f3 = 0.10 * np.sin(2 * np.pi * 2500 * time_arr) * np.exp(-((time_arr % 0.05) * 80))

    voice = glottal + f1 + f2 + f3
    return voice


def add_studio_effects(audio: np.ndarray, is_song: bool = True) -> np.ndarray:
    """Simulate studio compression, acoustic reverb, and musical backing track."""
    # 1. Convolution reverb (exponential decaying impulse response)
    decay_len = int(SAMPLE_RATE * (0.35 if is_song else 0.15))
    ir = np.exp(-np.linspace(0, 5, decay_len))
    ir = ir / np.sum(ir)
    reverb = np.convolve(audio, ir, mode='full')[:len(audio)]
    mixed = 0.75 * audio + 0.25 * reverb

    # 2. Musical backing (piano chords / acoustic guitar / percussion)
    if is_song:
        chord_freqs = [220.0, 277.18, 329.63]  # A-major chord
        music = np.zeros_like(audio)
        for cf in chord_freqs:
            music += 0.08 * np.sin(2 * np.pi * cf * t)
        # Gentle rhythmic beat (kick + hi-hat)
        beat = np.sin(2 * np.pi * 2.0 * t)  # 120 bpm
        kick = 0.06 * np.sin(2 * np.pi * 60 * t) * (beat > 0.8)
        hihat = np.random.normal(0, 0.015, len(t)) * (np.sin(2 * np.pi * 4.0 * t) > 0.5)
        mixed = 0.70 * mixed + 0.20 * music + 0.06 * kick + 0.04 * hihat

    # 3. Soft saturation / studio vocal compressor
    mixed = np.tanh(mixed * 1.3)
    return mixed


def generate_real_sample(idx: int) -> np.ndarray:
    """Generate diverse REAL human audio (speech, singing, studio song)."""
    category = idx % 3
    base_f0 = random.uniform(100.0, 240.0)

    if category == 0:
        # Real conversational speech
        audio = simulate_vocal_formants(base_f0, t, is_singing=False)
        audio += np.random.normal(0, 0.01, len(t))  # Room mic ambient
        audio = add_studio_effects(audio, is_song=False)
    elif category == 1:
        # Real human singing voice (acapella with vibrato & reverb)
        audio = simulate_vocal_formants(base_f0, t, is_singing=True)
        audio = add_studio_effects(audio, is_song=True)
    else:
        # Real studio song with music, vocal vibrato, and subtle auto-tune
        audio = simulate_vocal_formants(base_f0, t, is_singing=True)
        # Subtle pitch quantization (modern studio auto-tune)
        quantized_pitch = np.round(base_f0 / 10.0) * 10.0
        audio = 0.7 * audio + 0.3 * np.sin(2 * np.pi * quantized_pitch * t)
        audio = add_studio_effects(audio, is_song=True)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    return audio.astype(np.float32)


def generate_fake_sample(idx: int) -> np.ndarray:
    """Generate realistic AI DEEPFAKE audio (TTS, cloned song, vocoder artifacts)."""
    category = idx % 3
    base_f0 = random.uniform(110.0, 220.0)

    if category == 0:
        # AI TTS: Mathematical pitch rigidity + high spectral flatness + absence of natural jitter
        # Neural vocoder buzz in high frequencies
        phase = 2 * np.pi * base_f0 * t
        tts_voice = (
            0.50 * np.sin(phase) +
            0.25 * np.sin(2 * phase) +
            0.15 * np.sin(3 * phase) +
            0.10 * np.sin(4 * phase)
        )
        # High-frequency vocoder phase mismatch artifact (buzzing above 3500 Hz)
        vocoder_buzz = 0.08 * np.sin(2 * np.pi * 3800 * t) * np.sin(2 * np.pi * 180 * t)
        audio = tts_voice + vocoder_buzz

    elif category == 1:
        # AI Cloned Song (e.g. RVC / DiffSVC / So-Vits model)
        # Hallmarks: harmonic smearing, robotic phase continuity, formant discontinuity
        vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.5 * t)  # synthetic mechanical vibrato
        pitch = base_f0 * vibrato
        phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
        
        # Sawtooth harmonic spread (vocoder artifact)
        audio = 0.45 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.20 * np.sin(3 * phase)
        # Harmonic smearing: inter-modulation sidebands common in neural voice conversion
        sideband = 0.12 * np.sin(phase * 1.04) + 0.08 * np.sin(phase * 0.96)
        audio += sideband

        # Even with background music, the cloned voice has phase distortion
        audio = add_studio_effects(audio, is_song=True)
        # Add vocoder phase discontinuity
        audio += 0.05 * np.sign(np.sin(2 * np.pi * 4200 * t))

    else:
        # Generative TTS with over-smoothed spectral envelope
        sawtooth = 2 * (t * base_f0 - np.floor(0.5 + t * base_f0))
        audio = 0.35 * sawtooth + 0.30 * np.sin(2 * np.pi * base_f0 * t)
        # High spectral flatness artifact
        audio += 0.05 * np.random.uniform(-1, 1, len(t))

    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    return audio.astype(np.float32)


def build_dataset():
    print("=" * 65)
    print("🎵 Generating Robust Dataset: Human Speech, Songs & AI Deepfakes")
    print("=" * 65)

    splits = [
        ("train", TRAIN_REAL, TRAIN_FAKE, 100),
        ("validation", VAL_REAL, VAL_FAKE, 30),
    ]

    for split_name, real_dir, fake_dir, count in splits:
        print(f"\n[Split: {split_name}] Generating {count} Real and {count} Fake files...")
        
        for i in range(count):
            # Real
            real_audio = generate_real_sample(i)
            wav_io.write(str(real_dir / f"real_{i:03d}.wav"), SAMPLE_RATE, (real_audio * 32767).astype(np.int16))
            
            # Fake
            fake_audio = generate_fake_sample(i)
            wav_io.write(str(fake_dir / f"fake_{i:03d}.wav"), SAMPLE_RATE, (fake_audio * 32767).astype(np.int16))

        print(f"  ✓ {split_name}: {count} Real, {count} Fake files saved.")

    print("\n✅ Dataset generation complete!")


if __name__ == "__main__":
    build_dataset()
