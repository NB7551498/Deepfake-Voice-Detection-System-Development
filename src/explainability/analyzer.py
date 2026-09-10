"""
Explainability and Acoustic Evidence Analyzer.

Extracts forensic acoustic indicators per segment:
  - Spectral Flatness anomaly (robotic / uniform frequency content)
  - MFCC temporal variance (abnormally low variance = synthetic voice)
  - Pitch jitter / stability (unnatural prosody stability)
  - Zero-crossing rate anomaly
"""

import numpy as np
import librosa
from typing import List, Dict, Any, Tuple


def analyze_acoustic_evidence(chunk: np.ndarray, sr: int = 16000, fake_prob: float = 0.5) -> Dict[str, Any]:
    """Analyze a single audio chunk and identify acoustic evidence."""
    if len(chunk) < 2048:
        chunk = np.pad(chunk, (0, 2048 - len(chunk)), mode='constant')

    hop = 512

    # 1. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=chunk, hop_length=hop)
    flatness_mean = float(np.mean(flatness))

    # 2. MFCC Temporal Variance (variance across time frames)
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20, hop_length=hop)
    mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

    # 3. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(chunk, hop_length=hop)
    zcr_mean = float(np.mean(zcr))

    # 4. Pitch Micro-variation
    try:
        f0, voiced, _ = librosa.pyin(chunk, fmin=60, fmax=450, sr=sr)
        if voiced is not None and np.sum(voiced) > 2:
            voiced_f0 = f0[voiced > 0]
            pitch_std = float(np.std(voiced_f0))
        else:
            pitch_std = 0.0
    except Exception:
        pitch_std = 0.0

    # Music & Singing Awareness: Check if instrumental/percussive accompaniment is present
    try:
        harmonic, percussive = librosa.effects.hpss(chunk, margin=(1.0, 3.0))
        is_music = np.mean(percussive ** 2) > 0.04 * (np.mean(harmonic ** 2) + 1e-6)
    except Exception:
        is_music = False

    # Evidence rules
    evidence = []
    if is_music:
        evidence.append("Musical / singing accompaniment detected — vocal harmonics isolated")
        harm_flatness = float(np.mean(librosa.feature.spectral_flatness(y=harmonic, hop_length=hop)))
        if harm_flatness > 0.18 and fake_prob > 0.65:
            evidence.append("High vocal flatness (synthetic vocoder harmonic collapse)")
        if mfcc_var < 3.0 and fake_prob > 0.65:
            evidence.append("Unnaturally low vocal variance (voice cloning artifact)")
    else:
        if flatness_mean > 0.14 and fake_prob > 0.60:
            evidence.append("High spectral flatness (synthetic acoustic uniformity)")
        if mfcc_var < 4.0 and fake_prob > 0.60:
            evidence.append("Unnaturally low MFCC variance (smooth cloned phonation)")
        if 0 < pitch_std < 4.0 and fake_prob > 0.60:
            evidence.append("Robotic pitch consistency (attenuated prosodic jitter)")
        if zcr_mean > 0.18 and fake_prob > 0.60:
            evidence.append("Elevated zero-crossing rate (synthetic high-frequency artifacts)")

    if not evidence and fake_prob > 0.65:
        evidence.append("Acoustic distribution shifted towards generative synthetic profile")
    elif not evidence and fake_prob < 0.35:
        evidence.append("Harmonic distribution consistent with organic human phonation")
    elif not evidence:
        evidence.append("Acoustic features in neutral variance zone")

    severity = "HIGH" if fake_prob > 0.70 else "MEDIUM" if fake_prob >= 0.35 else "LOW"

    return {
        "fake_probability": round(fake_prob, 4),
        "severity": severity,
        "spectral_flatness": round(flatness_mean, 4),
        "mfcc_variance": round(mfcc_var, 3),
        "pitch_variation_hz": round(pitch_std, 2),
        "zcr": round(zcr_mean, 4),
        "evidence": evidence,
    }


def build_suspicious_timeline(
    segment_data: List[Dict[str, Any]],
    threshold: float = 0.50,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge adjacent suspicious segments into timeline regions."""
    suspicious_segs = [s for s in segment_data if s["fake_probability"] >= threshold]

    merged_regions = []
    for seg in suspicious_segs:
        if merged_regions and seg["start"] <= merged_regions[-1]["end"] + 0.5:
            merged_regions[-1]["end"] = max(merged_regions[-1]["end"], seg["end"])
            merged_regions[-1]["max_fake_prob"] = max(
                merged_regions[-1]["max_fake_prob"], seg["fake_probability"]
            )
            # Combine unique evidence
            for ev in seg.get("evidence", []):
                if ev not in merged_regions[-1]["evidence"]:
                    merged_regions[-1]["evidence"].append(ev)
            if seg.get("severity") == "HIGH":
                merged_regions[-1]["severity"] = "HIGH"
        else:
            merged_regions.append({
                "start": seg["start"],
                "end": seg["end"],
                "max_fake_prob": seg["fake_probability"],
                "severity": seg.get("severity", "MEDIUM"),
                "evidence": list(seg.get("evidence", [])),
            })

    summary = {
        "total_segments": len(segment_data),
        "suspicious_segment_count": len(suspicious_segs),
        "suspicious_region_count": len(merged_regions),
        "clean_segment_count": len(segment_data) - len(suspicious_segs),
    }

    return merged_regions, summary
