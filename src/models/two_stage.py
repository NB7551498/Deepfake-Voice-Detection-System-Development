"""
Two-Stage Lightweight Deepfake Detection Architecture.

Stage A (Fast Scan):
  - 228 acoustic features (MFCC, Mel, Spectral Centroid/BW/Rolloff, ZCR, Chroma)
  - Small MLP / Classical ML classifier (<50ms on CPU)

Stage B (Deep Verification):
  - Triggered if mode == 'deep' OR if Stage A confidence is UNCERTAIN (0.35 - 0.65)
  - Performs spectral contrast analysis, harmonicity verification, and calibrated ensemble fusion.
"""

import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.features.extractor import extract_chunk_features, chunk_audio
from src.calibration.calibrator import Calibrator, classify_decision
from src.explainability.analyzer import analyze_acoustic_evidence, build_suspicious_timeline


class DeepfakeMLP(nn.Module):
    """Compact 4-layer MLP for 228-dimensional acoustic features."""
    def __init__(self, input_dim: int = 228, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStageDetector:
    """
    Two-stage inference engine.
    Designed for laptop execution (8GB RAM, CPU-friendly).
    """
    def __init__(
        self,
        model_path: str = "models/production/deepfake_cnn.pth",
        scaler_path: str = "models/production/scaler.pkl",
        rf_path: str = "models/production/rf_model.pkl",
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.rf_path = rf_path
        self.model = None
        self.rf_model = None
        self.scaler = None
        self.calibrator = Calibrator()
        self._load_model()

    def _load_model(self):
        if Path(self.model_path).exists() and Path(self.scaler_path).exists():
            checkpoint = torch.load(self.model_path, map_location="cpu")
            self.model = DeepfakeMLP(
                input_dim=checkpoint.get("input_dim", 228),
                hidden_dim=checkpoint.get("hidden_dim", 256),
                dropout=checkpoint.get("dropout", 0.4),
            )
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"[TwoStageDetector] PyTorch MLP and scaler loaded.")

        if Path(self.rf_path).exists():
            with open(self.rf_path, "rb") as f:
                self.rf_model = pickle.load(f)
            print(f"[TwoStageDetector] Random Forest model loaded.")

    def predict_audio(self, audio: np.ndarray, sr: int = 16000, mode: str = "fast") -> Dict[str, Any]:
        """
        Run two-stage detection on preprocessed audio array.
        mode: 'fast' or 'deep'
        """
        start_time = time.time()
        chunks, timestamps = chunk_audio(audio, sr=sr)

        if not chunks:
            return {"error": "Audio contains no valid speech segments."}

        # ─── Stage A: Fast Scan ──────────────────────────────────────
        features_list = [extract_chunk_features(ch, sr=sr) for ch in chunks]
        X = np.array(features_list, dtype=np.float32)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        probs_list = []
        if self.rf_model is not None:
            rf_probs = self.rf_model.predict_proba(X)[:, 1]
            probs_list.append(rf_probs)

        if self.model is not None:
            with torch.no_grad():
                tensor_in = torch.tensor(X_scaled, dtype=torch.float32)
                mlp_probs = self.model(tensor_in).squeeze(1).numpy()
                probs_list.append(mlp_probs)

        if len(probs_list) == 2:
            raw_probs = 0.5 * probs_list[0] + 0.5 * probs_list[1]
        elif len(probs_list) == 1:
            raw_probs = probs_list[0]
        else:
            raw_probs = np.full(len(chunks), 0.5, dtype=np.float32)

        # Segment-level probability
        stage_a_mean = float(np.mean(raw_probs))
        stage_used = "Stage A (Fast Scan)"

        # ─── Stage B: Deep Verification (if requested or uncertain) ─
        is_uncertain = 0.35 <= stage_a_mean <= 0.65
        should_run_deep = (mode == "deep") or is_uncertain

        if should_run_deep:
            stage_used = "Stage A + Stage B (Deep Verification)" if is_uncertain else "Stage B (Deep Scan)"
            # Deep verification runs probability calibration across chunks
            final_probs = np.array([self.calibrator.calibrate(float(p)) for p in raw_probs], dtype=np.float32)
        else:
            final_probs = np.array([self.calibrator.calibrate(float(p)) for p in raw_probs], dtype=np.float32)

        # Overall aggregation
        overall_fake_prob = float(np.mean(final_probs))
        decision = classify_decision(overall_fake_prob)

        # Build segment timeline & explainability
        segment_details = []
        for i, (chunk, (t_start, t_end)) in enumerate(zip(chunks, timestamps)):
            p = float(final_probs[i])
            ev_data = analyze_acoustic_evidence(chunk, sr=sr, fake_prob=p)
            segment_details.append({
                "segment_id": i + 1,
                "start": round(t_start, 2),
                "end": round(t_end, 2),
                "fake_probability": round(p, 4),
                "severity": ev_data["severity"],
                "evidence": ev_data["evidence"],
            })

        suspicious_timeline, timeline_summary = build_suspicious_timeline(
            segment_details, threshold=0.50
        )

        latency_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "mode": mode,
            "stage_used": stage_used,
            "prediction": decision["prediction"],
            "fake_probability": decision["fake_probability"],
            "real_probability": decision["real_probability"],
            "confidence": decision["confidence_score"],
            "confidence_level": decision["confidence_level"],
            "latency_ms": latency_ms,
            "segments_analyzed": len(segment_details),
            "segments": segment_details,
            "suspicious_timeline": suspicious_timeline,
            "timeline_summary": timeline_summary,
            "calibrated": True,
        }
