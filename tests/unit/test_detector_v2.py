import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.two_stage import TwoStageDetector


def test_detector_v2():
    det = TwoStageDetector()
    dummy_audio = np.random.randn(16000 * 4).astype(np.float32)

    # 1. Test Fast Scan
    res_fast = det.predict_audio(dummy_audio, mode="fast")
    assert "prediction" in res_fast
    assert res_fast["prediction"] in ["REAL", "UNCERTAIN", "DEEPFAKE"]
    assert res_fast["model_version"] == "Ensemble-v2.0"
    assert "latency_ms" in res_fast
    print("[PASS] Fast Scan passed:", res_fast["prediction"], f"({res_fast['latency_ms']} ms)")

    # 2. Test Deep Verification with Fusion 2.0
    res_deep = det.predict_audio(dummy_audio, mode="deep")
    assert "prediction" in res_deep
    assert "suspicious_segments" in res_deep
    assert "benchmark_reference" in res_deep
    assert "timeline_summary" in res_deep
    print("[PASS] Deep Scan (Fusion 2.0) passed:", res_deep["prediction"], f"({res_deep['latency_ms']} ms)")


if __name__ == "__main__":
    test_detector_v2()
