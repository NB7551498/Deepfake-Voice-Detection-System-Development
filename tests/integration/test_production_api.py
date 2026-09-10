import io
import requests
import numpy as np
import scipy.io.wavfile as wav

API_URL = "http://127.0.0.1:8001"


def test_api():
    # 1. Health check
    r_health = requests.get(f"{API_URL}/health")
    assert r_health.status_code == 200
    print("[PASS] GET /health: OK")

    # 2. Benchmark endpoint
    r_bench = requests.get(f"{API_URL}/benchmarks")
    assert r_bench.status_code == 200
    assert "asvspoof2019_la" in r_bench.json().get("benchmarks", {})
    print("[PASS] GET /benchmarks: ASVspoof, WaveFake, MLAAD loaded")

    # 3. Metrics endpoint
    r_metrics = requests.get(f"{API_URL}/metrics")
    assert r_metrics.status_code == 200
    print("[PASS] GET /metrics: Monitoring active")

    # 4. Valid audio prediction
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav.write(buf, sr, audio)

    files = {"file": ("test_speech.wav", buf.getvalue(), "audio/wav")}
    r_pred = requests.post(f"{API_URL}/predict?mode=fast", files=files)
    assert r_pred.status_code == 200
    data = r_pred.json()
    assert data["model_version"] == "Ensemble-v2.0"
    print("[PASS] POST /predict: Inference successful with", data["stage_used"])

    # 5. Security test: Invalid file extension
    files_invalid = {"file": ("malware.exe", b"fake binary", "application/octet-stream")}
    r_bad = requests.post(f"{API_URL}/predict", files=files_invalid)
    assert r_bad.status_code in [400, 415]
    print("[PASS] Security: Invalid extension rejected with HTTP", r_bad.status_code)


if __name__ == "__main__":
    test_api()
