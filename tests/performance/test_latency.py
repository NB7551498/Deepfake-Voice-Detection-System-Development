import time
import requests
import io
import numpy as np
import scipy.io.wavfile as wav

API_URL = "http://127.0.0.1:8001"


def benchmark_latency(n_iterations: int = 5):
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav.write(buf, sr, audio)
    raw_wav = buf.getvalue()

    latencies = []
    for _ in range(n_iterations):
        t0 = time.time()
        files = {"file": ("bench.wav", raw_wav, "audio/wav")}
        r = requests.post(f"{API_URL}/predict?mode=fast", files=files)
        latencies.append((time.time() - t0) * 1000)

    print(f"Latency Benchmark ({n_iterations} runs on CPU):")
    print(f"  p50 Latency: {np.percentile(latencies, 50):.1f} ms")
    print(f"  p95 Latency: {np.percentile(latencies, 95):.1f} ms")
    print(f"  Mean Latency: {np.mean(latencies):.1f} ms")


if __name__ == "__main__":
    benchmark_latency()
