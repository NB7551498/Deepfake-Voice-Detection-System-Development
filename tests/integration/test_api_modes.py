import io
import requests
import numpy as np
import scipy.io.wavfile as wav

# 4-second test waveform
sr = 16000
t = np.linspace(0, 4.0, sr * 4, endpoint=False)
audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
buf = io.BytesIO()
wav.write(buf, sr, audio)
wav_bytes = buf.getvalue()

# 1. Test Fast Scan
files = {'file': ('test.wav', wav_bytes, 'audio/wav')}
r_fast = requests.post('http://127.0.0.1:8001/predict?mode=fast', files=files)
assert r_fast.status_code == 200, f"Fast mode failed: {r_fast.text}"
data_fast = r_fast.json()
print("✅ Fast Scan Mode:")
print("   Prediction:", data_fast.get("prediction"))
print("   Stage Used:", data_fast.get("stage_used"))
print("   Latency:   ", data_fast.get("latency_ms"), "ms")
print("   Confidence:", data_fast.get("confidence"))

# 2. Test Deep Scan
files = {'file': ('test.wav', wav_bytes, 'audio/wav')}
r_deep = requests.post('http://127.0.0.1:8001/predict?mode=deep', files=files)
assert r_deep.status_code == 200, f"Deep mode failed: {r_deep.text}"
data_deep = r_deep.json()
print("\n✅ Deep Scan Mode:")
print("   Prediction:", data_deep.get("prediction"))
print("   Stage Used:", data_deep.get("stage_used"))
print("   Latency:   ", data_deep.get("latency_ms"), "ms")
print("   Segments:  ", len(data_deep.get("segments", [])))
print("   Confidence:", data_deep.get("confidence"))

# 3. Test Experiments endpoint
r_exp = requests.get('http://127.0.0.1:8001/experiments')
assert r_exp.status_code == 200
print("\n✅ Experiments & Ablation endpoint working:", len(r_exp.json().get("ablation_study", [])), "models benchmarked.")
