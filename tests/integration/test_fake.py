import requests

with open("data/processed/train/fake/fake_001.wav", "rb") as f:
    files = {"file": ("fake.wav", f.read(), "audio/wav")}

r = requests.post("http://127.0.0.1:8001/predict?mode=deep", files=files)
data = r.json()
print("Fake Audio Test (Deep Mode):")
print("  Prediction:      ", data.get("prediction"))
print("  Fake Probability:", data.get("fake_probability"))
print("  Confidence:      ", data.get("confidence"))
print("  Evidence:        ", data.get("segments", [{}])[0].get("evidence"))
