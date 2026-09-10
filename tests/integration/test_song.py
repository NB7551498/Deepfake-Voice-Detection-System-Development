import requests

with open("data/processed/train/real/real_002.wav", "rb") as f:
    files = {"file": ("real_song.wav", f.read(), "audio/wav")}

# Test fast mode
r_fast = requests.post("http://127.0.0.1:8001/predict?mode=fast", files=files)
data_fast = r_fast.json()
print("Real Song Test (Fast Mode):")
print("  Prediction:      ", data_fast.get("prediction"))
print("  Fake Probability:", data_fast.get("fake_probability"))
print("  Confidence:      ", data_fast.get("confidence"))
print("  Stage Used:      ", data_fast.get("stage_used"))

# Test deep mode
with open("data/processed/train/real/real_002.wav", "rb") as f:
    files = {"file": ("real_song.wav", f.read(), "audio/wav")}
r_deep = requests.post("http://127.0.0.1:8001/predict?mode=deep", files=files)
data_deep = r_deep.json()
print("\nReal Song Test (Deep Mode):")
print("  Prediction:      ", data_deep.get("prediction"))
print("  Fake Probability:", data_deep.get("fake_probability"))
print("  Confidence:      ", data_deep.get("confidence"))
print("  Evidence:        ", data_deep.get("segments", [{}])[0].get("evidence"))
