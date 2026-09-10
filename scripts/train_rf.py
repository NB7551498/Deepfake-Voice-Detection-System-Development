import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.extractor import extract_chunk_features
import librosa

TRAIN_DIR = Path("data/processed/train")

print("Extracting features from robust dataset for Random Forest...")
X_list, y_list = [], []

for label_name, label_val in [("real", 0), ("fake", 1)]:
    files = list((TRAIN_DIR / label_name).glob("*.wav"))
    print(f"  Processing {len(files)} {label_name} files...")
    for fp in files:
        audio, sr = librosa.load(str(fp), sr=16000, mono=True)
        # Take 3s chunk
        if len(audio) >= 48000:
            chunk = audio[:48000]
        else:
            chunk = np.pad(audio, (0, 48000 - len(audio)))
        feat = extract_chunk_features(chunk, sr=16000)
        X_list.append(feat)
        y_list.append(label_val)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

print(f"Training Random Forest on {len(X)} samples with 228 features...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf.fit(X, y)

out_path = Path("models/production/rf_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump(rf, f)

print(f"✅ Random Forest model successfully saved to {out_path}")
