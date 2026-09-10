import sys
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_experiments import generate_local_benchmark_data

print("Generating features for Random Forest training...")
X, y = generate_local_benchmark_data(n_samples=200)

print(f"Training Random Forest on {len(X)} samples with 228 features...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf.fit(X, y)

out_dir = Path("models/production")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rf_model.pkl"

with open(out_path, "wb") as f:
    pickle.dump(rf, f)

print(f"✅ Random Forest model successfully saved to {out_path}")
