from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Mock analysis
    overall_fake_prob = random.uniform(0.6, 0.95)
    
    segments = []
    for i in range(5):
        segments.append({
            "start": i * 2.0,
            "end": (i * 2.0) + 3.0,
            "fake_probability": random.uniform(0.1, 0.99)
        })
        
    return {
        "prediction": "DEEPFAKE" if overall_fake_prob > 0.65 else "REAL",
        "fake_probability": overall_fake_prob,
        "real_probability": 1 - overall_fake_prob,
        "confidence": overall_fake_prob,
        "segments": segments
    }

if __name__ == "__main__":
    print("Starting Mock Deepfake API on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
