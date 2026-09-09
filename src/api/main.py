from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
from pydantic import BaseModel
from typing import List

from src.audio.preprocessing import AudioPreprocessor
# from src.models.ensemble.ensemble import DeepfakeEnsemble # Uncomment to load real model

app = FastAPI(title="Deepfake Voice Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SegmentResult(BaseModel):
    start: float
    end: float
    fake_probability: float

class PredictionResponse(BaseModel):
    prediction: str
    fake_probability: float
    real_probability: float
    confidence: float
    segments: List[SegmentResult]

# Global instances
preprocessor = AudioPreprocessor()
# model = DeepfakeEnsemble()
# model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
# model.eval()

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid audio format")
        
    audio_bytes = await file.read()
    
    try:
        waveform = preprocessor.load_and_preprocess(audio_bytes)
        chunks = preprocessor.chunk_audio(waveform)
        
        segment_results = []
        overall_fake_prob = 0.0
        
        # Mocking inference for the boilerplate
        import random
        for chunk, start_s, end_s in chunks:
            # chunk_tensor = chunk.unsqueeze(0)
            # with torch.no_grad():
            #     prob = model(chunk_tensor).item()
            prob = random.uniform(0.1, 0.9)
            segment_results.append(SegmentResult(start=start_s, end=end_s, fake_probability=prob))
            overall_fake_prob += prob
            
        if len(segment_results) > 0:
            overall_fake_prob /= len(segment_results)
            
        # Calibration / Thresholding logic (Phase 6)
        if overall_fake_prob > 0.65:
            prediction = "DEEPFAKE"
        elif overall_fake_prob < 0.35:
            prediction = "REAL"
        else:
            prediction = "UNCERTAIN"
            
        return PredictionResponse(
            prediction=prediction,
            fake_probability=overall_fake_prob,
            real_probability=1 - overall_fake_prob,
            confidence=abs(overall_fake_prob - 0.5) * 2, # naive confidence mapping
            segments=segment_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
