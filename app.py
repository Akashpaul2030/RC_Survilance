"""
FastAPI server for military audio classification.
Endpoints: /predict (file upload), /stream (WebSocket), /health
"""

import io
import os
import time
import tempfile
from contextlib import asynccontextmanager

import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from inference import AudioClassifier, LABELS, SAMPLE_RATE


classifier: AudioClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    model_dir = os.environ.get("MODEL_DIR", "model_onnx_int8")
    num_threads = int(os.environ.get("ORT_NUM_THREADS", "4"))
    print(f"Loading model from {model_dir} with {num_threads} threads...")
    classifier = AudioClassifier(model_dir=model_dir, num_threads=num_threads)
    print("Model loaded and ready.")
    yield
    classifier = None


app = FastAPI(
    title="Military Audio Classifier - Edge Deployment",
    description="ONNX-based audio classifier for military surveillance",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if classifier is not None else "not_ready",
        "model_loaded": classifier is not None,
        "labels": LABELS,
        "latency_budget_ms": 200.0,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded audio file.
    Accepts WAV, MP3, FLAC, OGG formats.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read uploaded file
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # Load audio from bytes using librosa
        audio, sr = librosa.load(io.BytesIO(contents), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    result = classifier.predict(audio, sr)

    return {
        "prediction": result.label,
        "confidence": round(result.confidence, 4),
        "scores": {k: round(v, 4) for k, v in result.all_scores.items()},
        "latency_ms": result.latency_ms,
        "within_budget": result.within_budget,
    }


@app.websocket("/stream")
async def stream(ws: WebSocket):
    """
    WebSocket endpoint for streaming audio classification.
    Expects raw 16-bit PCM audio chunks (16kHz, mono).
    Sends back JSON predictions for each chunk.
    """
    await ws.accept()

    if classifier is None:
        await ws.send_json({"error": "Model not loaded"})
        await ws.close()
        return

    try:
        while True:
            # Receive raw audio bytes
            data = await ws.receive_bytes()

            if len(data) < 1600:  # Less than 0.05s of audio at 16kHz 16-bit
                await ws.send_json({"error": "Audio chunk too short"})
                continue

            result = classifier.predict_bytes(data)

            await ws.send_json({
                "prediction": result.label,
                "confidence": round(result.confidence, 4),
                "scores": {k: round(v, 4) for k, v in result.all_scores.items()},
                "latency_ms": result.latency_ms,
                "within_budget": result.within_budget,
            })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
