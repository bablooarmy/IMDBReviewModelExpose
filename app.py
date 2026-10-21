import os
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Movie Review Sentiment API", version="1.0")

# Configuration from environment variables (defaults to Docker Compose service names)
OVMS_URL = os.getenv(
    "OVMS_URL", 
    "http://ovms:8091/v2/models/movie_review_sentiment_model/infer"
)
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "tfidf_vectorizer.pkl")

# Load the vectorizer once on startup
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Vectorizer loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to load vectorizer: {e}")
    vectorizer = None

# Define input schema for JSON payloads
class ReviewRequest(BaseModel):
    reviews: list[str] = Field(..., example=["This movie was fantastic!", "Boring and dull."])

@app.post("/predict")
def predict_sentiment(payload: ReviewRequest):
    if not vectorizer:
        raise HTTPException(status_code=500, detail="Vectorizer is not loaded on the server.")
    if not payload.reviews:
        raise HTTPException(status_code=400, detail="Review list cannot be empty.")

    try:
        # Step 1: Transform text strings to dense TF-IDF numerical features
        X_new_vec = vectorizer.transform(payload.reviews).toarray().astype(np.float32)

        print(f"Transformed {len(payload.reviews)} reviews into TF-IDF features with shape: {X_new_vec.shape}")

        # Step 2: Format payload for OpenVINO Model Server (OVMS) REST API
        ovms_payload = {
            "inputs": [
                {
                    "name": "float_input",
                    "shape": list(X_new_vec.shape),
                    "datatype": "FP32",
                    "data": X_new_vec.tolist(),
                }
            ]
        }

        print(f"Sending payload to OVMS at {OVMS_URL} with shape: {X_new_vec.shape}")

        # Step 3: Send request to OpenVINO Model Server container
        response = requests.post(OVMS_URL, json=ovms_payload)

        print(f"Received response from OVMS with status code: {response.status_code}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OVMS inference error: {response.text}")

        result_data = response.json()
        raw_preds = result_data["outputs"][0]["data"]

        # Define label mapping
        label_map = {1: "Positive 😃", 0: "Negative 😞"}

        # Step 4: Format results
        results = []
        for review, pred in zip(payload.reviews, raw_preds):
            pred_val = int(pred)
            sentiment = label_map.get(pred_val, "Unknown")
            results.append({
                "review": review,
                "prediction": sentiment,
                "raw_label": pred_val
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))