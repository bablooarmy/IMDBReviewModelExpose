# IMDB Review Sentiment Inference

This project serves an IMDB review sentiment model with OpenVINO Model Server
(OVMS) and a small FastAPI service. The FastAPI service accepts raw review
text, preprocesses it with the model's TF-IDF vectorizer, and sends the
resulting numeric features to OVMS for prediction.

## Architecture

```text
Client
  | POST /predict (raw review text)
  v
FastAPI sentiment-api (:9898)
  | tfidf_vectorizer.pkl transforms each review to 10,000 FP32 features
  | POST /v2/models/movie_review_sentiment_model/infer
  v
OpenVINO Model Server ovms (:8091 REST, :8001 gRPC)
  | logistic_regression_openvino.xml + .bin
  v
Predicted sentiment returned by FastAPI
```

The model repository contains both the OpenVINO model and its matching
`tfidf_vectorizer.pkl`:

```text
model_repository/
└── movie_review_sentiment_model/
    └── 1/
        ├── logistic_regression_openvino.xml
        ├── logistic_regression_openvino.bin
        └── tfidf_vectorizer.pkl
```

Compose mounts `model_repository` into the OVMS container at `/models`.
The Dockerfile also copies `tfidf_vectorizer.pkl` from this repository into
the FastAPI image, where `app.py` loads it at startup. Thus the vectorizer is
the preprocessing artifact kept alongside the served model, but vectorization
is performed by the API container, not by OVMS. The OpenVINO model expects
each review as a row of 10,000 FP32 TF-IDF features.

## Run with Docker Compose

From this directory, build and start both services:

```powershell
docker compose up --build
```

To stop the services, press `Ctrl+C`, then run:

```powershell
docker compose down
```

The services are available at:

| Service | Address | Purpose |
| --- | --- | --- |
| Sentiment API | `http://localhost:9898` | Accepts raw reviews at `/predict` |
| OVMS REST API | `http://localhost:8091` | Serves model inference |
| OVMS gRPC | `localhost:8001` | OVMS gRPC endpoint |

## Make a prediction

Send one or more reviews to the FastAPI endpoint:

```powershell
curl.exe -X POST "http://localhost:9898/predict" `
  -H "Content-Type: application/json" `
  -d '{"reviews":["This movie was fantastic!","Boring and dull."]}'
```

Example response:

```json
{
  "results": [
    {
      "review": "This movie was fantastic!",
      "prediction": "Positive 😃",
      "raw_label": 1
    },
    {
      "review": "Boring and dull.",
      "prediction": "Negative 😞",
      "raw_label": 0
    }
  ]
}
```

The request body must contain a non-empty `reviews` array of strings. The API
vectorizes each review, converts the features to FP32, and sends them to the
OVMS model named `movie_review_sentiment_model`. The API maps model label `1`
to Positive and `0` to Negative.

## Updating model artifacts

Keep the XML, BIN, and vectorizer artifacts together under
`model_repository/movie_review_sentiment_model/1/`. If the vectorizer changes,
rebuild the API image so it includes the matching file:

```powershell
docker compose up --build
```
