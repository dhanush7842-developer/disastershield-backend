# DisasterShield AI — Backend

ML-powered disaster prediction API for India Natural Disasters 1990–2021.

## Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (one-time, ~30 seconds)
```bash
python train.py
```
This will:
- Load and preprocess the 207-record CSV dataset
- Train Random Forest, AdaBoost, and Gradient Boosting classifiers
- Run K-Means clustering (K=4) for priority zone mapping
- Save all models + metadata to `models/`
- Generate 5 visualization charts to `charts/`

### 3. Start the API server
```bash
cd app
uvicorn main:app --reload --port 8000
```

### 4. Open API docs
Visit: http://localhost:8000/docs

---

## Quick Start (Docker)
```bash
docker compose up --build
```
API available at: http://localhost:8000

---

## Project Structure
```
disastershield/
├── app/
│   ├── main.py           # FastAPI app — all endpoints
│   ├── ml_pipeline.py    # Preprocessing + training + clustering
│   ├── visualizations.py # All 5 chart generators
│   ├── schemas.py        # Pydantic request/response models
│   └── auth.py           # JWT auth (no external libs needed)
├── data/
│   └── disasters.csv     # Natural Disasters India 1990-2021
├── models/               # Trained artifacts (created by train.py)
│   ├── rf.pkl
│   ├── adaboost.pkl
│   ├── gb.pkl
│   ├── vectorizer.pkl
│   ├── kmeans.pkl
│   ├── metadata.json
│   └── processed_data.csv
├── charts/               # Generated PNG charts (created by train.py)
├── tests/
│   └── test_pipeline.py  # pytest unit tests
├── train.py              # Standalone training script
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /auth/token | Get JWT token |
| POST | /v1/predict | Predict disaster type |
| POST | /v1/train | Retrain all models (SSE stream) |
| GET | /v1/dataset-preview | Paginated dataset view |
| GET | /v1/visualizations | All 5 charts as base64 |
| POST | /v1/alerts/simulate | Simulate CDOT/RAHAT alert |
| GET | /v1/models | Model metrics |
| GET | /v1/logs | Event log (admin only) |

---

## Authentication

By default `AUTH_REQUIRED=false` so you can test without tokens.

To enable JWT auth, set `AUTH_REQUIRED=true` and get a token:
```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=analyst&password=analyst123"
```

Demo users:
- `analyst / analyst123` — read access
- `admin / admin123` — full access including /train and /logs

---

## Example: Predict
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Heavy monsoon floods submerged dozens of villages in Assam causing massive displacement", "model": "best"}'
```

## Example: Retrain (SSE stream)
```bash
curl -N -X POST http://localhost:8000/v1/train \
  -H "Content-Type: application/json"
```

## Run Tests
```bash
cd app && python -m pytest ../tests/ -v
```

---

## Model Performance (on your dataset)

| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| Random Forest ★ | 0.7143 | 0.6861 |
| Gradient Boosting | 0.6667 | 0.6281 |
| AdaBoost | 0.4286 | 0.4056 |

> Note: The dataset has only 207 records with imbalanced classes (67 Flood, 1 Landslide).
> Accuracy improves significantly with more data. Landslide merged into Other due to single sample.

## Class Distribution
- Flood: 67 | Earthquake: 37 | Fire: 31 | Other: 24 | Rail: 19 | Stampede: 18 | Epidemic: 5 | Aviation: 3 | Cyclone: 3
