"""
DisasterShield AI — FastAPI Main Application
Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import uuid
import json
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .schemas import (
    PredictRequest, PredictResponse,
    TrainRequest, TrainResponse,
    DatasetPreviewResponse, DatasetRecord,
    VisualizationsResponse,
    AlertRequest, AlertResponse,
    LogsResponse, LogEntry,
    HealthResponse,
)
from .auth import get_current_user, require_admin, auth_router
from .ml_pipeline import (
    load_and_preprocess, build_tfidf, train_ensemble_models,
    run_kmeans_clustering, save_all_artifacts, load_artifacts, clean_text,
)
from .visualizations import generate_all_charts

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("disastershield")

# In-memory event log (replace with DB in production)
EVENT_LOG: list = []
def _log_event(level: str, event: str, details: dict = {}):
    EVENT_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "event": event,
        "details": details,
    })
    if len(EVENT_LOG) > 500:
        EVENT_LOG.pop(0)

# ── App State ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"
DATA_PATH  = BASE_DIR / "data" / "disasters.csv"
CHART_DIR  = BASE_DIR / "charts"

artifacts: dict = {}   # loaded at startup
training_lock = asyncio.Lock()

# ── App Init ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="DisasterShield AI",
    description="ML-powered disaster prediction and management API for India (1990–2021)",
    version="1.0.0",
    contact={"name": "DisasterShield Team"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


# ── Startup ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global artifacts
    if (MODEL_DIR / "metadata.json").exists():
        logger.info("Loading pre-trained artifacts...")
        try:
            artifacts = load_artifacts(str(MODEL_DIR))
            _log_event("INFO", "startup_loaded_artifacts",
                       {"best_model": artifacts["meta"]["best_model"]})
            logger.info(f"Loaded. Best model: {artifacts['meta']['best_model']}")
        except Exception as e:
            logger.warning(f"Could not load artifacts: {e} — run /train first")
    else:
        logger.warning("No trained models found. POST /v1/train to train.")
        _log_event("WARN", "startup_no_models", {})


# ── Helper ────────────────────────────────────────────────────────────────
def _require_artifacts():
    if not artifacts:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. POST /v1/train first.",
        )


# ── Health ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=bool(artifacts),
        best_model=artifacts.get("meta", {}).get("best_model"),
    )


# ── PREDICT ───────────────────────────────────────────────────────────────
@app.post("/v1/predict", response_model=PredictResponse, tags=["Inference"],
          summary="Predict disaster type from text description")
async def predict(req: PredictRequest, user=Depends(get_current_user)):
    _require_artifacts()

    model_key = (
        artifacts["meta"]["best_model"] if req.model == "best" else req.model
    )
    if model_key not in artifacts["models"]:
        raise HTTPException(400, f"Unknown model: {model_key}")

    model      = artifacts["models"][model_key]
    vectorizer = artifacts["vectorizer"]
    km         = artifacts["km"]
    meta       = artifacts["meta"]

    # Vectorize
    vec    = vectorizer.transform([clean_text(req.description)])
    pred   = model.predict(vec)[0]
    proba  = dict(zip(model.classes_, [round(float(p), 4) for p in model.predict_proba(vec)[0]]))
    conf   = round(float(max(proba.values())), 4)

    # Cluster
    cluster_id  = int(km.predict(vec)[0])
    cluster_priority = meta["priority_map"].get(str(cluster_id), "Unknown")

    # Alert trigger
    alert_triggered = conf > 0.75 and cluster_priority == "High"
    alert_id        = None
    if alert_triggered:
        alert_id = f"ALT-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
        _log_event("WARN", "alert_triggered", {
            "disaster_type": pred, "confidence": conf,
            "cluster_priority": cluster_priority, "alert_id": alert_id
        })

    _log_event("INFO", "prediction_made", {
        "model": model_key, "predicted": pred, "confidence": conf,
        "cluster_priority": cluster_priority, "user": user.get("sub"),
    })

    return PredictResponse(
        predicted_class=pred,
        confidence=conf,
        probabilities=proba,
        cluster_id=cluster_id,
        cluster_priority=cluster_priority,
        model_used=model_key,
        model_accuracy=meta["accuracies"][model_key],
        model_f1=meta["f1_scores"][model_key],
        alert_triggered=alert_triggered,
        alert_id=alert_id,
    )


# ── TRAIN ─────────────────────────────────────────────────────────────────
@app.post("/v1/train", tags=["Training"],
          summary="Retrain all models (SSE streaming progress)")
async def train(
    body: TrainRequest = TrainRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user=Depends(require_admin),
):
    """
    Streams Server-Sent Events during training.
    Connect with: curl -N http://localhost:8000/v1/train -X POST -H "Content-Type: application/json"
    """
    global artifacts

    async def stream():
        stages = [
            ("loading_csv",      "Loading and preprocessing CSV dataset..."),
            ("tfidf_fit",        "Fitting TF-IDF vectorizer (max_features=5000, ngram 1-2)..."),
            ("rf_training",      "Training Random Forest (n_estimators=200)..."),
            ("adaboost_training","Training AdaBoost (n_estimators=100)..."),
            ("gb_training",      "Training Gradient Boosting (n_estimators=150)..."),
            ("kmeans",           "Running K-Means clustering (K=4, elbow K=2..10)..."),
            ("saving",           "Saving models, vectorizer, metadata to disk..."),
            ("done",             "Training complete!"),
        ]

        try:
            for i, (stage, msg) in enumerate(stages[:-1]):
                progress = int((i / (len(stages) - 1)) * 90)
                yield f"data: {json.dumps({'stage': stage, 'progress': progress, 'message': msg})}\n\n"
                await asyncio.sleep(0.1)

            # Actually train
            df          = load_and_preprocess(str(DATA_PATH))
            X, vectorizer = build_tfidf(df, max_features=body.max_features)
            results, best = train_ensemble_models(X, df["disaster_type"])
            km, df, p_map, inertias = run_kmeans_clustering(X, df)
            k_range      = list(range(2, 11))
            save_all_artifacts(results, best, vectorizer, km, p_map, df,
                               inertias, k_range, str(MODEL_DIR))

            # Reload
            artifacts = load_artifacts(str(MODEL_DIR))
            _log_event("INFO", "training_complete", {
                "best_model": best,
                "accuracies": {k: v["accuracy"] for k, v in results.items()},
                "user": user.get("sub"),
            })

            final = {
                "stage": "done", "progress": 100, "message": "Training complete!",
                "best_model": best,
                "accuracies": {k: round(v["accuracy"], 4) for k, v in results.items()},
                "f1_scores":  {k: round(v["f1_weighted"], 4) for k, v in results.items()},
            }
            yield f"data: {json.dumps(final)}\n\n"

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            yield f"data: {json.dumps({'stage': 'error', 'progress': -1, 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── DATASET PREVIEW ───────────────────────────────────────────────────────
@app.get("/v1/dataset-preview", response_model=DatasetPreviewResponse, tags=["Data"],
         summary="Paginated dataset view with extracted columns")
async def dataset_preview(
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    sort:      str = Query("year"),
    order:     str = Query("asc", pattern="^(asc|desc)$"),
    filter:    str = Query("", description="Filter by Title (case-insensitive)"),
    user=Depends(get_current_user),
):
    _require_artifacts()
    df = artifacts["df"].copy()

    # Filter
    if filter:
        df = df[df["Title"].str.lower().str.contains(filter.lower(), na=False)]

    # Sort
    sort_col_map = {
        "year": "Year", "title": "Title",
        "disaster_type": "disaster_type", "cluster_priority": "cluster_priority",
    }
    sort_col = sort_col_map.get(sort, "Year")
    df = df.sort_values(sort_col, ascending=(order == "asc"))

    total = len(df)
    pages = max(1, -(-total // page_size))  # ceiling division
    start = (page - 1) * page_size
    slice_df = df.iloc[start : start + page_size]

    records = []
    for i, (_, row) in enumerate(slice_df.iterrows()):
        records.append(DatasetRecord(
            record_id=int(start + i),
            year=int(row.get("Year", 0)),
            date=str(row.get("Date", "")),
            title=str(row.get("Title", "")),
            disaster_type=str(row.get("disaster_type", "")),
            cluster_priority=str(row.get("cluster_priority", "")),
            duration=str(row.get("Duration", "")) or None,
            snippet=str(row.get("Disaster_Info", ""))[:120] + "...",
        ))

    return DatasetPreviewResponse(
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
        data=records,
        class_counts=artifacts["df"]["disaster_type"].value_counts().to_dict(),
        priority_counts=artifacts["df"]["cluster_priority"].value_counts().to_dict(),
    )


# ── VISUALIZATIONS ────────────────────────────────────────────────────────
@app.get("/v1/visualizations", response_model=VisualizationsResponse, tags=["Visualizations"],
         summary="Get all 5 charts as base64 PNG")
async def visualizations(
    model: str = Query("best", description="Model: best | rf | adaboost | gb"),
    user=Depends(get_current_user),
):
    _require_artifacts()
    model_key = artifacts["meta"]["best_model"] if model == "best" else model
    if model_key not in artifacts["models"]:
        raise HTTPException(400, f"Unknown model: {model_key}")

    # Build results dict using the ALREADY-FITTED vectorizer (not re-fit)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix as _cm
    import json as _json

    df         = artifacts["df"]
    vectorizer = artifacts["vectorizer"]   # use saved vectorizer — same feature space as trained models
    X          = vectorizer.transform(df["clean_text"])

    class_counts  = df["disaster_type"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask          = df["disaster_type"].isin(valid_classes).values
    X_filtered    = X[mask]
    y_filtered    = df["disaster_type"][mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )

    # Try to load pre-computed confusion matrices from disk (fast path)
    cm_path = MODEL_DIR / "confusion_matrices.json"
    saved_cms = {}
    if cm_path.exists():
        with open(cm_path) as _f:
            saved_cms = _json.load(_f)

    partial_results = {}
    for name, mdl in artifacts["models"].items():
        y_pred = mdl.predict(X_te)
        # Use saved CM if available, otherwise compute from fresh predictions
        if name in saved_cms:
            cm_data = saved_cms[name]["matrix"]
        else:
            cm_data = _cm(y_te, y_pred, labels=artifacts["meta"]["classes"]).tolist()
        partial_results[name] = {
            "model":            mdl,
            "accuracy":         artifacts["meta"]["accuracies"][name],
            "f1_weighted":      artifacts["meta"]["f1_scores"][name],
            "confusion_matrix": cm_data,
            "classes":          artifacts["meta"]["classes"],
            "X_test":           X_te,
            "y_test":           y_te,
            "y_pred":           y_pred,
        }

    charts = generate_all_charts(partial_results, artifacts["meta"], df, model_key)
    _log_event("INFO", "visualizations_generated", {"model": model_key})

    return VisualizationsResponse(
        charts=charts,
        model=model_key,
        generated_at=datetime.now().isoformat(),
    )


# ── ALERTS/SIMULATE ───────────────────────────────────────────────────────
SEVERITY_MAP = {
    "High":        "CRITICAL",
    "Medium-High": "HIGH",
    "Medium-Low":  "MEDIUM",
    "Low":         "LOW",
}
ACTIONS_MAP = {
    "Flood":      "Evacuate low-lying areas; deploy NDRF teams; issue IMD flood advisory; open relief camps.",
    "Earthquake": "Activate SDRF; deploy search & rescue; inspect structures; set up medical camps.",
    "Cyclone":    "Issue coastal evacuation orders; secure fishing vessels; activate shelter-in-place protocol.",
    "Stampede":   "Deploy crowd control; set up medical triage; establish safe evacuation corridors.",
    "Fire":       "Deploy fire brigades; evacuate affected buildings; set up fire containment perimeter.",
    "Landslide":  "Block affected roads; deploy NDRF; arrange emergency shelter for hill communities.",
    "Epidemic":   "Activate health emergency; deploy medical teams; issue public health advisory.",
    "Aviation":   "Activate crash response team; deploy ARFF units; notify DGCA and families.",
    "Rail":       "Activate railway disaster management; deploy medical units; notify Railway Protection Force.",
    "Other":      "Activate disaster response protocol; alert local district administration.",
}

@app.post("/v1/alerts/simulate", response_model=AlertResponse, tags=["Alerts"],
          summary="Simulate CDOT/RAHAT emergency alert distribution")
async def simulate_alert(payload: AlertRequest, user=Depends(get_current_user)):
    dtype    = payload.disaster_type
    priority = payload.cluster_priority
    alert_id = f"ALT-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"

    _log_event("WARN", "alert_simulated", {
        "alert_id": alert_id, "disaster_type": dtype,
        "priority": priority, "confidence": payload.probability,
    })

    return AlertResponse(
        alert_id=alert_id,
        timestamp=datetime.now().isoformat(),
        disaster_type=dtype,
        severity=SEVERITY_MAP.get(priority, "UNKNOWN"),
        zone=f"{priority} Priority Zone",
        recommended_action=ACTIONS_MAP.get(dtype, ACTIONS_MAP["Other"]),
        simulated_recipients=[
            "NDRF HQ (National Disaster Response Force)",
            "State Disaster Management Authority",
            "IMD Alert System (India Met. Dept.)",
            "CDOT Emergency Network (Simulated)",
            "RAHAT Disaster Platform (Simulated)",
        ],
        delivery_channels=["SMS_BULK", "IVR_CALL", "MOBILE_APP_PUSH", "EMAIL_BLAST"],
    )


# ── LOGS ──────────────────────────────────────────────────────────────────
@app.get("/v1/logs", response_model=LogsResponse, tags=["System"],
         summary="View system event log")
async def get_logs(
    level:  str = Query("", description="Filter by level: INFO | WARN | ERROR"),
    limit:  int = Query(100, ge=1, le=500),
    user=Depends(require_admin),
):
    logs = EVENT_LOG.copy()
    if level:
        logs = [l for l in logs if l["level"] == level.upper()]
    logs = logs[-limit:][::-1]   # newest first
    return LogsResponse(
        logs=[LogEntry(**l) for l in logs],
        total=len(logs),
    )


# ── MODEL INFO ────────────────────────────────────────────────────────────
@app.get("/v1/models", tags=["Inference"],
         summary="List available models and their metrics")
async def list_models(user=Depends(get_current_user)):
    _require_artifacts()
    meta = artifacts["meta"]
    return {
        "available_models": ["rf", "adaboost", "gb"],
        "best_model": meta["best_model"],
        "accuracies": meta["accuracies"],
        "f1_scores":  meta["f1_scores"],
        "classes":    meta["classes"],
        "n_records":  meta["n_records"],
        "trained_at": meta["trained_at"],
    }
