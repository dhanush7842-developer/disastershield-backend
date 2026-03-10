"""
DisasterShield AI — Pydantic Schemas
All request/response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any


# ── Auth ──────────────────────────────────────────────────────────────────
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


# ── Predict ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Free-text disaster description (20–2000 characters)",
        examples=["Heavy monsoon floods submerged dozens of villages in Assam"],
    )
    model: str = Field(
        "best",
        description="Model to use: best | rf | adaboost | gb",
        examples=["best"],
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if v not in {"best", "rf", "adaboost", "gb"}:
            raise ValueError("model must be one of: best, rf, adaboost, gb")
        return v


class PredictResponse(BaseModel):
    predicted_class:  str
    confidence:       float
    probabilities:    Dict[str, float]
    cluster_id:       int
    cluster_priority: str
    model_used:       str
    model_accuracy:   float
    model_f1:         float
    alert_triggered:  bool
    alert_id:         Optional[str] = None


# ── Train ─────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    force_retrain: bool = True
    max_features:  int  = Field(5000, ge=100, le=50000)


class TrainResponse(BaseModel):
    status:       str
    best_model:   str
    accuracies:   Dict[str, float]
    f1_scores:    Dict[str, float]
    train_times:  Dict[str, float]
    n_records:    int
    trained_at:   str


# ── Dataset Preview ───────────────────────────────────────────────────────
class DatasetRecord(BaseModel):
    record_id:        int
    year:             int
    date:             str
    title:            str
    disaster_type:    str
    cluster_priority: str
    duration:         Optional[str] = None
    snippet:          str


class DatasetPreviewResponse(BaseModel):
    total:      int
    page:       int
    page_size:  int
    pages:      int
    data:       List[DatasetRecord]
    class_counts: Dict[str, int]
    priority_counts: Dict[str, int]


# ── Visualizations ────────────────────────────────────────────────────────
class VisualizationsResponse(BaseModel):
    charts:       Dict[str, str]   # name → base64 PNG
    model:        str
    generated_at: str


# ── Alert ─────────────────────────────────────────────────────────────────
class AlertRequest(BaseModel):
    disaster_type:    str
    probability:      float
    cluster_priority: str
    description:      Optional[str] = ""


class AlertResponse(BaseModel):
    alert_id:             str
    timestamp:            str
    disaster_type:        str
    severity:             str
    zone:                 str
    recommended_action:   str
    simulated_recipients: List[str]
    delivery_channels:    List[str]
    simulated:            bool = True


# ── Logs ──────────────────────────────────────────────────────────────────
class LogEntry(BaseModel):
    timestamp: str
    level:     str
    event:     str
    details:   Dict[str, Any] = {}


class LogsResponse(BaseModel):
    logs:  List[LogEntry]
    total: int


# ── Health ────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:        str
    models_loaded: bool
    best_model:    Optional[str] = None
    version:       str = "1.0.0"
