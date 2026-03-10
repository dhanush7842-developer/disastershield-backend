"""
DisasterShield AI — Standalone Training Script
Run from project root: python train.py

Trains all 3 models on the real CSV, saves artifacts, generates all 5 charts.
"""

import sys
import logging
from pathlib import Path

# Add app/ to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data"  / "disasters.csv"
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "charts"

from ml_pipeline import (
    load_and_preprocess, build_tfidf,
    train_ensemble_models, run_kmeans_clustering,
    save_all_artifacts,
)
from visualizations import generate_all_charts, save_charts_to_disk
from sklearn.metrics import classification_report
import json


def main():
    print("\n" + "═"*60)
    print("  DisasterShield AI — Training Pipeline")
    print("═"*60 + "\n")

    # 1. Load & preprocess
    logger.info("Step 1/6 — Loading and preprocessing dataset")
    df = load_and_preprocess(str(DATA_PATH))
    print(f"\n  Dataset: {len(df)} records loaded")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Disaster type distribution:")
    for dtype, cnt in df["disaster_type"].value_counts().items():
        bar = "█" * cnt
        print(f"    {dtype:<14} {cnt:>3}  {bar}")

    # 2. TF-IDF
    logger.info("Step 2/6 — Building TF-IDF features")
    X, vectorizer = build_tfidf(df, max_features=5000)
    print(f"\n  TF-IDF matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # 3. Train models
    logger.info("Step 3/6 — Training ensemble models")
    results, best_name = train_ensemble_models(X, df["disaster_type"])

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │         Model Performance Summary        │")
    print("  ├──────────────────┬──────────┬───────────┤")
    print("  │ Model            │ Accuracy │  F1 Score │")
    print("  ├──────────────────┼──────────┼───────────┤")
    for name, r in results.items():
        marker = " ★" if name == best_name else "  "
        label  = {"rf":"Random Forest","adaboost":"AdaBoost","gb":"Grad. Boosting"}.get(name, name)
        print(f"  │ {label:<16}  │  {r['accuracy']:.4f}  │   {r['f1_weighted']:.4f}  │{marker}")
    print("  └──────────────────┴──────────┴───────────┘")
    print(f"\n  ★ Best model selected: {best_name.upper()}")

    # 4. K-Means
    logger.info("Step 4/6 — K-Means clustering (K=4)")
    k_range = range(2, 11)
    km, df, priority_map, inertias = run_kmeans_clustering(X, df, n_clusters=4, k_range=k_range)

    print(f"\n  Cluster → Priority mapping: {priority_map}")
    print("  Priority zone distribution:")
    for zone, cnt in df["cluster_priority"].value_counts().items():
        print(f"    {zone:<14}: {cnt} records")

    # 5. Save artifacts
    logger.info("Step 5/6 — Saving artifacts")
    save_all_artifacts(
        results, best_name, vectorizer, km, priority_map, df,
        inertias, list(k_range), str(MODEL_DIR)
    )
    print(f"\n  Artifacts saved to: {MODEL_DIR}/")

    # 6. Generate charts
    logger.info("Step 6/6 — Generating visualizations")
    # Rebuild partial_results for chart generation
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import json

    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)

    X2, _ = build_tfidf(df, max_features=5000)
    y_vis = df["disaster_type"].copy()
    y_vis[y_vis == "Landslide"] = "Other"
    X_tr, X_te, y_tr, y_te = train_test_split(
        X2, y_vis, test_size=0.2, random_state=42, stratify=y_vis
    )
    partial_results = {}
    for name, r in results.items():
        y_pred = r["model"].predict(X_te)
        partial_results[name] = {
            "model":            r["model"],
            "accuracy":         r["accuracy"],
            "f1_weighted":      r["f1_weighted"],
            "confusion_matrix": confusion_matrix(y_te, y_pred, labels=meta["classes"]).tolist(),
            "classes":          meta["classes"],
            "X_test":           X_te,
            "y_test":           y_te,
            "y_pred":           y_pred,
        }

    charts = generate_all_charts(partial_results, meta, df, best_name)
    save_charts_to_disk(charts, str(CHART_DIR))
    print(f"  Charts saved to: {CHART_DIR}/")

    # Print classification report for best model
    print(f"\n  Classification Report — {best_name.upper()}:")
    print("  " + "-"*50)
    best_r = results[best_name]
    report = classification_report(
        best_r["y_test"], best_r["y_pred"], zero_division=0
    )
    for line in report.split("\n"):
        print("  " + line)

    print("\n" + "═"*60)
    print("  ✅  Training complete! Backend is ready.")
    print("  ▶   Start server: cd app && uvicorn main:app --reload")
    print("  📖  API docs:      http://localhost:8000/docs")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
