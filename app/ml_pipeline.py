"""
DisasterShield AI — ML Pipeline
Handles: preprocessing, TF-IDF, ensemble training, K-Means clustering
No NLTK required — stop-words are bundled.
"""

import pandas as pd
import numpy as np
import re
import pickle
import json
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ── Bundled English stop-words (no NLTK needed) ──────────────────────────
STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","get","got","had","hadn't","has","hasn't",
    "have","haven't","having","he","he'd","he'll","he's","her","here","here's",
    "hers","herself","him","himself","his","how","how's","i","i'd","i'll",
    "i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's","me","more","most","mustn't","my","myself","no","nor","not","of",
    "off","on","once","only","or","other","ought","our","ours","ourselves",
    "out","over","own","same","shan't","she","she'd","she'll","she's","should",
    "shouldn't","so","some","such","than","that","that's","the","their",
    "theirs","them","themselves","then","there","there's","these","they",
    "they'd","they'll","they're","they've","this","those","through","to","too",
    "under","until","up","very","was","wasn't","we","we'd","we'll","we're",
    "we've","were","weren't","what","what's","when","when's","where","where's",
    "which","while","who","who's","whom","why","why's","will","with","won't",
    "would","wouldn't","you","you'd","you'll","you're","you've","your","yours",
    "yourself","yourselves","also","may","said","one","two","three","many",
    "also","however","therefore","thus","hence","among","within","without",
    "upon","since","still","already","just","even","around","near","far",
    "due","per","via","et","al","eg","ie","vs","etc","india","indian",
}

# ── Disaster keyword taxonomy ─────────────────────────────────────────────
DISASTER_TAXONOMY: Dict[str, list] = {
    "Flood":       ["flood","inundation","submerged","overflow","monsoon rain",
                    "river burst","flash flood","waterlog","deluge","dam break"],
    "Earthquake":  ["earthquake","tremor","seismic","magnitude","richter","epicenter",
                    "fault","aftershock","quake","temblor","tectonic"],
    "Cyclone":     ["cyclone","hurricane","typhoon","storm surge","windspeed","tropical storm",
                    "depression","super cyclonic","coastal storm","bay of bengal storm"],
    "Stampede":    ["stampede","crowd crush","trampled","pilgrims crush","rush crowd",
                    "crowd panic","human crush","overcrowd"],
    "Fire":        ["fire","blaze","arson","inferno","flames","combustion","burnt",
                    "fire tragedy","cinema fire","building fire","factory fire"],
    "Landslide":   ["landslide","mudslide","rockfall","debris flow","slope failure",
                    "hill collapse","landslip","mud flow"],
    "Epidemic":    ["plague","epidemic","outbreak","cholera","typhoid","disease spread",
                    "bubonic","pneumonic","infection","viral outbreak","dengue"],
    "Aviation":    ["aircraft","airplane","airline","flight","crash landing","pilot error",
                    "air crash","aviation","runway","cockpit","boeing","airbus"],
    "Rail":        ["train","railway","derail","rail collision","express train","rail accident",
                    "track","locomotive","passenger train","rail disaster"],
    "Other":       [],
}


def clean_text(text: str) -> str:
    """Normalize, remove noise, strip stop-words."""
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def extract_disaster_type(text: str) -> str:
    """Rule-based disaster type extraction from raw text."""
    text_lower = str(text).lower()
    for dtype, keywords in DISASTER_TAXONOMY.items():
        if dtype == "Other":
            continue
        if any(kw in text_lower for kw in keywords):
            return dtype
    return "Other"


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load CSV and enrich with clean_text + disaster_type columns."""
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Rename unnamed index col if present
    if "" in df.columns:
        df = df.rename(columns={"": "record_id"})

    df["clean_text"]     = df["Disaster_Info"].apply(clean_text)
    df["disaster_type"]  = df["Disaster_Info"].apply(extract_disaster_type)
    df["Year"]           = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)

    logger.info(f"Loaded {len(df)} records | Classes: {df['disaster_type'].value_counts().to_dict()}")
    return df


def build_tfidf(df: pd.DataFrame, max_features: int = 5000):
    """Fit TF-IDF vectorizer and return (matrix, vectorizer)."""
    logger.info(f"Fitting TF-IDF (max_features={max_features})")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(df["clean_text"])
    logger.info(f"TF-IDF matrix shape: {X.shape}")
    return X, vectorizer


def train_ensemble_models(
    X, y, random_state: int = 42
) -> Tuple[Dict, str]:
    """
    Train RF, AdaBoost, GradientBoosting.
    Returns (results_dict, best_model_name).
    """
    # Drop classes with fewer than 2 samples (cannot stratify)
    import scipy.sparse as sp_mod
    import pandas as _pd
    y_series = _pd.Series(y) if not hasattr(y, 'value_counts') else y
    rare = y_series.value_counts()
    rare = rare[rare < 2].index.tolist()
    if rare:
        logger.warning(f"Merging rare classes {rare} into 'Other'")
        y_series = y_series.copy()
        y_series[y_series.isin(rare)] = 'Other'
        y = y_series
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model_configs = {
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=random_state,

        ),
        "gb": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state,
        ),
    }

    results = {}
    for name, model in model_configs.items():
        logger.info(f"Training {name.upper()}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        y_pred = model.predict(X_test)
        acc    = round(accuracy_score(y_test, y_pred), 4)
        f1     = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred, labels=sorted(y.unique())).tolist()

        results[name] = {
            "model":       model,
            "accuracy":    acc,
            "f1_weighted": f1,
            "train_time":  elapsed,
            "report":      report,
            "confusion_matrix": cm,
            "classes":     sorted(y.unique()),
            "X_test":      X_test,
            "y_test":      y_test,
            "y_pred":      y_pred,
        }
        logger.info(f"  {name.upper()} → accuracy={acc:.4f} | f1={f1:.4f} | time={elapsed}s")

    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    logger.info(f"Best model: {best_name.upper()} (f1={results[best_name]['f1_weighted']})")
    return results, best_name


def run_kmeans_clustering(
    X, df: pd.DataFrame, n_clusters: int = 4, k_range: range = range(2, 11)
) -> Tuple[KMeans, pd.DataFrame, Dict, list]:
    """
    Fit K-Means, compute priority zones, compute elbow inertias.
    Returns (km_model, df_with_clusters, priority_map, inertias).
    """
    logger.info(f"Running K-Means (k={n_clusters})")

    # Elbow data
    inertias = []
    for k in k_range:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_tmp.fit(X)
        inertias.append(float(km_tmp.inertia_))
    logger.info(f"Elbow inertias computed for k={list(k_range)}")

    # Final model
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster_id"] = km.fit_predict(X)

    # Map cluster IDs → priority zones by disaster frequency per cluster
    cluster_disaster_counts = (
        df.groupby("cluster_id")["disaster_type"]
        .count()
        .sort_values(ascending=False)
    )
    priority_labels = ["High", "Medium-High", "Medium-Low", "Low"]
    priority_map: Dict[int, str] = {
        int(cid): label
        for cid, label in zip(cluster_disaster_counts.index, priority_labels)
    }
    df["cluster_priority"] = df["cluster_id"].map(priority_map)

    logger.info(f"Cluster distribution: {df['cluster_priority'].value_counts().to_dict()}")
    return km, df, priority_map, inertias


def save_all_artifacts(
    results: Dict,
    best_name: str,
    vectorizer: TfidfVectorizer,
    km: KMeans,
    priority_map: Dict,
    df: pd.DataFrame,
    inertias: list,
    k_range,
    out_dir: str = "models",
) -> None:
    """Persist all trained artifacts to disk."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save models
    for name, r in results.items():
        path = Path(out_dir) / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(r["model"], f)
        logger.info(f"Saved {path}")

    # Save vectorizer and km
    with open(Path(out_dir) / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(Path(out_dir) / "kmeans.pkl", "wb") as f:
        pickle.dump(km, f)

    # Save metadata
    meta = {
        "best_model": best_name,
        "accuracies": {k: v["accuracy"] for k, v in results.items()},
        "f1_scores":  {k: v["f1_weighted"] for k, v in results.items()},
        "train_times": {k: v["train_time"] for k, v in results.items()},
        "classes":    results[best_name]["classes"],
        "priority_map": {str(k): v for k, v in priority_map.items()},
        "elbow_inertias": inertias,
        "elbow_k_range":  list(k_range),
        "n_records":  len(df),
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    with open(Path(out_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save confusion matrices
    cm_data = {
        name: {
            "matrix":  r["confusion_matrix"],
            "classes": r["classes"],
        }
        for name, r in results.items()
    }
    with open(Path(out_dir) / "confusion_matrices.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    # Save processed dataframe
    df.to_csv(Path(out_dir) / "processed_data.csv", index=False)
    logger.info(f"All artifacts saved to {out_dir}/")


def load_artifacts(model_dir: str = "models") -> Dict[str, Any]:
    """Load all artifacts for inference."""
    base = Path(model_dir)
    with open(base / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(base / "kmeans.pkl", "rb") as f:
        km = pickle.load(f)
    with open(base / "metadata.json") as f:
        meta = json.load(f)

    models = {}
    for name in ["rf", "adaboost", "gb"]:
        with open(base / f"{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)

    df = pd.read_csv(base / "processed_data.csv")
    return {"vectorizer": vectorizer, "km": km, "meta": meta, "models": models, "df": df}
