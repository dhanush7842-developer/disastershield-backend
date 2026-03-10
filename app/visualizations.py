"""
DisasterShield AI — Visualization Generator
Produces all 5 required charts as base64 PNG strings.
"""

import io
import base64
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────
PALETTE = {
    "navy":   "#1B3A6B",
    "blue":   "#2563EB",
    "teal":   "#0D9488",
    "red":    "#DC2626",
    "amber":  "#D97706",
    "green":  "#16A34A",
    "purple": "#7C3AED",
    "gray":   "#6B7280",
    "light":  "#F1F5F9",
}
PRIORITY_COLORS = {
    "High":        "#DC2626",
    "Medium-High": "#D97706",
    "Medium-Low":  "#F59E0B",
    "Low":         "#16A34A",
}
MODEL_COLORS = {
    "rf":       "#2563EB",
    "adaboost": "#D97706",
    "gb":       "#16A34A",
}
MODEL_LABELS = {
    "rf":       "Random Forest",
    "adaboost": "AdaBoost",
    "gb":       "Gradient Boosting",
}

sns.set_theme(style="whitegrid", palette="muted", font="DejaVu Sans")


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chart_confusion_matrix(results: dict, model_key: str) -> str:
    r = results[model_key]
    classes = r["classes"]
    cm = np.array(r["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(11, 9))
    mask_diag = np.zeros_like(cm, dtype=bool)
    np.fill_diagonal(mask_diag, True)

    # Background heatmap
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="#E5E7EB", cbar_kws={"shrink": 0.8})

    # Annotate cells manually with color coding
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            color = "white" if i == j else ("#1E293B" if cm[i, j] < cm.max() * 0.5 else "white")
            weight = "bold" if i == j else "normal"
            ax.text(j + 0.5, i + 0.5, str(val),
                    ha="center", va="center", fontsize=11,
                    color=color, fontweight=weight)

    acc = r["accuracy"]
    f1  = r["f1_weighted"]
    ax.set_title(
        f"Confusion Matrix — {MODEL_LABELS.get(model_key, model_key)}\n"
        f"Accuracy: {acc:.1%}  |  Weighted F1: {f1:.1%}",
        fontsize=14, fontweight="bold", color=PALETTE["navy"], pad=16
    )
    ax.set_ylabel("True Label", fontsize=12, color=PALETTE["gray"])
    ax.set_xlabel("Predicted Label", fontsize=12, color=PALETTE["gray"])
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_roc_curves(results: dict, model_key: str) -> str:
    r       = results[model_key]
    model   = r["model"]
    X_test  = r["X_test"]
    y_test  = r["y_test"]
    classes = r["classes"]

    y_bin   = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(11, 8))
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(classes)))

    macro_tpr, macro_fpr = [], []
    for i, (cls, col) in enumerate(zip(classes, colors)):
        if i >= y_bin.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2.0,
                label=f"{cls}  (AUC={roc_auc:.2f})", alpha=0.85)

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_title(
        f"ROC Curves per Class — {MODEL_LABELS.get(model_key, model_key)}",
        fontsize=14, fontweight="bold", color=PALETTE["navy"], pad=14
    )
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95,
              title="Disaster Class", title_fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_disaster_distribution(df) -> str:
    counts = df["disaster_type"].value_counts()
    top5   = counts.head(5)
    others = counts.iloc[5:].sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart — top 5
    bar_colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["amber"],
                  PALETTE["red"], PALETTE["purple"]]
    bars = axes[0].bar(top5.index, top5.values, color=bar_colors, edgecolor="white",
                       linewidth=1.5, zorder=3)
    for bar, val in zip(bars, top5.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.4, str(val),
                     ha="center", fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_title("Top 5 Disaster Types (1990–2021)",
                      fontsize=13, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_ylabel("Number of Events", fontsize=11)
    axes[0].tick_params(axis="x", rotation=25, labelsize=10)
    axes[0].set_facecolor("#F8FAFC")
    axes[0].grid(axis="y", alpha=0.4, zorder=0)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Pie chart — all classes
    all_labels  = list(counts.index)
    all_values  = list(counts.values)
    pie_colors  = plt.cm.Set2(np.linspace(0, 1, len(all_labels)))
    explode     = [0.04] * len(all_labels)
    wedges, texts, autotexts = axes[1].pie(
        all_values, labels=None, colors=pie_colors,
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        startangle=140, explode=explode,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        textprops={"fontsize": 9}
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(9)
    axes[1].legend(wedges, [f"{l} ({v})" for l, v in zip(all_labels, all_values)],
                   loc="lower left", fontsize=8, bbox_to_anchor=(-0.1, -0.15),
                   title="All Disaster Types", title_fontsize=9, ncol=2)
    axes[1].set_title("Full Distribution — All Classes",
                      fontsize=13, fontweight="bold", color=PALETTE["navy"])

    fig.suptitle(f"India Natural Disasters 1990–2021  |  {len(df)} Total Events",
                 fontsize=12, color=PALETTE["gray"], y=1.02)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_accuracy_comparison(meta: dict) -> str:
    names  = list(meta["accuracies"].keys())
    accs   = [meta["accuracies"][n] for n in names]
    f1s    = [meta["f1_scores"][n]  for n in names]
    labels = [MODEL_LABELS.get(n, n) for n in names]
    x      = np.arange(len(names))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy",
                   color=[MODEL_COLORS[n] for n in names], alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, f1s, width, label="Weighted F1",
                   color=[MODEL_COLORS[n] for n in names], alpha=0.45, zorder=3,
                   edgecolor="white", linewidth=1.5, hatch="///")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", fontsize=10, fontweight="bold",
                    color=PALETTE["navy"])

    # Best model annotation
    best_idx = names.index(meta["best_model"])
    ax.annotate("★ Best", xy=(best_idx - width/2, accs[best_idx]),
                xytext=(best_idx - width/2 + 0.35, accs[best_idx] + 0.04),
                fontsize=11, color=PALETTE["red"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=1.5))

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — Accuracy & F1",
                 fontsize=14, fontweight="bold", color=PALETTE["navy"], pad=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_facecolor("#F8FAFC")
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_elbow_plot(meta: dict) -> str:
    ks       = meta["elbow_k_range"]
    inertias = meta["elbow_inertias"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ks, inertias, "o-", color=PALETTE["blue"], lw=2.5,
            markersize=8, markerfacecolor=PALETTE["navy"], zorder=3)

    # Fill under curve
    ax.fill_between(ks, inertias, alpha=0.08, color=PALETTE["blue"])

    # Highlight selected K=4
    selected_k = 4
    if selected_k in ks:
        idx = ks.index(selected_k)
        ax.axvline(x=selected_k, color=PALETTE["red"], linestyle="--", lw=2, alpha=0.8)
        ax.plot(selected_k, inertias[idx], "D", color=PALETTE["red"],
                markersize=12, zorder=5, label=f"K={selected_k} Selected")
        ax.annotate(f"K={selected_k}\n(Selected)",
                    xy=(selected_k, inertias[idx]),
                    xytext=(selected_k + 0.5, inertias[idx] + (max(inertias) - min(inertias)) * 0.1),
                    fontsize=10, color=PALETTE["red"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=PALETTE["red"]))

    # Rate-of-change annotation
    ax.annotate("Diminishing returns\nbeyond K=4",
                xy=(5, inertias[ks.index(5)] if 5 in ks else inertias[3]),
                xytext=(6.5, inertias[-1] + (max(inertias) - min(inertias)) * 0.3),
                fontsize=9, color=PALETTE["gray"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["gray"], lw=1.2))

    ax.set_title("K-Means Elbow Plot — Optimal Cluster Selection",
                 fontsize=14, fontweight="bold", color=PALETTE["navy"], pad=14)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    ax.set_xticks(ks)
    ax.legend(fontsize=10)
    ax.set_facecolor("#F8FAFC")
    ax.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_b64(fig)


def generate_all_charts(results: dict, meta: dict, df, model_key: str) -> dict:
    """Generate all 5 charts and return as base64 dict."""
    logger.info(f"Generating charts for model: {model_key}")
    return {
        "confusion_matrix":      chart_confusion_matrix(results, model_key),
        "roc_curves":            chart_roc_curves(results, model_key),
        "disaster_distribution": chart_disaster_distribution(df),
        "accuracy_comparison":   chart_accuracy_comparison(meta),
        "elbow_plot":            chart_elbow_plot(meta),
    }


def save_charts_to_disk(charts: dict, out_dir: str = "charts") -> None:
    """Save base64 charts as PNG files."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, b64 in charts.items():
        data = base64.b64decode(b64.split(",")[1])
        path = Path(out_dir) / f"{name}.png"
        with open(path, "wb") as f:
            f.write(data)
        logger.info(f"Saved chart: {path}")
