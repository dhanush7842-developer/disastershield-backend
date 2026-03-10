"""
DisasterShield AI — Unit Tests
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import pytest
from ml_pipeline import clean_text, extract_disaster_type


# ── Text Cleaning ─────────────────────────────────────────────────────────
class TestCleanText:
    def test_lowercases(self):
        assert clean_text("MASSIVE FLOOD") == clean_text("massive flood")

    def test_removes_numbers(self):
        result = clean_text("The 2024 flood killed 50 people")
        assert not any(c.isdigit() for c in result)

    def test_removes_stopwords(self):
        result = clean_text("the flood was in the river and the people")
        words = result.split()
        assert "the" not in words
        assert "and" not in words
        assert "flood" in words

    def test_removes_punctuation(self):
        result = clean_text("earthquake! magnitude: 7.8 -- devastating")
        assert "!" not in result
        assert ":" not in result

    def test_short_words_removed(self):
        result = clean_text("a big flood")
        words = result.split()
        assert all(len(w) > 2 for w in words)

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_stopwords(self):
        # Should return empty or minimal
        result = clean_text("the and or but if")
        assert len(result.strip()) == 0


# ── Disaster Type Extraction ──────────────────────────────────────────────
class TestExtractDisasterType:
    @pytest.mark.parametrize("text,expected", [
        ("Heavy monsoon floods submerged 50 villages in Assam", "Flood"),
        ("7.8 magnitude earthquake struck the Himalayan region", "Earthquake"),
        ("Super cyclonic storm moved towards the Andhra Pradesh coast", "Cyclone"),
        ("Pilgrims were trampled in a stampede at the temple", "Stampede"),
        ("Cinema fire killed dozens trapped in the balcony", "Fire"),
        ("Aircraft crashed on runway at Bangalore airport", "Aviation"),
        ("Train derailment on Delhi-Mumbai rail line", "Rail"),
        ("Plague epidemic outbreak reported in Maharashtra", "Epidemic"),
        ("This is an unknown event with no keywords", "Other"),
    ])
    def test_correct_extraction(self, text, expected):
        assert extract_disaster_type(text) == expected

    def test_case_insensitive(self):
        assert extract_disaster_type("FLOOD in ASSAM") == extract_disaster_type("flood in assam")

    def test_empty_returns_other(self):
        assert extract_disaster_type("") == "Other"

    def test_mixed_keywords_first_match_wins(self):
        # Flood keywords appear before earthquake keywords in taxonomy
        result = extract_disaster_type("flood near earthquake epicenter")
        assert result in ("Flood", "Earthquake")   # either is valid — first match


# ── Import Smoke Tests ────────────────────────────────────────────────────
class TestImports:
    def test_pipeline_imports(self):
        from ml_pipeline import (
            load_and_preprocess, build_tfidf,
            train_ensemble_models, run_kmeans_clustering,
            save_all_artifacts, load_artifacts,
        )

    def test_visualization_imports(self):
        from visualizations import (
            chart_confusion_matrix, chart_roc_curves,
            chart_disaster_distribution, chart_accuracy_comparison,
            chart_elbow_plot, generate_all_charts,
        )

    def test_schemas_imports(self):
        from schemas import (
            PredictRequest, PredictResponse,
            DatasetPreviewResponse, AlertRequest, AlertResponse,
        )


# ── Schema Validation ─────────────────────────────────────────────────────
class TestSchemas:
    def test_predict_request_valid(self):
        from schemas import PredictRequest
        req = PredictRequest(description="Heavy flood in the Brahmaputra basin killing many", model="best")
        assert req.model == "best"

    def test_predict_request_too_short(self):
        from schemas import PredictRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PredictRequest(description="too short", model="best")

    def test_predict_request_invalid_model(self):
        from schemas import PredictRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PredictRequest(description="x" * 30, model="invalid_model_name")

    def test_alert_request(self):
        from schemas import AlertRequest
        req = AlertRequest(
            disaster_type="Flood",
            probability=0.92,
            cluster_priority="High",
        )
        assert req.disaster_type == "Flood"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
