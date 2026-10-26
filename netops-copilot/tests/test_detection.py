"""Tests des règles, de l'Isolation Forest et de la comparaison des méthodes."""

import numpy as np
import pandas as pd
import pytest

from netops_copilot.detection import (
    IsolationForestDetector, RuleThresholds, compare_methods, detect_with_rules,
)
from netops_copilot.simulation import generate_metrics


def _rows(**overrides):
    base = {"timestamp": pd.Timestamp("2026-01-01 10:00"), "device_id": "srv-app-01", "role": "server",
            "latency_ms": 2.0, "packet_loss_pct": 0.0, "cpu_pct": 40.0, "memory_pct": 50.0, "available": 1}
    base.update(overrides)
    return pd.DataFrame([base])


def test_rules_normal_measure_is_not_flagged():
    result = detect_with_rules(_rows())
    assert not result["rule_anomaly"].iloc[0]
    assert result["rule_severity"].iloc[0] == "normal"
    assert result["rule_reasons"].iloc[0] == ""


@pytest.mark.parametrize("field,value,severity", [
    ("latency_ms", 25.0, "avertissement"),
    ("latency_ms", 60.0, "critique"),
    ("packet_loss_pct", 2.0, "avertissement"),
    ("packet_loss_pct", 5.0, "critique"),  # seuil inclus
    ("cpu_pct", 85.0, "avertissement"),
    ("memory_pct", 96.0, "critique"),
])
def test_rules_thresholds(field, value, severity):
    result = detect_with_rules(_rows(**{field: value}))
    assert result["rule_anomaly"].iloc[0]
    assert result["rule_severity"].iloc[0] == severity
    assert field in result["rule_metrics"].iloc[0]
    assert "seuil" in result["rule_reasons"].iloc[0]


def test_rules_unavailable_device_is_critical_even_with_missing_values():
    frame = _rows(available=0, latency_ms=np.nan, cpu_pct=np.nan, memory_pct=np.nan, packet_loss_pct=100.0)
    result = detect_with_rules(frame)
    assert result["rule_severity"].iloc[0] == "critique"
    assert result["rule_metrics"].iloc[0] == "available"
    assert "injoignable" in result["rule_reasons"].iloc[0]


def test_rules_missing_cpu_does_not_create_alert():
    result = detect_with_rules(_rows(cpu_pct=np.nan, memory_pct=np.nan))
    assert not result["rule_anomaly"].iloc[0]


def test_custom_thresholds_are_applied():
    result = detect_with_rules(_rows(cpu_pct=60.0), RuleThresholds(cpu_warning_pct=50, cpu_critical_pct=70))
    assert result["rule_severity"].iloc[0] == "avertissement"


def _trained_detector(seed=0):
    history = generate_metrics("normal", 720, 60, seed=1000 + seed)
    return IsolationForestDetector(expected_false_positive_rate=0.01, random_state=seed).fit(history)


def test_isolation_forest_false_positive_rate_is_bounded_on_normal_data():
    detector = _trained_detector()
    scored = detector.score(generate_metrics("normal", 360, 60, seed=5))
    rate = scored.loc[scored["ml_evaluated"], "ml_anomaly"].mean()
    assert rate < 0.03  # cible 1 %, tolérance pour la variabilité d'échantillonnage


def test_isolation_forest_detects_server_overload_with_explanation():
    detector = _trained_detector()
    scored = detector.score(generate_metrics("serveur_surcharge", 360, 60, seed=42))
    faulty = scored[scored["ground_truth_fault"]]
    assert faulty["ml_anomaly"].mean() > 0.9
    explanation = faulty.loc[faulty["ml_anomaly"], "ml_explanation"].iloc[10]
    assert "CPU" in explanation or "mémoire" in explanation


def test_isolation_forest_skips_unreachable_and_incomplete_rows():
    detector = _trained_detector()
    scored = detector.score(generate_metrics("routeur_indisponible", 360, 60, seed=42))
    down = scored[scored["available"] == 0]
    assert not down["ml_evaluated"].any() and not down["ml_anomaly"].any()
    assert down["ml_score"].isna().all()
    assert "non évalué" in down["ml_explanation"].iloc[0]


def test_isolation_forest_requires_training():
    with pytest.raises(RuntimeError):
        IsolationForestDetector().score(_rows())
    with pytest.raises(ValueError):
        IsolationForestDetector().fit(generate_metrics("normal", 120, 60).head(20))


def test_compare_methods_counts_are_consistent():
    detector = _trained_detector()
    detections = detector.score(detect_with_rules(generate_metrics("perte_paquets", 360, 60, seed=42)))
    comparison = compare_methods(detections)
    agreement = comparison["accord"]
    assert (agreement["regles_et_ml"] + agreement["regles_seules"] + agreement["ml_seul"] + agreement["aucune"]
            == agreement["lignes_evaluees"])
    evaluation = comparison["evaluation"].set_index("methode")
    assert evaluation.loc["Règles", "rappel"] > 0.95
    assert evaluation.loc["Règles OU ML", "rappel"] >= evaluation.loc["Isolation Forest", "rappel"]
