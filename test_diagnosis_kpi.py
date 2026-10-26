"""Tests des incidents, du diagnostic, des KPI, des sources de données et du pipeline."""

import io

import numpy as np
import pandas as pd
import pytest

from netops_copilot.data_sources import CsvMetricsSource, validate_metrics
from netops_copilot.detection import detect_with_rules
from netops_copilot.diagnosis import build_incidents, incidents_for_export
from netops_copilot.kpi import STATUS_DEGRADED, STATUS_DOWN, STATUS_OK, compute_kpis, device_status
from netops_copilot.pipeline import AnalysisConfig, run_analysis
from netops_copilot.topology import build_default_topology, monitored_devices


@pytest.fixture(scope="module")
def results():
    keys = ["normal", "latence_elevee", "perte_paquets", "routeur_indisponible", "serveur_surcharge"]
    return {key: run_analysis(AnalysisConfig(scenario_key=key)) for key in keys}


def test_normal_scenario_has_no_major_incident(results):
    incidents = results["normal"].incidents
    assert incidents.empty or set(incidents["severity"]) <= {"mineure"}


@pytest.mark.parametrize("scenario,device,category", [
    ("latence_elevee", "fw-01", "latence"),
    ("perte_paquets", "sw-srv-01", "perte_paquets"),
    ("routeur_indisponible", "rtr-edge-01", "indisponibilite"),
    ("serveur_surcharge", "srv-app-01", "surcharge"),
])
def test_each_fault_scenario_produces_the_expected_root_incident(results, scenario, device, category):
    incidents = results[scenario].incidents
    root = incidents[(incidents["device_id"] == device) & (incidents["probable_root_cause"] == "")]
    assert len(root) == 1
    incident = root.iloc[0]
    assert incident["category"] == category
    assert incident["severity"] == "critique"
    assert incident["priority"] == "P1"
    assert incident["evidence"] and incident["hypotheses"] and incident["recommendations"]


def test_downstream_incidents_are_correlated_to_upstream_root_cause(results):
    incidents = results["perte_paquets"].incidents
    symptoms = incidents[incidents["probable_root_cause"] == "sw-srv-01"]
    assert set(symptoms["device_id"]) == {"srv-app-01", "srv-db-01", "srv-dns-01"}
    assert (symptoms["priority"] != "P1").all()  # un symptôme est moins prioritaire que la cause
    latency = results["latence_elevee"].incidents
    assert latency.loc[latency["device_id"] == "rtr-edge-01", "probable_root_cause"].iloc[0] == "fw-01"


def test_incident_status_reflects_whether_fault_is_ongoing(results):
    assert results["serveur_surcharge"].incidents["status"].iloc[0] == "Ouvert"
    assert results["routeur_indisponible"].incidents["status"].iloc[0] == "Rétabli – à confirmer"


def test_router_incident_mentions_internet_impact(results):
    incident = results["routeur_indisponible"].incidents.iloc[0]
    assert "Internet" in incident["impact"]
    assert any("hypoth" in h.lower() or "panne" in h.lower() for h in incident["hypotheses"])


def test_isolated_spikes_do_not_open_incidents():
    graph = build_default_topology()
    times = pd.date_range("2026-01-01", periods=30, freq="min")
    frame = pd.DataFrame({"timestamp": times, "device_id": "srv-db-01", "device_name": "db", "role": "server",
                          "latency_ms": 2.0, "packet_loss_pct": 0.0, "cpu_pct": 40.0, "memory_pct": 50.0,
                          "available": 1})
    frame.loc[10, "cpu_pct"] = 95.0  # pic unique
    detections = detect_with_rules(frame)
    detections["ml_anomaly"] = False
    detections["ml_top_feature"] = ""
    detections["ml_explanation"] = ""
    assert build_incidents(detections, graph).empty
    frame.loc[11:14, "cpu_pct"] = 95.0  # anomalie persistante
    detections = detect_with_rules(frame)
    detections["ml_anomaly"] = False
    detections["ml_top_feature"] = ""
    detections["ml_explanation"] = ""
    incidents = build_incidents(detections, graph)
    assert len(incidents) == 1 and incidents.iloc[0]["category"] == "surcharge"


def test_device_status_and_replay(results):
    result = results["routeur_indisponible"]
    final = result.statuses.set_index("device_id")["status"]
    assert final["rtr-edge-01"] == STATUS_OK  # rétabli en fin de période
    during = result.incidents.iloc[0]["start"] + pd.Timedelta(minutes=10)
    replay = device_status(result.detections, at=during).set_index("device_id")["status"]
    assert replay["rtr-edge-01"] == STATUS_DOWN
    overload = results["serveur_surcharge"].statuses.set_index("device_id")["status"]
    assert overload["srv-app-01"] == STATUS_DEGRADED


def test_kpis_are_computed_correctly_on_known_data():
    times = pd.date_range("2026-01-01", periods=4, freq="min")
    frame = pd.DataFrame({"timestamp": list(times) * 2, "device_id": ["a"] * 4 + ["b"] * 4, "role": "server",
                          "latency_ms": [1, 2, 3, 4, 10, np.nan, 10, 10],
                          "packet_loss_pct": [0, 0, 0, 0, 0, 100, 0, 0],
                          "cpu_pct": [10, 20, 30, 40, 50, np.nan, 50, 50],
                          "memory_pct": [50] * 8, "available": [1, 1, 1, 1, 1, 0, 1, 1]})
    detections = detect_with_rules(frame)
    detections["ml_anomaly"] = False
    statuses = device_status(detections)
    empty_incidents = pd.DataFrame(columns=["severity", "status", "duration_min"])
    kpis = compute_kpis(detections, empty_incidents, statuses)
    assert kpis["availability_pct"] == 87.5
    assert kpis["latency_mean_ms"] == pytest.approx((1 + 2 + 3 + 4 + 10 + 10 + 10) / 7, abs=0.01)
    assert kpis["packet_loss_mean_pct"] == 0  # perte calculée sur les mesures joignables
    assert kpis["devices_total"] == 2 and kpis["devices_ok"] == 1 and kpis["devices_degraded"] == 1
    assert kpis["incidents_total"] == 0
    with pytest.raises(ValueError):
        compute_kpis(detections.iloc[0:0], empty_incidents, statuses)


def test_pipeline_kpis_match_scenarios(results):
    assert results["normal"].kpis["availability_pct"] == 100
    assert results["routeur_indisponible"].kpis["availability_pct"] < 100
    assert results["serveur_surcharge"].kpis["incidents_open"] >= 1


def test_validate_metrics_rejects_bad_inputs():
    good = pd.DataFrame({"timestamp": ["2026-01-01 10:00"], "device_id": ["fw-01"], "latency_ms": [1.0],
                         "packet_loss_pct": [0.0], "cpu_pct": [10.0], "memory_pct": [20.0], "available": [1]})
    assert validate_metrics(good)["timestamp"].dtype.kind == "M"
    with pytest.raises(ValueError, match="manquantes"):
        validate_metrics(good.drop(columns=["cpu_pct"]))
    with pytest.raises(ValueError, match="intervalle"):
        validate_metrics(good.assign(cpu_pct=150))
    with pytest.raises(ValueError, match="horodatage"):
        validate_metrics(good.assign(timestamp="pas une date"))
    with pytest.raises(ValueError, match="available"):
        validate_metrics(good.assign(available=3))
    with pytest.raises(ValueError, match="topologie"):
        validate_metrics(good.assign(device_id="inconnu"), known_devices=["fw-01"])
    with pytest.raises(ValueError, match="vide"):
        validate_metrics(good.iloc[0:0])


def test_csv_round_trip_through_pipeline(results):
    exported = results["serveur_surcharge"].detections.to_csv(index=False)
    source = CsvMetricsSource(io.StringIO(exported), known_devices=monitored_devices(build_default_topology()))
    replayed = run_analysis(AnalysisConfig(), source)
    original = results["serveur_surcharge"].incidents
    assert list(replayed.incidents["device_id"]) == list(original["device_id"])
    assert (replayed.detections["source"] == "simulation").all()  # l'origine simulée est conservée
    with pytest.raises(ValueError):
        CsvMetricsSource(io.StringIO("")).load()


def test_incident_export_flattens_lists(results):
    flat = incidents_for_export(results["perte_paquets"].incidents)
    assert isinstance(flat["recommendations"].iloc[0], str)
    assert " | " in flat["recommendations"].iloc[0]
