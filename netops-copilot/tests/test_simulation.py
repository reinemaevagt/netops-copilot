"""Tests de la topologie et de la génération des métriques simulées."""

import numpy as np
import pandas as pd
import pytest

from netops_copilot.simulation import SCENARIOS, generate_metrics
from netops_copilot.topology import (
    build_default_topology, downstream_devices, is_on_internet_path, monitored_devices, probe_path,
)


def test_topology_contains_required_device_types():
    graph = build_default_topology()
    roles = {graph.nodes[d]["role"] for d in monitored_devices(graph)}
    assert {"router", "firewall", "switch", "server"} <= roles
    assert len(monitored_devices(graph)) == 10


def test_downstream_devices_follow_probe_paths():
    graph = build_default_topology()
    assert set(downstream_devices(graph, "sw-srv-01")) == {"srv-app-01", "srv-db-01", "srv-dns-01"}
    assert downstream_devices(graph, "fw-01") == ["rtr-edge-01"]
    assert downstream_devices(graph, "srv-web-01") == []
    assert probe_path(graph, "srv-web-01") == ["sw-core-01", "sw-dmz-01", "srv-web-01"]
    assert is_on_internet_path(graph, "rtr-edge-01") and not is_on_internet_path(graph, "srv-db-01")


def test_metrics_shape_and_columns():
    metrics = generate_metrics("normal", duration_minutes=120, interval_seconds=60, seed=1)
    assert len(metrics) == 120 * 10
    for column in ["timestamp", "device_id", "latency_ms", "packet_loss_pct", "cpu_pct",
                   "memory_pct", "available", "ground_truth_fault", "source"]:
        assert column in metrics.columns
    assert (metrics["source"] == "simulation").all()
    assert metrics["cpu_pct"].dropna().between(0, 100).all()
    assert metrics["packet_loss_pct"].between(0, 100).all()


def test_generation_is_reproducible_with_seed():
    first = generate_metrics("perte_paquets", 180, 60, seed=7)
    second = generate_metrics("perte_paquets", 180, 60, seed=7)
    other = generate_metrics("perte_paquets", 180, 60, seed=8)
    pd.testing.assert_frame_equal(first, second)
    assert not first["latency_ms"].equals(other["latency_ms"])


def test_normal_scenario_has_no_injected_fault():
    metrics = generate_metrics("normal", 360, 60, seed=42)
    assert not metrics["ground_truth_fault"].any()
    assert (metrics["available"] == 1).all()


def test_router_down_scenario_makes_router_unreachable():
    metrics = generate_metrics("routeur_indisponible", 360, 60, seed=42)
    router = metrics[metrics["device_id"] == "rtr-edge-01"]
    down = router[router["available"] == 0]
    assert len(down) > 30
    assert down["latency_ms"].isna().all() and (down["packet_loss_pct"] == 100).all()
    # les autres équipements restent joignables
    assert (metrics.loc[metrics["device_id"] != "rtr-edge-01", "available"] == 1).all()


def test_packet_loss_propagates_to_downstream_servers():
    metrics = generate_metrics("perte_paquets", 360, 60, seed=42)
    fault_rows = metrics[metrics["ground_truth_fault"]]
    assert set(fault_rows["device_id"]) == {"sw-srv-01", "srv-app-01", "srv-db-01", "srv-dns-01"}
    assert fault_rows["packet_loss_pct"].median() > 4
    web = metrics[metrics["device_id"] == "srv-web-01"]
    assert web["packet_loss_pct"].max() < 2


def test_server_overload_raises_cpu_and_memory():
    metrics = generate_metrics("serveur_surcharge", 360, 60, seed=42)
    app = metrics[metrics["device_id"] == "srv-app-01"]
    during = app[app["ground_truth_fault"]]
    before = app[~app["ground_truth_fault"]]
    assert during["cpu_pct"].median() > 90 and during["memory_pct"].median() > 85
    assert before["cpu_pct"].median() < 50
    assert during["timestamp"].max() == metrics["timestamp"].max()  # toujours en cours


@pytest.mark.parametrize("scenario_key", list(SCENARIOS))
def test_every_scenario_generates_valid_data(scenario_key):
    metrics = generate_metrics(scenario_key, 240, 60, seed=3)
    assert len(metrics) == 240 * 10
    assert set(np.unique(metrics["available"])) <= {0, 1}
    if scenario_key != "normal":
        assert metrics["ground_truth_fault"].any()


def test_invalid_parameters_raise_clear_errors():
    with pytest.raises(ValueError):
        generate_metrics("inexistant")
    with pytest.raises(ValueError):
        generate_metrics("normal", duration_minutes=0)
    with pytest.raises(ValueError):
        generate_metrics("normal", duration_minutes=5, interval_seconds=60)
