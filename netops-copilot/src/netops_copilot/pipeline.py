"""Chaîne d'analyse complète : métriques -> détection -> incidents -> KPI.

Point d'entrée unique utilisé par l'interface Streamlit et par les tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from .data_sources import MetricsSource, SimulatedSource, validate_metrics
from .detection import IsolationForestDetector, RuleThresholds, compare_methods, detect_with_rules
from .diagnosis import IncidentSettings, build_incidents
from .kpi import compute_kpis, device_status
from .simulation import generate_metrics
from .topology import build_default_topology, monitored_devices

HISTORY_SEED_OFFSET = 10_000  # l'historique d'apprentissage utilise une graine distincte


@dataclass(frozen=True)
class AnalysisConfig:
    scenario_key: str = "normal"
    duration_minutes: int = 360
    interval_seconds: int = 60
    seed: int = 42
    history_minutes: int = 720
    expected_false_positive_rate: float = 0.01
    thresholds: RuleThresholds = field(default_factory=RuleThresholds)
    incident_settings: IncidentSettings = field(default_factory=IncidentSettings)


@dataclass
class AnalysisResult:
    config: AnalysisConfig
    graph: nx.Graph
    detections: pd.DataFrame
    incidents: pd.DataFrame
    statuses: pd.DataFrame
    kpis: dict
    comparison: dict
    detector: IsolationForestDetector


def train_detector(config: AnalysisConfig, graph: nx.Graph) -> IsolationForestDetector:
    """Entraîne le modèle sur un historique simulé SANS défaut (période de référence)."""
    history = generate_metrics("normal", config.history_minutes, config.interval_seconds,
                               seed=config.seed + HISTORY_SEED_OFFSET, graph=graph,
                               start_time=pd.Timestamp("2026-01-14 08:00:00"))
    detector = IsolationForestDetector(expected_false_positive_rate=config.expected_false_positive_rate,
                                       random_state=config.seed)
    return detector.fit(history)


def run_analysis(config: AnalysisConfig, source: MetricsSource | None = None) -> AnalysisResult:
    graph = build_default_topology()
    source = source or SimulatedSource(config.scenario_key, config.duration_minutes,
                                       config.interval_seconds, config.seed)
    metrics = validate_metrics(source.load(), known_devices=monitored_devices(graph))

    # compléter nom/rôle des équipements à partir de la topologie si absents (CSV externe)
    for column, attribute in (("device_name", "name"), ("role", "role")):
        if column not in metrics.columns:
            metrics[column] = metrics["device_id"].map(lambda d: graph.nodes[d][attribute])

    detector = train_detector(config, graph)
    detections = detect_with_rules(metrics, config.thresholds)
    detections = detector.score(detections)
    incidents = build_incidents(detections, graph, config.incident_settings)
    statuses = device_status(detections)
    kpis = compute_kpis(detections, incidents, statuses)
    comparison = compare_methods(detections)
    return AnalysisResult(config, graph, detections, incidents, statuses, kpis, comparison, detector)
