"""Génération de métriques réseau SIMULÉES et reproductibles.

Modèle simplifié mais cohérent :
- chaque équipement a une contribution propre à la latence (temps de
  traitement), un taux de perte propre, une charge CPU et mémoire ;
- la latence observée par la sonde est la somme des contributions des
  équipements traversés depuis le cœur de réseau ; la perte observée combine
  les pertes de chaque saut : 1 - Π(1 - p_i) ;
- un équipement injoignable n'a ni latence, ni CPU, ni mémoire mesurés (NaN),
  une perte de 100 % et une disponibilité de 0 ;
- un faible pourcentage de mesures CPU/mémoire manque (timeouts SNMP simulés)
  et de courts pics CPU bénins apparaissent (sauvegardes, tâches planifiées) :
  ils servent à illustrer les faux positifs.

La colonne ``ground_truth_fault`` indique les mesures réellement affectées
par le défaut injecté. Elle n'existe que parce que les données sont simulées
et sert uniquement à évaluer les méthodes de détection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from .topology import build_default_topology, downstream_devices, monitored_devices, probe_path

METRIC_COLUMNS = ["latency_ms", "packet_loss_pct", "cpu_pct", "memory_pct", "available"]
DEFAULT_START_TIME = pd.Timestamp("2026-01-15 08:00:00")
BASE_PROBE_RTT_MS = 0.4  # aller-retour minimal sonde -> cœur

# Profil nominal par rôle : contribution propre à la latence (ms), CPU (%), mémoire (%).
ROLE_BASELINES = {
    "router": {"latency": 1.5, "cpu": 25.0, "memory": 40.0},
    "firewall": {"latency": 1.0, "cpu": 30.0, "memory": 45.0},
    "switch": {"latency": 0.3, "cpu": 15.0, "memory": 30.0},
    "server": {"latency": 0.8, "cpu": 35.0, "memory": 55.0},
}


@dataclass(frozen=True)
class FaultProfile:
    """Défaut injecté sur l'équipement cible pendant la fenêtre d'incident."""

    extra_latency_ms: float = 0.0  # latence ajoutée (propagée en aval)
    loss_pct: float = 0.0  # perte propre ajoutée (propagée en aval)
    cpu_target_pct: float | None = None  # niveau CPU atteint pendant l'incident
    memory_target_pct: float | None = None  # niveau mémoire atteint
    unavailable: bool = False  # équipement injoignable


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    description: str
    target_device: str | None = None
    start_ratio: float = 0.0  # début de l'incident (fraction de la durée)
    end_ratio: float = 0.0  # fin de l'incident (1.0 = toujours en cours)
    fault: FaultProfile = field(default_factory=FaultProfile)


SCENARIOS: dict[str, Scenario] = {
    "normal": Scenario(
        "normal", "Fonctionnement normal",
        "Aucun défaut injecté : sert de référence pour mesurer les faux positifs."),
    "latence_elevee": Scenario(
        "latence_elevee", "Latence élevée",
        "Le pare-feu ajoute 50 à 90 ms de latence (charge d'inspection accrue). "
        "Le routeur, joint à travers le pare-feu, subit la même latence.",
        "fw-01", 0.55, 0.80, FaultProfile(extra_latency_ms=70.0, cpu_target_pct=72.0)),
    "perte_paquets": Scenario(
        "perte_paquets", "Perte de paquets",
        "Le commutateur serveurs perd 4 à 10 % des paquets (défaut de port ou de "
        "câble simulé). Les serveurs situés derrière lui sont affectés.",
        "sw-srv-01", 0.50, 0.75, FaultProfile(loss_pct=7.0)),
    "routeur_indisponible": Scenario(
        "routeur_indisponible", "Routeur indisponible",
        "Le routeur de bordure ne répond plus : perte de l'accès WAN/Internet.",
        "rtr-edge-01", 0.60, 0.78, FaultProfile(unavailable=True)),
    "serveur_surcharge": Scenario(
        "serveur_surcharge", "Serveur surchargé",
        "Le serveur applicatif monte progressivement à ~95 % CPU et ~90 % mémoire ; "
        "son temps de réponse se dégrade. L'incident est toujours en cours en fin de période.",
        "srv-app-01", 0.65, 1.0,
        FaultProfile(extra_latency_ms=45.0, cpu_target_pct=95.0, memory_target_pct=90.0)),
}


def get_scenario(key: str) -> Scenario:
    if key not in SCENARIOS:
        raise ValueError(f"Scénario inconnu : '{key}'. Choix possibles : {', '.join(SCENARIOS)}")
    return SCENARIOS[key]


def _fault_intensity(n_samples: int, scenario: Scenario, ramp_samples: int = 4) -> np.ndarray:
    """Intensité du défaut (0..1) à chaque instant, avec une montée progressive."""
    intensity = np.zeros(n_samples)
    if scenario.target_device is None:
        return intensity
    start = int(round(scenario.start_ratio * n_samples))
    end = n_samples if scenario.end_ratio >= 1.0 else int(round(scenario.end_ratio * n_samples))
    if end <= start:
        return intensity
    steps = np.arange(end - start)
    if scenario.fault.unavailable:
        intensity[start:end] = 1.0  # une panne est franche
    else:
        intensity[start:end] = np.minimum(1.0, (steps + 1) / ramp_samples)
    return intensity


def generate_metrics(
    scenario_key: str = "normal",
    duration_minutes: int = 360,
    interval_seconds: int = 60,
    seed: int = 42,
    start_time: pd.Timestamp | None = None,
    graph: nx.Graph | None = None,
    missing_rate: float = 0.004,
    benign_spike_rate: float = 0.006,
) -> pd.DataFrame:
    """Génère un DataFrame « long » : une ligne par (horodatage, équipement)."""
    if duration_minutes <= 0 or interval_seconds <= 0:
        raise ValueError("La durée et l'intervalle doivent être strictement positifs.")
    n_samples = int(duration_minutes * 60 // interval_seconds)
    if n_samples < 10:
        raise ValueError("La période simulée doit contenir au moins 10 mesures par équipement.")

    scenario = get_scenario(scenario_key)
    graph = graph if graph is not None else build_default_topology()
    devices = monitored_devices(graph)
    if scenario.target_device is not None and scenario.target_device not in devices:
        raise ValueError(f"L'équipement cible {scenario.target_device} n'existe pas dans la topologie.")

    rng = np.random.default_rng(seed)
    start_time = start_time if start_time is not None else DEFAULT_START_TIME
    timestamps = pd.date_range(start_time, periods=n_samples, freq=pd.Timedelta(seconds=interval_seconds))
    elapsed_hours = np.arange(n_samples) * interval_seconds / 3600.0
    intensity = _fault_intensity(n_samples, scenario)
    fault = scenario.fault
    target = scenario.target_device

    # 1) Comportement propre de chaque équipement -----------------------------
    own_latency, own_loss, cpu, memory, reachable_self = {}, {}, {}, {}, {}
    for device_id in devices:
        base = ROLE_BASELINES[graph.nodes[device_id]["role"]]
        # profil de charge lent (cycle de ~4 h) commun à l'équipement
        phase = rng.uniform(0, 2 * np.pi)
        load_wave = np.sin(2 * np.pi * elapsed_hours / 4.0 + phase)

        latency = base["latency"] * (1 + 0.10 * load_wave) + np.abs(rng.normal(0, 0.12 * base["latency"], n_samples))
        loss = np.where(rng.random(n_samples) < 0.02, rng.uniform(0.05, 0.4, n_samples), 0.0)
        cpu_series = base["cpu"] + 6 * load_wave + rng.normal(0, 2.5, n_samples)
        mem_series = base["memory"] + 2 * load_wave + rng.normal(0, 1.2, n_samples)

        # pics CPU bénins et ponctuels (tâches planifiées)
        spikes = rng.random(n_samples) < benign_spike_rate
        cpu_series = cpu_series + spikes * rng.uniform(25, 50, n_samples)

        is_target = device_id == target
        if is_target:
            jitter = rng.uniform(0.7, 1.3, n_samples)
            latency = latency + intensity * fault.extra_latency_ms * jitter
            if fault.loss_pct:
                loss = loss + intensity * fault.loss_pct * rng.uniform(0.6, 1.4, n_samples)
            if fault.cpu_target_pct is not None:
                cpu_series = cpu_series + intensity * (fault.cpu_target_pct - cpu_series) + intensity * rng.normal(0, 1.5, n_samples)
            if fault.memory_target_pct is not None:
                mem_series = mem_series + intensity * (fault.memory_target_pct - mem_series) + intensity * rng.normal(0, 1.0, n_samples)

        own_latency[device_id] = latency
        own_loss[device_id] = np.clip(loss, 0, 100)
        cpu[device_id] = np.clip(cpu_series, 0, 100)
        memory[device_id] = np.clip(mem_series, 0, 100)
        reachable_self[device_id] = ~((intensity > 0) & fault.unavailable) if is_target else np.ones(n_samples, bool)

    # 2) Mesures observées depuis la sonde (propagation le long du chemin) ----
    affected_by_fault = set()
    if target is not None:
        affected_by_fault.add(target)
        if fault.extra_latency_ms or fault.loss_pct or fault.unavailable:
            affected_by_fault.update(downstream_devices(graph, target))

    frames = []
    for device_id in devices:
        path = probe_path(graph, device_id)
        reachable = np.logical_and.reduce([reachable_self[hop] for hop in path])
        observed_latency = BASE_PROBE_RTT_MS + np.sum([own_latency[hop] for hop in path], axis=0)
        delivery = np.prod([1 - own_loss[hop] / 100 for hop in path], axis=0)
        observed_loss = (1 - delivery) * 100

        node = graph.nodes[device_id]
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "device_id": device_id,
            "device_name": node["name"],
            "role": node["role"],
            "latency_ms": np.where(reachable, observed_latency, np.nan),
            "packet_loss_pct": np.where(reachable, observed_loss, 100.0),
            "cpu_pct": np.where(reachable, cpu[device_id], np.nan),
            "memory_pct": np.where(reachable, memory[device_id], np.nan),
            "available": reachable.astype(int),
        })
        # mesures CPU/mémoire manquantes (timeout de collecte simulé)
        missing = (rng.random(n_samples) < missing_rate) & reachable
        frame.loc[missing, ["cpu_pct", "memory_pct"]] = np.nan
        frame["ground_truth_fault"] = (intensity > 0) & (device_id in affected_by_fault)
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    metrics["scenario"] = scenario.key
    metrics["source"] = "simulation"
    for column in ["latency_ms", "packet_loss_pct", "cpu_pct", "memory_pct"]:
        metrics[column] = metrics[column].round(3)
    return metrics.sort_values(["timestamp", "device_id"], ignore_index=True)
