"""Sources de métriques et validation du format d'échange.

Toute source (simulation, fichier CSV, et dans une version future SNMP,
syslog ou API d'un outil de supervision) doit produire un DataFrame au
format décrit par ``REQUIRED_COLUMNS``. Le reste de l'application
(détection, diagnostic, KPI) ne dépend que de ce format : brancher une
source réelle consiste à écrire une nouvelle classe ``MetricsSource``.

Aucune intégration réseau réelle n'est fournie dans cette version.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO

import pandas as pd

from .simulation import generate_metrics

REQUIRED_COLUMNS = [
    "timestamp", "device_id", "latency_ms", "packet_loss_pct",
    "cpu_pct", "memory_pct", "available",
]
NUMERIC_COLUMNS = ["latency_ms", "packet_loss_pct", "cpu_pct", "memory_pct", "available"]
VALUE_RANGES = {"packet_loss_pct": (0, 100), "cpu_pct": (0, 100), "memory_pct": (0, 100)}


class MetricsSource(ABC):
    """Interface commune à toutes les sources de métriques."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Retourne les métriques au format standard (voir REQUIRED_COLUMNS)."""


class SimulatedSource(MetricsSource):
    def __init__(self, scenario_key: str, duration_minutes: int = 360,
                 interval_seconds: int = 60, seed: int = 42) -> None:
        self.scenario_key = scenario_key
        self.duration_minutes = duration_minutes
        self.interval_seconds = interval_seconds
        self.seed = seed

    def load(self) -> pd.DataFrame:
        return generate_metrics(self.scenario_key, self.duration_minutes,
                                self.interval_seconds, self.seed)


class CsvMetricsSource(MetricsSource):
    """Charge des métriques depuis un CSV au format NetOps Copilot.

    Permet de rejouer un export précédent ou des mesures issues d'un autre
    outil après conversion au format attendu.
    """

    def __init__(self, path_or_buffer: str | Path | IO, known_devices: list[str] | None = None) -> None:
        self.path_or_buffer = path_or_buffer
        self.known_devices = known_devices

    def load(self) -> pd.DataFrame:
        try:
            raw = pd.read_csv(self.path_or_buffer)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise ValueError(f"Fichier CSV illisible : {error}") from error
        # colonnes calculées d'une analyse précédente (export) : recalculées ici
        derived = [c for c in raw.columns if c.startswith(("rule_", "ml_"))]
        metrics = validate_metrics(raw.drop(columns=derived), self.known_devices)
        if "source" not in metrics.columns:
            metrics["source"] = "csv"
        return metrics


def validate_metrics(frame: pd.DataFrame, known_devices: list[str] | None = None) -> pd.DataFrame:
    """Vérifie et normalise un DataFrame de métriques.

    - colonnes obligatoires présentes ;
    - horodatages et valeurs numériques convertibles (sinon NaN, puis contrôle) ;
    - valeurs hors bornes physiques rejetées ;
    - équipements inconnus de la topologie rejetés (si la liste est fournie).
    Lève ``ValueError`` avec un message explicite en cas de problème.
    """
    if frame is None or frame.empty:
        raise ValueError("Aucune métrique à analyser (jeu de données vide).")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes : {', '.join(missing)}")

    metrics = frame.copy()
    metrics["timestamp"] = pd.to_datetime(metrics["timestamp"], errors="coerce")
    if metrics["timestamp"].isna().any():
        bad = int(metrics["timestamp"].isna().sum())
        raise ValueError(f"{bad} horodatage(s) invalide(s) dans les métriques.")
    metrics["device_id"] = metrics["device_id"].astype(str)
    for column in NUMERIC_COLUMNS:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    if metrics["available"].isna().any() or not metrics["available"].isin([0, 1]).all():
        raise ValueError("La colonne 'available' doit contenir uniquement 0 ou 1.")
    metrics["available"] = metrics["available"].astype(int)

    for column, (low, high) in VALUE_RANGES.items():
        values = metrics[column].dropna()
        if ((values < low) | (values > high)).any():
            raise ValueError(f"Valeurs hors de l'intervalle [{low}, {high}] dans '{column}'.")
    if (metrics["latency_ms"].dropna() < 0).any():
        raise ValueError("La latence ne peut pas être négative.")

    if known_devices is not None:
        unknown = sorted(set(metrics["device_id"]) - set(known_devices))
        if unknown:
            raise ValueError(f"Équipements absents de la topologie : {', '.join(unknown)}")

    return metrics.sort_values(["timestamp", "device_id"], ignore_index=True)
