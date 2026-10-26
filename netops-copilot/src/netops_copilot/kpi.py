"""Indicateurs clés (KPI) et statut courant des équipements."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATUS_OK = "Opérationnel"
STATUS_DEGRADED = "Dégradé"
STATUS_DOWN = "Indisponible"


def _safe(value: float) -> float:
    """Convertit NaN/None en NaN float, arrondi à 2 décimales sinon."""
    return float("nan") if value is None or pd.isna(value) else round(float(value), 2)


def device_status(detections: pd.DataFrame, window: int = 5, at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Statut de chaque équipement sur les ``window`` dernières mesures avant ``at``.

    - Indisponible : la dernière mesure indique un équipement injoignable ;
    - Dégradé : au moins une alerte (règle) dans la fenêtre récente ;
    - Opérationnel : sinon.
    ``at`` (optionnel) permet de « rejouer » l'état du réseau à un instant passé.
    """
    rows = []
    data = detections if at is None else detections[detections["timestamp"] <= at]
    if data.empty:
        raise ValueError("Aucune mesure disponible avant l'instant demandé.")
    for device_id, group in data.sort_values("timestamp").groupby("device_id", sort=False):
        recent = group.tail(window)
        last = recent.iloc[-1]
        if last["available"] == 0:
            status = STATUS_DOWN
        elif recent["rule_anomaly"].any():
            status = STATUS_DEGRADED
        else:
            status = STATUS_OK
        rows.append({
            "device_id": device_id,
            "device_name": last.get("device_name", device_id),
            "role": last.get("role", ""),
            "status": status,
            "last_latency_ms": _safe(last["latency_ms"]),
            "last_packet_loss_pct": _safe(last["packet_loss_pct"]),
            "last_cpu_pct": _safe(last["cpu_pct"]),
            "last_memory_pct": _safe(last["memory_pct"]),
            "availability_pct": _safe(group["available"].mean() * 100),
            "last_seen": last["timestamp"],
        })
    return pd.DataFrame(rows)


def compute_kpis(detections: pd.DataFrame, incidents: pd.DataFrame, statuses: pd.DataFrame) -> dict:
    """KPI globaux de la période analysée."""
    if detections.empty:
        raise ValueError("Impossible de calculer des KPI sur un jeu de données vide.")
    reachable = detections[detections["available"] == 1]
    status_counts = statuses["status"].value_counts() if not statuses.empty else pd.Series(dtype=int)
    open_mask = incidents["status"].isin(["Ouvert", "En cours d'analyse"]) if not incidents.empty else pd.Series(dtype=bool)
    return {
        "availability_pct": _safe(detections["available"].mean() * 100),
        "latency_mean_ms": _safe(reachable["latency_ms"].mean()),
        "latency_p95_ms": _safe(reachable["latency_ms"].quantile(0.95)) if reachable["latency_ms"].notna().any() else float("nan"),
        "packet_loss_mean_pct": _safe(reachable["packet_loss_pct"].mean()),
        "cpu_mean_pct": _safe(reachable["cpu_pct"].mean()),
        "memory_mean_pct": _safe(reachable["memory_pct"].mean()),
        "devices_total": int(detections["device_id"].nunique()),
        "devices_ok": int(status_counts.get(STATUS_OK, 0)),
        "devices_degraded": int(status_counts.get(STATUS_DEGRADED, 0)),
        "devices_down": int(status_counts.get(STATUS_DOWN, 0)),
        "incidents_total": int(len(incidents)),
        "incidents_open": int(open_mask.sum()) if len(open_mask) else 0,
        "incidents_critical": int((incidents["severity"] == "critique").sum()) if not incidents.empty else 0,
        "incident_mean_duration_min": _safe(incidents["duration_min"].mean()) if not incidents.empty else float("nan"),
        "missing_measurements_pct": _safe(reachable[["cpu_pct", "memory_pct"]].isna().any(axis=1).mean() * 100) if len(reachable) else float("nan"),
        "anomalous_rows_rules": int(detections["rule_anomaly"].sum()),
        "anomalous_rows_ml": int(detections["ml_anomaly"].sum()) if "ml_anomaly" in detections else 0,
        "period_start": detections["timestamp"].min(),
        "period_end": detections["timestamp"].max(),
    }


def availability_by_device(detections: pd.DataFrame) -> pd.DataFrame:
    summary = detections.groupby(["device_id"], sort=False).agg(
        availability_pct=("available", lambda s: s.mean() * 100),
        latency_mean_ms=("latency_ms", "mean"),
        latency_p95_ms=("latency_ms", lambda s: s.quantile(0.95) if s.notna().any() else np.nan),
        packet_loss_mean_pct=("packet_loss_pct", "mean"),
        cpu_max_pct=("cpu_pct", "max"),
        memory_max_pct=("memory_pct", "max"),
    )
    return summary.round(2).reset_index()
