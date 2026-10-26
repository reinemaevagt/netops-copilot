"""Construction des incidents et aide au diagnostic.

Étapes :
1. chaque mesure anormale (règles et/ou ML) reçoit une catégorie :
   indisponibilité > perte de paquets > surcharge > latence (ordre de priorité) ;
2. les mesures anormales consécutives d'un même équipement sont regroupées
   en épisodes (tolérance de quelques mesures normales entre deux alertes) ;
3. un épisode devient un incident s'il est PERSISTANT (anti-rebond) : les
   anomalies ponctuelles isolées (pics bénins, bruit) sont écartées ;
4. les incidents simultanés situés sur un même chemin réseau sont corrélés :
   l'équipement le plus en amont est désigné « cause racine probable » ;
5. chaque incident reçoit des preuves chiffrées, des HYPOTHÈSES de cause et
   des recommandations. Les causes ne sont jamais présentées comme certaines.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .detection import FEATURE_LABELS
from .topology import downstream_devices, is_on_internet_path, probe_path

CATEGORY_PRIORITY = ["indisponibilite", "perte_paquets", "surcharge", "latence"]
METRIC_CATEGORY = {
    "available": "indisponibilite",
    "packet_loss_pct": "perte_paquets",
    "cpu_pct": "surcharge",
    "memory_pct": "surcharge",
    "latency_ms": "latence",
}
CATEGORY_TITLES = {
    "indisponibilite": "Équipement indisponible",
    "perte_paquets": "Perte de paquets",
    "surcharge": "Surcharge de ressources",
    "latence": "Latence élevée",
}
CATEGORY_METRICS = {
    "indisponibilite": ["available"],
    "perte_paquets": ["packet_loss_pct", "latency_ms"],
    "surcharge": ["cpu_pct", "memory_pct", "latency_ms"],
    "latence": ["latency_ms", "cpu_pct"],
}
INCIDENT_STATUSES = ["Ouvert", "En cours d'analyse", "Rétabli – à confirmer", "Clos"]
SEVERITY_RANK = {"critique": 3, "majeure": 2, "mineure": 1}


@dataclass(frozen=True)
class IncidentSettings:
    min_points: int = 3  # nombre minimal de mesures anormales pour ouvrir un incident
    max_gap_samples: int = 2  # mesures normales tolérées à l'intérieur d'un épisode


def _row_category(row: pd.Series) -> str | None:
    if row["rule_anomaly"] and row["rule_metrics"]:
        categories = {METRIC_CATEGORY[m] for m in row["rule_metrics"].split(",")}
        return next(c for c in CATEGORY_PRIORITY if c in categories)
    if row.get("ml_anomaly", False) and row.get("ml_top_feature"):
        return METRIC_CATEGORY.get(row["ml_top_feature"])
    return None


def _episodes(positions: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """Découpe une suite d'indices croissants en groupes quasi contigus."""
    if len(positions) == 0:
        return []
    cut_points = np.flatnonzero(np.diff(positions) > max_gap + 1) + 1
    return np.split(positions, cut_points)


def _evidence(episode: pd.DataFrame, reference: pd.DataFrame, category: str) -> tuple[list[str], dict]:
    lines, peaks = [], {}
    for metric in CATEGORY_METRICS[category]:
        if metric == "available":
            down = int((episode["available"] == 0).sum())
            lines.append(f"Équipement injoignable sur {down} mesure(s) consécutive(s) (perte 100 %)")
            peaks[metric] = 0
            continue
        values = episode[metric].dropna()
        if values.empty:
            continue
        label, unit = FEATURE_LABELS[metric]
        baseline = reference[metric].median() if not reference.empty else np.nan
        peak = float(values.max())
        peaks[metric] = peak
        baseline_text = f"{baseline:.1f} {unit}" if not np.isnan(baseline) else "indisponible"
        lines.append(f"{label.capitalize()} : max {peak:.1f} {unit}, moyenne {values.mean():.1f} {unit} "
                     f"(référence de l'équipement ≈ {baseline_text})")
    return lines, peaks


def _hypotheses_and_actions(category: str, role: str, peaks: dict, ml_only: bool) -> tuple[list[str], list[str]]:
    if ml_only:
        return (
            ["Combinaison de métriques inhabituelle sans dépassement de seuil : "
             "peut être un début de dégradation… ou un faux positif du modèle."],
            ["Vérifier manuellement les graphiques de l'équipement sur la période",
             "Comparer avec le journal des changements (maintenance, mise à jour)",
             "Si l'alerte se répète sans cause, réentraîner le modèle sur un historique plus long"],
        )
    cpu_peak = peaks.get("cpu_pct", 0.0) or 0.0
    memory_peak = peaks.get("memory_pct", 0.0) or 0.0

    if category == "indisponibilite":
        hypotheses = ["Panne matérielle ou coupure d'alimentation",
                      "Redémarrage ou blocage du système (crash logiciel)",
                      "Perte du seul chemin de supervision : l'équipement peut être actif mais injoignable"]
        actions = ["Tenter un accès console / hors bande (OOB) pour confirmer l'état réel",
                   "Consulter les journaux syslog précédant la coupure",
                   "Vérifier l'alimentation et l'état des liens physiques",
                   "Basculer sur l'équipement de secours s'il existe (VRRP/HSRP)"]
        if role == "router":
            actions.append("Prévenir les utilisateurs et ouvrir un ticket auprès du fournisseur d'accès si le lien WAN est en cause")
    elif category == "perte_paquets":
        hypotheses = ["Défaut physique : câble, connecteur ou module optique (SFP) dégradé",
                      "Erreurs d'interface (CRC, désaccord de duplex/vitesse)",
                      "Saturation des tampons d'un port (micro-rafales de trafic)"]
        actions = ["Relever les compteurs d'erreurs et de rejets des interfaces (CRC, input errors, drops)",
                   "Contrôler la négociation vitesse/duplex des deux côtés du lien",
                   "Remplacer le câble ou le module optique suspect, ou changer de port",
                   "Vérifier le taux d'utilisation des liens montants"]
    elif category == "surcharge":
        if cpu_peak >= 80 and memory_peak >= 85:
            hypotheses = ["Pic d'activité applicative (forte demande utilisateurs)",
                          "Fuite mémoire entraînant une pression sur le CPU (swap, ramasse-miettes)"]
        elif memory_peak >= 85:
            hypotheses = ["Fuite mémoire probable ou cache surdimensionné"]
        else:
            hypotheses = ["Processus très consommateur (tâche planifiée, sauvegarde, boucle applicative)",
                          "Hausse de charge applicative"]
        actions = ["Identifier les processus les plus consommateurs (CPU et mémoire)",
                   "Consulter les journaux applicatifs sur la même période",
                   "Décaler les traitements planifiés hors des heures de pointe",
                   "Envisager une répartition de charge ou un dimensionnement supérieur si la tendance se confirme"]
    else:  # latence
        if cpu_peak >= 60 and role in ("firewall", "router"):
            hypotheses = ["Charge de traitement élevée de l'équipement (inspection, ACL, sessions) ralentissant le transit",
                          "Congestion d'une interface et files d'attente saturées"]
        else:
            hypotheses = ["Congestion d'un lien ou file d'attente saturée",
                          "Changement de routage vers un chemin moins performant"]
        actions = ["Localiser le saut fautif avec un traceroute / mtr depuis la sonde",
                   "Vérifier l'utilisation et les rejets des interfaces sur le chemin",
                   "Contrôler les politiques de QoS et la table de sessions (pare-feu)",
                   "Comparer avec les changements récents de configuration"]
    return hypotheses, actions


def _priority(severity: str, criticality: int, is_symptom: bool) -> str:
    if severity == "critique":
        level = 1 if criticality >= 3 else 2
    elif severity == "majeure":
        level = 2 if criticality >= 3 else 3
    else:
        level = 4
    if is_symptom:
        level = min(4, level + 1)
    return f"P{level}"


def build_incidents(detections: pd.DataFrame, graph: nx.Graph,
                    settings: IncidentSettings | None = None) -> pd.DataFrame:
    """Transforme les mesures anormales en incidents diagnostiqués."""
    settings = settings or IncidentSettings()
    data = detections.sort_values(["device_id", "timestamp"], ignore_index=True)
    if "ml_anomaly" not in data.columns:
        data["ml_anomaly"] = False
        data["ml_top_feature"] = ""
    data["is_anomalous"] = data["rule_anomaly"] | data["ml_anomaly"]
    data["category"] = [_row_category(row) if row["is_anomalous"] else None for _, row in data.iterrows()]

    raw_incidents = []
    for device_id, device_rows in data.groupby("device_id", sort=False):
        device_rows = device_rows.reset_index(drop=True)
        reference = device_rows[~device_rows["is_anomalous"]]
        last_timestamp = device_rows["timestamp"].max()
        anomalous_positions = np.flatnonzero(device_rows["is_anomalous"].to_numpy())
        for positions in _episodes(anomalous_positions, settings.max_gap_samples):
            episode = device_rows.iloc[positions]
            counts = episode["category"].value_counts()
            category = sorted(counts.index, key=lambda c: (-counts[c], CATEGORY_PRIORITY.index(c)))[0]
            n_points = len(episode)
            if n_points < settings.min_points and not (category == "indisponibilite" and n_points >= 2):
                continue  # anomalie ponctuelle : non promue en incident

            rule_points = int(episode["rule_anomaly"].sum())
            ml_points = int(episode["ml_anomaly"].sum())
            critical_points = int((episode["rule_severity"] == "critique").sum())
            if category == "indisponibilite" or critical_points >= 2:
                severity = "critique"
            elif rule_points >= settings.min_points:
                severity = "majeure"
            else:
                severity = "mineure"
            ml_only = rule_points == 0
            source = "Règles + ML" if rule_points and ml_points else ("ML" if ml_only else "Règles")

            evidence, peaks = _evidence(episode, reference, category)
            hypotheses, actions = _hypotheses_and_actions(category, episode["role"].iloc[0], peaks, ml_only)
            examples = episode.loc[episode["rule_reasons"] != "", "rule_reasons"]
            if not examples.empty:
                evidence.append(f"Exemple d'alerte : {examples.iloc[len(examples) // 2]}")
            elif ml_only:
                evidence.append(f"Explication ML : {episode['ml_explanation'].iloc[len(episode) // 2]}")

            start, end = episode["timestamp"].min(), episode["timestamp"].max()
            raw_incidents.append({
                "device_id": device_id,
                "device_name": episode["device_name"].iloc[0] if "device_name" in episode else device_id,
                "role": episode["role"].iloc[0] if "role" in episode else "",
                "category": category,
                "title": CATEGORY_TITLES[category] + (" (signal ML faible)" if ml_only else ""),
                "severity": severity,
                "start": start,
                "end": end,
                "duration_min": round((end - start).total_seconds() / 60, 1),
                "n_points": n_points,
                "detection_source": source,
                "evidence": evidence,
                "hypotheses": hypotheses,
                "recommendations": actions,
                "status": "Ouvert" if end >= last_timestamp else "Rétabli – à confirmer",
            })

    return _correlate_and_finalize(raw_incidents, graph)


def _correlate_and_finalize(raw_incidents: list[dict], graph: nx.Graph) -> pd.DataFrame:
    columns = ["incident_id", "priority", "severity", "title", "device_id", "device_name", "role",
               "category", "status", "start", "end", "duration_min", "n_points", "detection_source",
               "probable_root_cause", "impact", "evidence", "hypotheses", "recommendations"]
    if not raw_incidents:
        return pd.DataFrame(columns=columns)

    for incident in raw_incidents:
        device = incident["device_id"]
        upstream = probe_path(graph, device)[:-1] if device in graph else []
        root = None
        for other in raw_incidents:
            overlaps = other["start"] <= incident["end"] and incident["start"] <= other["end"]
            compatible = other["category"] in (incident["category"], "indisponibilite")
            if other is not incident and other["device_id"] in upstream and overlaps and compatible:
                # garder l'équipement le plus en amont (le plus proche du cœur)
                if root is None or upstream.index(other["device_id"]) < upstream.index(root):
                    root = other["device_id"]
        incident["probable_root_cause"] = root or ""
        if root:
            incident["hypotheses"] = [f"Symptôme probable d'un incident en amont sur {root} "
                                      f"(même type d'anomalie, même période, sur le chemin de supervision)"] + incident["hypotheses"]
            incident["recommendations"] = [f"Traiter d'abord l'incident de {root}, puis vérifier si celui-ci disparaît"] + incident["recommendations"]

        impacts = []
        if device in graph:
            dependents = downstream_devices(graph, device)
            if dependents and incident["category"] != "surcharge":
                impacts.append(f"{len(dependents)} équipement(s) en aval potentiellement affecté(s) : {', '.join(dependents)}")
            if is_on_internet_path(graph, device) and incident["category"] in ("indisponibilite", "perte_paquets", "latence"):
                impacts.append("Accès WAN/Internet de l'entreprise potentiellement dégradé")
        incident["impact"] = "; ".join(impacts) if impacts else "Impact limité à l'équipement (d'après la topologie)"
        criticality = graph.nodes[device]["criticality"] if device in graph else 2
        incident["priority"] = _priority(incident["severity"], criticality, bool(root))

    incidents = pd.DataFrame(raw_incidents)
    incidents = incidents.sort_values(["priority", "start"], ignore_index=True)
    incidents.insert(0, "incident_id", [f"INC-{number:04d}" for number in range(1, len(incidents) + 1)])
    return incidents[columns]


def incidents_for_export(incidents: pd.DataFrame) -> pd.DataFrame:
    """Version « à plat » des incidents (listes jointes) pour l'export CSV."""
    flat = incidents.copy()
    for column in ("evidence", "hypotheses", "recommendations"):
        if column in flat.columns:
            flat[column] = flat[column].apply(lambda items: " | ".join(items) if isinstance(items, list) else items)
    return flat
