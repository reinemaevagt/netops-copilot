"""Détection d'anomalies : règles explicites + Isolation Forest.

1. Règles : seuils lisibles (avertissement / critique) sur chaque métrique.
   Chaque alerte est accompagnée de la valeur mesurée et du seuil franchi.

2. Machine learning non supervisé : Isolation Forest (scikit-learn).
   - Le modèle apprend le comportement normal sur une période de référence
     (historique simulé sans défaut injecté).
   - Les métriques sont normalisées PAR ÉQUIPEMENT (écart robuste à la
     médiane, échelle = IQR/1.349) : un serveur à 35 % CPU et un commutateur
     à 15 % CPU sont tous les deux « normaux ».
   - Le seuil de décision est le quantile des scores d'entraînement
     correspondant au taux de faux positifs accepté (1 % par défaut).
   - L'explication donnée est la ou les métriques les plus éloignées de la
     normale de l'équipement : c'est une aide à l'interprétation, pas une
     attribution exacte de la décision du modèle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ML_FEATURES = ["latency_ms", "packet_loss_pct", "cpu_pct", "memory_pct"]
FEATURE_LABELS = {
    "latency_ms": ("latence", "ms"),
    "packet_loss_pct": ("perte de paquets", "%"),
    "cpu_pct": ("CPU", "%"),
    "memory_pct": ("mémoire", "%"),
    "available": ("disponibilité", ""),
}
# échelle minimale par métrique pour éviter des écarts robustes infinis
SCALE_FLOOR = {"latency_ms": 0.25, "packet_loss_pct": 0.25, "cpu_pct": 2.0, "memory_pct": 1.0}

SEVERITY_ORDER = {"normal": 0, "avertissement": 1, "critique": 2}


@dataclass(frozen=True)
class RuleThresholds:
    latency_warning_ms: float = 20.0
    latency_critical_ms: float = 50.0
    loss_warning_pct: float = 1.0
    loss_critical_pct: float = 5.0
    cpu_warning_pct: float = 80.0
    cpu_critical_pct: float = 90.0
    memory_warning_pct: float = 85.0
    memory_critical_pct: float = 95.0

    def as_rules(self) -> list[tuple[str, float, float]]:
        return [
            ("latency_ms", self.latency_warning_ms, self.latency_critical_ms),
            ("packet_loss_pct", self.loss_warning_pct, self.loss_critical_pct),
            ("cpu_pct", self.cpu_warning_pct, self.cpu_critical_pct),
            ("memory_pct", self.memory_warning_pct, self.memory_critical_pct),
        ]


def _format_value(metric: str, value: float) -> str:
    label, unit = FEATURE_LABELS[metric]
    return f"{label} {value:.1f} {unit}".strip()


def detect_with_rules(metrics: pd.DataFrame, thresholds: RuleThresholds | None = None) -> pd.DataFrame:
    """Ajoute rule_severity, rule_anomaly, rule_metrics et rule_reasons."""
    thresholds = thresholds or RuleThresholds()
    result = metrics.copy()
    severity = np.zeros(len(result), dtype=int)
    reasons: list[list[str]] = [[] for _ in range(len(result))]
    breached: list[list[str]] = [[] for _ in range(len(result))]

    unavailable = (result["available"] == 0).to_numpy()
    severity[unavailable] = 2
    for position in np.flatnonzero(unavailable):
        reasons[position].append("équipement injoignable (disponibilité = 0)")
        breached[position].append("available")

    for metric, warning, critical in thresholds.as_rules():
        values = result[metric].to_numpy(dtype=float)
        # NaN : pas de mesure -> pas d'alerte de seuil (l'indisponibilité est traitée à part)
        with np.errstate(invalid="ignore"):
            is_critical = values >= critical
            is_warning = (values >= warning) & ~is_critical
        for level, mask, limit in ((2, is_critical, critical), (1, is_warning, warning)):
            positions = np.flatnonzero(mask & ~unavailable)
            severity[positions] = np.maximum(severity[positions], level)
            word = "critique" if level == 2 else "avertissement"
            unit = FEATURE_LABELS[metric][1]
            for position in positions:
                reasons[position].append(f"{_format_value(metric, values[position])} ≥ seuil {word} {limit:g} {unit}".rstrip())
                breached[position].append(metric)

    labels = np.array(["normal", "avertissement", "critique"])
    result["rule_severity"] = labels[severity]
    result["rule_anomaly"] = severity > 0
    result["rule_metrics"] = [",".join(items) for items in breached]
    result["rule_reasons"] = ["; ".join(items) for items in reasons]
    return result


@dataclass
class IsolationForestDetector:
    """Détecteur Isolation Forest avec normalisation robuste par équipement."""

    expected_false_positive_rate: float = 0.01
    n_estimators: int = 200
    random_state: int = 0
    model: IsolationForest | None = field(default=None, init=False, repr=False)
    device_stats: dict = field(default_factory=dict, init=False, repr=False)
    role_stats: dict = field(default_factory=dict, init=False, repr=False)
    decision_threshold: float | None = field(default=None, init=False)
    training_rows: int = field(default=0, init=False)

    @staticmethod
    def _robust_stats(frame: pd.DataFrame) -> dict:
        stats = {}
        for feature in ML_FEATURES:
            values = frame[feature].dropna()
            median = float(values.median()) if len(values) else 0.0
            iqr = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else 0.0
            stats[feature] = (median, max(iqr / 1.349, SCALE_FLOOR[feature]))
        return stats

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = pd.DataFrame(index=frame.index, columns=ML_FEATURES, dtype=float)
        for device_id, group in frame.groupby("device_id"):
            stats = self.device_stats.get(device_id)
            if stats is None:  # équipement absent de l'historique : statistiques du rôle
                role = group["role"].iloc[0] if "role" in group.columns else None
                stats = self.role_stats.get(role)
            if stats is None:
                raise ValueError(f"Aucune référence d'apprentissage pour l'équipement {device_id}.")
            for feature in ML_FEATURES:
                median, scale = stats[feature]
                normalized.loc[group.index, feature] = (group[feature] - median) / scale
        return normalized

    def fit(self, history: pd.DataFrame) -> "IsolationForestDetector":
        if not 0 < self.expected_false_positive_rate < 0.5:
            raise ValueError("Le taux de faux positifs attendu doit être compris entre 0 et 0,5.")
        reference = history[history["available"] == 1]
        self.device_stats = {device: self._robust_stats(group) for device, group in reference.groupby("device_id")}
        if "role" in reference.columns:
            self.role_stats = {role: self._robust_stats(group) for role, group in reference.groupby("role")}
        features = self._normalize(reference).dropna()
        if len(features) < 50:
            raise ValueError("Historique trop court pour entraîner le modèle (≥ 50 mesures complètes).")
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination="auto",
                                     random_state=self.random_state)
        self.model.fit(features.to_numpy())
        training_scores = -self.model.score_samples(features.to_numpy())
        self.decision_threshold = float(np.quantile(training_scores, 1 - self.expected_false_positive_rate))
        self.training_rows = len(features)
        return self

    def score(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Ajoute ml_evaluated, ml_score, ml_anomaly, ml_top_feature, ml_explanation."""
        if self.model is None:
            raise RuntimeError("Le modèle doit être entraîné (fit) avant le scoring.")
        result = metrics.copy()
        normalized = self._normalize(result)
        evaluable = normalized.notna().all(axis=1) & (result["available"] == 1)

        scores = pd.Series(np.nan, index=result.index)
        if evaluable.any():
            scores.loc[evaluable] = -self.model.score_samples(normalized.loc[evaluable].to_numpy())
        result["ml_evaluated"] = evaluable
        result["ml_score"] = scores.round(4)
        result["ml_anomaly"] = evaluable & (scores > self.decision_threshold)

        top_features, explanations = [], []
        for index in result.index:
            if not result.at[index, "ml_anomaly"]:
                top_features.append("")
                explanations.append("" if evaluable[index] else "non évalué (mesure incomplète ou équipement injoignable)")
                continue
            deviations = normalized.loc[index].abs().sort_values(ascending=False)
            device_stats = self.device_stats.get(result.at[index, "device_id"]) or self.role_stats.get(result.at[index, "role"])
            parts = []
            for feature in deviations.index[:2]:
                if deviations[feature] < 3:  # écart non significatif
                    continue
                median, _ = device_stats[feature]
                value = result.at[index, feature]
                parts.append(f"{_format_value(feature, value)} (normale ≈ {median:.1f}, écart robuste {normalized.at[index, feature]:+.1f})")
            top_features.append(deviations.index[0])
            explanations.append("; ".join(parts) if parts else "combinaison inhabituelle de métriques, aucun écart individuel marqué")
        result["ml_top_feature"] = top_features
        result["ml_explanation"] = explanations
        return result


def compare_methods(detections: pd.DataFrame) -> dict:
    """Compare règles et ML ; évalue chaque méthode si la vérité terrain existe."""
    evaluated = detections[detections["ml_evaluated"]]
    rules = evaluated["rule_anomaly"]
    ml = evaluated["ml_anomaly"]
    agreement = {
        "lignes_evaluees": int(len(evaluated)),
        "regles_et_ml": int((rules & ml).sum()),
        "regles_seules": int((rules & ~ml).sum()),
        "ml_seul": int((~rules & ml).sum()),
        "aucune": int((~rules & ~ml).sum()),
        "lignes_non_evaluees_par_ml": int(len(detections) - len(evaluated)),
    }
    comparison = {"accord": agreement, "evaluation": None}

    if "ground_truth_fault" not in detections.columns:
        return comparison
    truth = detections["ground_truth_fault"].astype(bool)
    combined = detections["rule_anomaly"] | detections["ml_anomaly"]
    rows = []
    for name, predicted in (("Règles", detections["rule_anomaly"]),
                            ("Isolation Forest", detections["ml_anomaly"]),
                            ("Règles OU ML", combined)):
        true_pos = int((predicted & truth).sum())
        false_pos = int((predicted & ~truth).sum())
        false_neg = int((~predicted & truth).sum())
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else float("nan")
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else float("nan")
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator > 0 else float("nan")  # NaN > 0 est faux
        rows.append({"methode": name, "vrais_positifs": true_pos, "faux_positifs": false_pos,
                     "faux_negatifs": false_neg, "precision": precision, "rappel": recall, "f1": f1})
    comparison["evaluation"] = pd.DataFrame(rows)
    return comparison
