"""NetOps Copilot – tableau de bord Streamlit.

Lancement :  streamlit run app.py

L'interface ne contient pas de logique métier : elle appelle
``netops_copilot.pipeline.run_analysis`` puis affiche les résultats.
"""

from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from netops_copilot import __version__  # noqa: E402
from netops_copilot.charts import (  # noqa: E402
    SEVERITY_COLORS, STATUS_ICONS, agreement_figure, device_styles, metric_figure,
    ml_score_figure, topology_figure,
)
from netops_copilot.data_sources import CsvMetricsSource  # noqa: E402
from netops_copilot.detection import RuleThresholds  # noqa: E402
from netops_copilot.diagnosis import INCIDENT_STATUSES, incidents_for_export  # noqa: E402
from netops_copilot.kpi import availability_by_device, device_status  # noqa: E402
from netops_copilot.pipeline import AnalysisConfig, run_analysis  # noqa: E402
from netops_copilot.simulation import SCENARIOS  # noqa: E402
from netops_copilot.topology import build_default_topology, device_table, monitored_devices  # noqa: E402

st.set_page_config(page_title="NetOps Copilot", page_icon="📡", layout="wide")

METRIC_CHOICES = {"latency_ms": "Latence", "packet_loss_pct": "Perte de paquets",
                  "cpu_pct": "CPU", "memory_pct": "Mémoire"}
SEVERITY_ICONS = {"critique": "🔴", "majeure": "🟠", "mineure": "🟡"}


@st.cache_resource(show_spinner="Simulation et analyse en cours…")
def analyse(scenario_key: str, duration_minutes: int, interval_seconds: int, seed: int,
            fp_rate: float, csv_bytes: bytes | None):
    config = AnalysisConfig(scenario_key=scenario_key, duration_minutes=duration_minutes,
                            interval_seconds=interval_seconds, seed=seed,
                            expected_false_positive_rate=fp_rate)
    source = None
    if csv_bytes is not None:
        source = CsvMetricsSource(io.BytesIO(csv_bytes), known_devices=monitored_devices(build_default_topology()))
    return run_analysis(config, source)


def full_width(function) -> dict:
    """Argument « pleine largeur » compatible avec les anciennes et nouvelles versions de Streamlit."""
    parameters = inspect.signature(function).parameters
    return {"use_container_width": True} if "use_container_width" in parameters else {"width": "stretch"}


def fmt(value, unit: str = "", decimals: int = 1) -> str:
    return "n/d" if value is None or pd.isna(value) else f"{value:.{decimals}f}{unit}"


# --------------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.title("📡 NetOps Copilot")
    st.caption(f"v{__version__} – supervision d'un réseau **simulé**")

    data_mode = st.radio("Source des données", ["Simulation", "Importer un CSV"], horizontal=True)
    csv_bytes = None
    if data_mode == "Simulation":
        scenario_key = st.selectbox("Scénario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k].label)
        st.caption(SCENARIOS[scenario_key].description)
    else:
        scenario_key = "normal"
        uploaded = st.file_uploader("Fichier CSV (format NetOps Copilot)", type="csv")
        st.caption("Colonnes : timestamp, device_id, latency_ms, packet_loss_pct, cpu_pct, memory_pct, available. "
                   "Un export de l'onglet *Export* peut être réimporté.")
        if uploaded is not None:
            csv_bytes = uploaded.getvalue()

    with st.expander("Paramètres de simulation et de détection"):
        duration_minutes = st.slider("Durée simulée (minutes)", 120, 1440, 360, step=60)
        interval_seconds = st.select_slider("Intervalle de collecte (s)", options=[30, 60, 120, 300], value=60)
        seed = int(st.number_input("Graine aléatoire", min_value=0, max_value=99_999, value=42, step=1))
        fp_rate = st.slider("Taux de faux positifs accepté par le ML (%)", 0.5, 5.0, 1.0, step=0.5) / 100

try:
    if data_mode == "Importer un CSV" and csv_bytes is None:
        st.info("Importez un fichier CSV dans la barre latérale pour lancer l'analyse.")
        st.stop()
    result = analyse(scenario_key, duration_minutes, interval_seconds, seed, fp_rate, csv_bytes)
except ValueError as error:
    st.error(f"Analyse impossible : {error}")
    st.stop()

graph = result.graph
detections = result.detections
styles = device_styles(graph)
all_devices = monitored_devices(graph)
thresholds: RuleThresholds = result.config.thresholds
has_ground_truth = "ground_truth_fault" in detections.columns

# état de traitement modifiable par l'utilisateur (conservé pendant la session)
overrides = st.session_state.setdefault("status_overrides", {})
incidents = result.incidents.copy()
if not incidents.empty:
    incidents["key"] = [f"{scenario_key}|{seed}|{d}|{s}" for d, s in zip(incidents["device_id"], incidents["start"])]
    incidents["status"] = [overrides.get(k, s) for k, s in zip(incidents["key"], incidents["status"])]

with st.sidebar:
    st.subheader("Filtres")
    roles = sorted(detections["role"].unique())
    selected_roles = st.multiselect("Rôles", roles, default=roles)
    default_devices = (list(dict.fromkeys(incidents["device_id"]))[:4] if not incidents.empty else all_devices[:3])
    candidate_devices = [d for d in all_devices if detections.loc[detections["device_id"] == d, "role"].iloc[0] in selected_roles]
    selected_devices = st.multiselect("Équipements (graphiques)", candidate_devices,
                                      default=[d for d in default_devices if d in candidate_devices],
                                      max_selections=6)
    severities = st.multiselect("Gravité des incidents", list(SEVERITY_ICONS), default=list(SEVERITY_ICONS))

# --------------------------------------------------------------------------- en-tête et KPI
source_label = "CSV importé" if csv_bytes is not None else f"Simulation – {SCENARIOS[scenario_key].label}"
st.title("Supervision réseau intelligente")
if csv_bytes is None:
    st.warning("⚠️ **Données simulées.** Ce tableau de bord est un démonstrateur pédagogique : "
               "les mesures ne proviennent d'aucun équipement réel et les diagnostics sont des hypothèses.")
else:
    st.info("Données importées depuis un CSV : la vérité terrain n'est connue que si le fichier contient la colonne ground_truth_fault ; les diagnostics restent des hypothèses.")
kpis = result.kpis
st.caption(f"Source : {source_label} · période du {kpis['period_start']:%d/%m/%Y %H:%M} au "
           f"{kpis['period_end']:%d/%m/%Y %H:%M} · {len(detections):,} mesures".replace(",", " "))

columns = st.columns(6)
columns[0].metric("Disponibilité globale", fmt(kpis["availability_pct"], " %", 2))
columns[1].metric("Latence moyenne", fmt(kpis["latency_mean_ms"], " ms"), help=f"P95 : {fmt(kpis['latency_p95_ms'], ' ms')}")
columns[2].metric("Perte moyenne", fmt(kpis["packet_loss_mean_pct"], " %", 2))
columns[3].metric("Incidents", kpis["incidents_total"], help=f"dont {kpis['incidents_critical']} critique(s)")
columns[4].metric("Incidents ouverts", int(incidents["status"].isin(["Ouvert", "En cours d'analyse"]).sum()) if not incidents.empty else 0)
columns[5].metric("Équipements dégradés / HS", f"{kpis['devices_degraded']} / {kpis['devices_down']}",
                  help="Statut en fin de période")

tab_overview, tab_metrics, tab_incidents, tab_detection, tab_methods, tab_export = st.tabs(
    ["🗺️ Vue d'ensemble", "📈 Métriques", "🚨 Incidents", "🤖 Règles vs ML", "📘 Méthodes & limites", "💾 Export"])

# --------------------------------------------------------------------------- vue d'ensemble
with tab_overview:
    timestamps = list(pd.DatetimeIndex(detections["timestamp"].unique()).sort_values())
    replay_at = st.select_slider("Rejouer l'état du réseau à l'instant :", options=timestamps, value=timestamps[-1],
                                 format_func=lambda t: pd.Timestamp(t).strftime("%d/%m %H:%M"))
    statuses_at = device_status(detections, at=pd.Timestamp(replay_at))
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(topology_figure(graph, statuses_at), **full_width(st.plotly_chart))
    with right:
        st.markdown("**État des équipements**")
        table = statuses_at[statuses_at["role"].isin(selected_roles)].copy()
        table["status"] = table["status"].map(lambda s: f"{STATUS_ICONS[s]} {s}")
        st.dataframe(table[["device_id", "status", "last_latency_ms", "last_packet_loss_pct", "last_cpu_pct", "last_memory_pct"]],
                     hide_index=True, **full_width(st.dataframe),
                     column_config={"device_id": "Équipement", "status": "Statut", "last_latency_ms": "Latence (ms)",
                                    "last_packet_loss_pct": "Perte (%)", "last_cpu_pct": "CPU (%)", "last_memory_pct": "Mém. (%)"})
        st.markdown("**Incidents de la période**")
        if incidents.empty:
            st.success("Aucun incident détecté sur la période.")
        for _, incident in incidents.iterrows():
            st.markdown(f"{SEVERITY_ICONS[incident['severity']]} **{incident['priority']}** – {incident['title']} "
                        f"sur `{incident['device_id']}` ({incident['status']})")

# --------------------------------------------------------------------------- métriques
with tab_metrics:
    if not selected_devices:
        st.info("Sélectionnez au moins un équipement dans la barre latérale.")
    else:
        st.caption("Les croix signalent les mesures jugées anormales (règles ou ML) ; les bandes colorées "
                   "les périodes d'incident ; les pointillés les seuils d'alerte.")
        for metric in METRIC_CHOICES:
            st.plotly_chart(metric_figure(detections, metric, selected_devices, styles, thresholds, incidents),
                            **full_width(st.plotly_chart))
        missing = detections[(detections["available"] == 1) & detections["cpu_pct"].isna()]
        if not missing.empty:
            st.caption(f"ℹ️ {len(missing)} mesure(s) CPU/mémoire manquante(s) (collecte en échec simulée) : "
                       "elles apparaissent comme des trous dans les courbes et ne sont pas évaluées par le ML.")
        st.markdown("**Synthèse par équipement**")
        st.dataframe(availability_by_device(detections), hide_index=True, **full_width(st.dataframe))

# --------------------------------------------------------------------------- incidents
with tab_incidents:
    visible = incidents[incidents["severity"].isin(severities)] if not incidents.empty else incidents
    if visible.empty:
        st.success("Aucun incident pour les filtres sélectionnés.")
    else:
        st.caption("L'état de traitement est modifiable (colonne *État*) ; il est conservé pendant la session.")
        editor = st.data_editor(
            visible[["incident_id", "priority", "severity", "title", "device_id", "start", "duration_min",
                     "detection_source", "probable_root_cause", "status"]],
            hide_index=True, **full_width(st.data_editor), key=f"editor_{scenario_key}_{seed}",
            disabled=["incident_id", "priority", "severity", "title", "device_id", "start", "duration_min",
                      "detection_source", "probable_root_cause"],
            column_config={
                "incident_id": "ID", "priority": "Priorité", "severity": "Gravité", "title": "Incident",
                "device_id": "Équipement", "start": st.column_config.DatetimeColumn("Début", format="DD/MM HH:mm"),
                "duration_min": st.column_config.NumberColumn("Durée (min)", format="%.0f"),
                "detection_source": "Détecté par", "probable_root_cause": "Cause racine probable",
                "status": st.column_config.SelectboxColumn("État", options=INCIDENT_STATUSES, required=True),
            })
        changed = False
        for key, new_status, old_status in zip(visible["key"], editor["status"], visible["status"]):
            if new_status != old_status:
                overrides[key] = new_status
                changed = True
        if changed:
            st.rerun()

        for _, incident in visible.iterrows():
            header = (f"{SEVERITY_ICONS[incident['severity']]} {incident['incident_id']} · {incident['priority']} · "
                      f"{incident['title']} – {incident['device_name']} ({incident['device_id']})")
            with st.expander(header):
                a, b, c = st.columns(3)
                a.markdown(f"**Gravité :** {incident['severity']}  \n**Priorité :** {incident['priority']}  \n**État :** {incident['status']}")
                b.markdown(f"**Début :** {incident['start']:%d/%m %H:%M}  \n**Fin :** {incident['end']:%d/%m %H:%M}  \n"
                           f"**Durée :** {incident['duration_min']:.0f} min ({incident['n_points']} mesures)")
                c.markdown(f"**Détecté par :** {incident['detection_source']}  \n**Impact :** {incident['impact']}")
                st.markdown("**Métriques justifiant l'alerte**")
                st.markdown("\n".join(f"- {line}" for line in incident["evidence"]))
                st.markdown("**Hypothèses de cause (non confirmées)**")
                st.markdown("\n".join(f"- {line}" for line in incident["hypotheses"]))
                st.markdown("**Recommandations**")
                st.markdown("\n".join(f"{i}. {line}" for i, line in enumerate(incident["recommendations"], 1)))
                st.plotly_chart(metric_figure(detections, {"indisponibilite": "packet_loss_pct", "perte_paquets": "packet_loss_pct",
                                                           "surcharge": "cpu_pct", "latence": "latency_ms"}[incident["category"]],
                                              [incident["device_id"]], styles, thresholds, incidents[incidents["incident_id"] == incident["incident_id"]]),
                                **full_width(st.plotly_chart), key=f"chart_{incident['incident_id']}")

# --------------------------------------------------------------------------- règles vs ML
with tab_detection:
    comparison = result.comparison
    agreement = comparison["accord"]
    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(agreement_figure(agreement), **full_width(st.plotly_chart))
        st.caption(f"{agreement['lignes_evaluees']} mesures évaluées par les deux méthodes ; "
                   f"{agreement['lignes_non_evaluees_par_ml']} non évaluées par le ML (équipement injoignable "
                   "ou mesure incomplète) – ces cas restent couverts par les règles.")
    with right:
        if comparison["evaluation"] is not None and has_ground_truth:
            st.markdown("**Évaluation face à la vérité terrain de la simulation**")
            evaluation = comparison["evaluation"].copy()
            st.dataframe(evaluation, hide_index=True, **full_width(st.dataframe), column_config={
                "methode": "Méthode", "vrais_positifs": "Vrais positifs", "faux_positifs": "Faux positifs",
                "faux_negatifs": "Faux négatifs",
                "precision": st.column_config.NumberColumn("Précision", format="%.2f"),
                "rappel": st.column_config.NumberColumn("Rappel", format="%.2f"),
                "f1": st.column_config.NumberColumn("F1", format="%.2f")})
            st.caption("La vérité terrain n'existe que parce que les défauts sont injectés par la simulation. "
                       "Évaluation au niveau de chaque mesure (avant regroupement en incidents). "
                       "Valeur vide : non calculable (aucun défaut injecté ou aucune alerte).")
        else:
            st.info("Pas de vérité terrain pour ces données : seule la concordance entre méthodes est affichée.")

    if selected_devices:
        st.plotly_chart(ml_score_figure(detections, selected_devices, styles, result.detector.decision_threshold),
                        **full_width(st.plotly_chart))
    st.markdown("**Mesures signalées par une seule méthode** (utile pour repérer les faux positifs)")
    disagreements = detections[(detections["rule_anomaly"] != detections["ml_anomaly"]) & detections["ml_evaluated"]]
    disagreements = disagreements[disagreements["device_id"].isin(selected_devices or all_devices)]
    st.dataframe(disagreements[["timestamp", "device_id", "latency_ms", "packet_loss_pct", "cpu_pct", "memory_pct",
                                "rule_anomaly", "rule_reasons", "ml_anomaly", "ml_score", "ml_explanation"]
                               + (["ground_truth_fault"] if has_ground_truth else [])],
                 hide_index=True, **full_width(st.dataframe), height=300)

# --------------------------------------------------------------------------- méthodes
with tab_methods:
    st.markdown(f"""
### Chaîne de traitement
1. **Collecte (simulée)** – latence, perte, CPU, mémoire et disponibilité de {len(all_devices)} équipements,
   mesurées depuis une sonde rattachée au commutateur cœur. Un défaut sur un équipement se propage aux
   mesures des équipements situés derrière lui.
2. **Règles explicites** – seuils d'avertissement / critique :
   latence {thresholds.latency_warning_ms:g} / {thresholds.latency_critical_ms:g} ms ·
   perte {thresholds.loss_warning_pct:g} / {thresholds.loss_critical_pct:g} % ·
   CPU {thresholds.cpu_warning_pct:g} / {thresholds.cpu_critical_pct:g} % ·
   mémoire {thresholds.memory_warning_pct:g} / {thresholds.memory_critical_pct:g} % ·
   équipement injoignable = critique.
3. **Isolation Forest** – entraîné sur {result.detector.training_rows} mesures d'un historique simulé *sans défaut*.
   Les métriques sont normalisées par équipement (écart à la médiane / IQR) ; le seuil de décision correspond
   à un taux de faux positifs attendu de {fp_rate:.1%} sur l'historique.
4. **Incidents** – regroupement des anomalies consécutives d'un équipement ; un incident n'est ouvert
   qu'après au moins {result.config.incident_settings.min_points} mesures anormales (2 pour une indisponibilité) :
   les pics isolés sont ignorés (anti-rebond).
5. **Corrélation topologique** – si un équipement situé en amont présente le même type d'anomalie au même
   moment, il est proposé comme *cause racine probable* et l'incident en aval est traité comme un symptôme.
6. **Diagnostic** – preuves chiffrées, hypothèses et recommandations issues d'une base de règles d'expertise.

### Limites à connaître
- Les données sont **simulées** : distributions, pannes et propagation sont volontairement simplifiées.
- Les seuils sont génériques ; sur un vrai réseau ils doivent être adaptés à chaque équipement et lien.
- L'Isolation Forest repère mieux les anomalies touchant **plusieurs métriques** à la fois. Une hausse portant
  sur une seule métrique (ex. latence seule) peut être partiellement manquée : d'où la combinaison avec les règles.
- Le modèle ne peut pas évaluer un équipement injoignable (pas de mesure) ; l'indisponibilité est gérée par règle.
- Avec un taux de faux positifs de {fp_rate:.1%}, quelques mesures normales sont signalées même sans panne
  (visible dans le scénario *Fonctionnement normal*).
- Les causes proposées sont des **hypothèses** : elles orientent l'investigation, elles ne la remplacent pas.
- Ce n'est pas un outil de production et il n'a pas été validé sur un réseau réel.
""")
    st.markdown("### Inventaire simulé")
    st.dataframe(pd.DataFrame(device_table(graph)), hide_index=True, **full_width(st.dataframe))

# --------------------------------------------------------------------------- export
with tab_export:
    st.markdown("Les fichiers exportés contiennent la colonne `source` = `simulation` pour éviter toute "
                "confusion avec des mesures réelles.")
    metrics_csv = detections.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Métriques et détections (CSV)", metrics_csv,
                       file_name=f"netops_metriques_{scenario_key}_seed{seed}.csv", mime="text/csv")
    if not incidents.empty:
        incidents_csv = incidents_for_export(incidents.drop(columns=["key"])).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Incidents (CSV)", incidents_csv,
                           file_name=f"netops_incidents_{scenario_key}_seed{seed}.csv", mime="text/csv")
    else:
        st.caption("Aucun incident à exporter.")
