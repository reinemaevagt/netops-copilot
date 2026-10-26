"""Figures Plotly utilisées par le tableau de bord.

Conventions visuelles :
- chaque équipement garde toujours la même couleur (la couleur suit
  l'équipement, pas son rang dans la sélection) ; au-delà de 8 équipements,
  les teintes sont réutilisées avec un trait pointillé ;
- les couleurs de statut (vert / orange / rouge) sont réservées aux états et
  toujours accompagnées d'un libellé ;
- un seul axe vertical par graphique.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from .detection import FEATURE_LABELS, RuleThresholds
from .kpi import STATUS_DEGRADED, STATUS_DOWN, STATUS_OK
from .topology import INTERNET_NODE, monitored_devices

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS_COLORS = {STATUS_OK: "#0ca30c", STATUS_DEGRADED: "#fab219", STATUS_DOWN: "#d03b3b"}
STATUS_ICONS = {STATUS_OK: "✔", STATUS_DEGRADED: "▲", STATUS_DOWN: "✖"}
SEVERITY_COLORS = {"critique": "#d03b3b", "majeure": "#ec835a", "mineure": "#fab219"}
NEUTRAL = "#8a8984"
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=80, b=10), hovermode="x unified",
              legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
              font=dict(size=12))


def _title(text: str) -> dict:
    """Titre placé au-dessus de la légende (évite les chevauchements)."""
    return dict(text=text, x=0, xanchor="left", y=0.98, yanchor="top")


def device_styles(graph: nx.Graph) -> dict[str, dict]:
    """Couleur et style de trait fixes par équipement."""
    styles = {}
    for index, device_id in enumerate(monitored_devices(graph)):
        styles[device_id] = {"color": CATEGORICAL[index % len(CATEGORICAL)],
                             "dash": "solid" if index < len(CATEGORICAL) else "dot"}
    return styles


def _tree_positions(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    """Disposition hiérarchique : feuilles réparties, parents centrés au-dessus."""
    tree = nx.bfs_tree(graph, INTERNET_NODE)
    x_positions: dict[str, float] = {}
    next_leaf = [0.0]

    def place(node: str) -> float:
        children = list(tree.successors(node))
        if not children:
            x_positions[node] = next_leaf[0]
            next_leaf[0] += 1.0
        else:
            x_positions[node] = sum(place(child) for child in children) / len(children)
        return x_positions[node]

    place(INTERNET_NODE)
    return {node: (x_positions[node], -graph.nodes[node]["layer"]) for node in graph.nodes}


def topology_figure(graph: nx.Graph, statuses: pd.DataFrame, highlighted: str | None = None) -> go.Figure:
    positions = _tree_positions(graph)
    status_by_device = dict(zip(statuses["device_id"], statuses["status"])) if not statuses.empty else {}
    figure = go.Figure()

    for node_a, node_b, data in graph.edges(data=True):
        statuses_on_link = {status_by_device.get(node_a), status_by_device.get(node_b)}
        color = STATUS_COLORS[STATUS_DOWN] if STATUS_DOWN in statuses_on_link else "#b5b4ae"
        (xa, ya), (xb, yb) = positions[node_a], positions[node_b]
        figure.add_trace(go.Scatter(x=[xa, xb], y=[ya, yb], mode="lines",
                                    line=dict(color=color, width=2), hoverinfo="skip", showlegend=False))
        # point invisible au milieu du lien : affiche le débit au survol
        figure.add_trace(go.Scatter(x=[(xa + xb) / 2], y=[(ya + yb) / 2], mode="markers",
                                    marker=dict(size=14, color="rgba(0,0,0,0)"), showlegend=False,
                                    hovertext=f"Lien {node_a} ↔ {node_b} : {data['bandwidth_mbps']:g} Mbit/s",
                                    hoverinfo="text"))

    # un tracé par statut pour obtenir une légende explicite
    for status in (STATUS_OK, STATUS_DEGRADED, STATUS_DOWN, "Non supervisé"):
        nodes = [n for n in graph.nodes if (status_by_device.get(n, "Non supervisé") if graph.nodes[n].get("monitored") else "Non supervisé") == status]
        if not nodes:
            continue
        color = STATUS_COLORS.get(status, NEUTRAL)
        icon = STATUS_ICONS.get(status, "")
        hover = []
        for node in nodes:
            data = graph.nodes[node]
            row = statuses[statuses["device_id"] == node]
            details = ""
            if not row.empty:
                r = row.iloc[0]
                details = (f"<br>Latence : {r['last_latency_ms']} ms<br>Perte : {r['last_packet_loss_pct']} %"
                           f"<br>CPU : {r['last_cpu_pct']} %<br>Mémoire : {r['last_memory_pct']} %")
            hover.append(f"<b>{data['name']}</b> ({node})<br>{data['ip_address']} – {data['role']}<br>"
                         f"Statut : {icon} {status}{details}")
        figure.add_trace(go.Scatter(
            x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes],
            mode="markers+text", name=f"{icon} {status}".strip(),
            text=[n for n in nodes], textposition="bottom center",
            marker=dict(size=[30 if n == highlighted else 22 for n in nodes], color=color,
                        line=dict(color="#ffffff", width=2), symbol="circle"),
            hovertext=hover, hoverinfo="text"))

    figure.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=460,
                         xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-5.8, 0.5]),
                         title=_title("Topologie et état des équipements (survolez un nœud ou un lien)"))
    return figure


def metric_figure(detections: pd.DataFrame, metric: str, devices: list[str], styles: dict,
                  thresholds: RuleThresholds, incidents: pd.DataFrame | None = None) -> go.Figure:
    label, unit = FEATURE_LABELS[metric]
    figure = go.Figure()
    for device_id in devices:
        series = detections[detections["device_id"] == device_id]
        style = styles.get(device_id, {"color": NEUTRAL, "dash": "solid"})
        figure.add_trace(go.Scatter(x=series["timestamp"], y=series[metric], mode="lines", name=device_id,
                                    line=dict(color=style["color"], width=2, dash=style["dash"]),
                                    hovertemplate=f"{device_id} : %{{y:.2f}} {unit}<extra></extra>"))
        flagged = series[series["rule_anomaly"] | series["ml_anomaly"]]
        if not flagged.empty:
            figure.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged[metric], mode="markers",
                                        name=f"{device_id} – anomalie", showlegend=False,
                                        marker=dict(size=8, color=style["color"], symbol="x",
                                                    line=dict(width=1, color="#ffffff")),
                                        hoverinfo="skip"))

    limits = {rule[0]: rule[1:] for rule in thresholds.as_rules()}
    if metric in limits:
        warning, critical = limits[metric]
        figure.add_hline(y=warning, line=dict(color=STATUS_COLORS[STATUS_DEGRADED], dash="dot", width=1),
                         annotation_text=f"avertissement {warning:g} {unit}", annotation_position="top left")
        figure.add_hline(y=critical, line=dict(color=STATUS_COLORS[STATUS_DOWN], dash="dash", width=1),
                         annotation_text=f"critique {critical:g} {unit}", annotation_position="top left")

    if incidents is not None and not incidents.empty:
        for _, incident in incidents[incidents["device_id"].isin(devices)].iterrows():
            figure.add_vrect(x0=incident["start"], x1=incident["end"], fillcolor=SEVERITY_COLORS[incident["severity"]],
                             opacity=0.08, line_width=0)

    figure.update_layout(**LAYOUT, height=320, title=_title(f"{label.capitalize()} ({unit})"),
                         yaxis=dict(title=unit, rangemode="tozero", gridcolor="#e6e5e0"),
                         xaxis=dict(gridcolor="#f0efec"))
    return figure


def ml_score_figure(detections: pd.DataFrame, devices: list[str], styles: dict, threshold: float) -> go.Figure:
    figure = go.Figure()
    for device_id in devices:
        series = detections[detections["device_id"] == device_id]
        style = styles.get(device_id, {"color": NEUTRAL, "dash": "solid"})
        figure.add_trace(go.Scatter(x=series["timestamp"], y=series["ml_score"], mode="lines", name=device_id,
                                    line=dict(color=style["color"], width=2, dash=style["dash"]),
                                    hovertemplate=f"{device_id} : score %{{y:.3f}}<extra></extra>"))
    figure.add_hline(y=threshold, line=dict(color=STATUS_COLORS[STATUS_DOWN], dash="dash", width=1),
                     annotation_text="seuil de décision", annotation_position="top left")
    figure.update_layout(**LAYOUT, height=320, title=_title("Score d'anomalie Isolation Forest (plus haut = plus inhabituel)"),
                         yaxis=dict(gridcolor="#e6e5e0"))
    return figure


def agreement_figure(agreement: dict) -> go.Figure:
    labels = ["Règles et ML", "Règles seules", "ML seul"]
    values = [agreement["regles_et_ml"], agreement["regles_seules"], agreement["ml_seul"]]
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=CATEGORICAL[0],
                              text=values, textposition="outside", hovertemplate="%{y} : %{x} mesures<extra></extra>"))
    figure.update_layout(**{**LAYOUT, "hovermode": "closest", "margin": dict(l=10, r=40, t=50, b=10)}, height=240,
                         title=_title("Mesures signalées par chaque méthode"), xaxis=dict(title="mesures", gridcolor="#e6e5e0"),
                         yaxis=dict(autorange="reversed"))
    return figure
