"""Topologie du réseau d'entreprise simulé.

Le réseau est modélisé par un graphe NetworkX non orienté. La sonde de
supervision (NMS) est rattachée au commutateur cœur : toutes les mesures de
latence et de perte sont donc prises « depuis le cœur de réseau », ce qui
permet de propager un défaut aux équipements situés derrière l'équipement
fautif et de corréler les alertes (cause racine probable vs symptômes).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

INTERNET_NODE = "internet"
PROBE_ATTACHMENT = "sw-core-01"  # point d'attache de la sonde de supervision


@dataclass(frozen=True)
class Device:
    """Équipement supervisé."""

    device_id: str
    name: str
    role: str  # router | firewall | switch | server
    ip_address: str
    layer: int  # niveau hiérarchique (sert à l'affichage)
    criticality: int  # 1 = faible, 2 = moyenne, 3 = élevée
    description: str


DEFAULT_DEVICES: tuple[Device, ...] = (
    Device("rtr-edge-01", "Routeur de bordure", "router", "10.0.0.1", 1, 3,
           "Passerelle WAN vers le fournisseur d'accès"),
    Device("fw-01", "Pare-feu périmétrique", "firewall", "10.0.0.2", 2, 3,
           "Filtrage entre WAN, DMZ et LAN"),
    Device("sw-core-01", "Commutateur cœur", "switch", "10.0.1.1", 3, 3,
           "Cœur de réseau L3, interconnexion des VLAN"),
    Device("sw-dmz-01", "Commutateur DMZ", "switch", "10.0.2.1", 4, 2,
           "Raccordement des services exposés"),
    Device("sw-srv-01", "Commutateur serveurs", "switch", "10.0.10.1", 4, 3,
           "Raccordement des serveurs internes"),
    Device("sw-acc-01", "Commutateur d'accès", "switch", "10.0.20.1", 4, 2,
           "Raccordement des postes utilisateurs"),
    Device("srv-web-01", "Serveur web", "server", "10.0.2.10", 5, 2,
           "Site web public (DMZ)"),
    Device("srv-app-01", "Serveur applicatif", "server", "10.0.10.20", 5, 3,
           "Application métier interne"),
    Device("srv-db-01", "Serveur base de données", "server", "10.0.10.30", 5, 3,
           "Base de données de l'application métier"),
    Device("srv-dns-01", "Serveur DNS", "server", "10.0.10.53", 5, 3,
           "Résolution de noms interne"),
)

DEFAULT_LINKS: tuple[tuple[str, str, int], ...] = (
    # (extrémité A, extrémité B, débit nominal en Mbit/s)
    (INTERNET_NODE, "rtr-edge-01", 500),
    ("rtr-edge-01", "fw-01", 1000),
    ("fw-01", "sw-core-01", 10000),
    ("sw-core-01", "sw-dmz-01", 1000),
    ("sw-core-01", "sw-srv-01", 10000),
    ("sw-core-01", "sw-acc-01", 1000),
    ("sw-dmz-01", "srv-web-01", 1000),
    ("sw-srv-01", "srv-app-01", 10000),
    ("sw-srv-01", "srv-db-01", 10000),
    ("sw-srv-01", "srv-dns-01", 1000),
)


def build_default_topology() -> nx.Graph:
    """Construit le graphe de la topologie par défaut."""
    graph = nx.Graph()
    graph.add_node(INTERNET_NODE, name="Internet (FAI)", role="internet",
                   ip_address="-", layer=0, criticality=0,
                   description="Réseau externe (non supervisé)", monitored=False)
    for device in DEFAULT_DEVICES:
        graph.add_node(device.device_id, name=device.name, role=device.role,
                       ip_address=device.ip_address, layer=device.layer,
                       criticality=device.criticality,
                       description=device.description, monitored=True)
    for node_a, node_b, bandwidth in DEFAULT_LINKS:
        graph.add_edge(node_a, node_b, bandwidth_mbps=bandwidth)
    return graph


def monitored_devices(graph: nx.Graph) -> list[str]:
    """Identifiants des équipements supervisés, dans l'ordre d'insertion."""
    return [node for node, data in graph.nodes(data=True) if data.get("monitored")]


def probe_path(graph: nx.Graph, device_id: str) -> list[str]:
    """Chemin parcouru par les sondes depuis le cœur jusqu'à l'équipement (inclus)."""
    if device_id not in graph:
        raise KeyError(f"Équipement inconnu dans la topologie : {device_id}")
    return nx.shortest_path(graph, PROBE_ATTACHMENT, device_id)


def downstream_devices(graph: nx.Graph, device_id: str) -> list[str]:
    """Équipements supervisés joints par la sonde en traversant ``device_id``.

    Ce sont ceux dont les mesures sont affectées si ``device_id`` dégrade le
    trafic (latence, pertes) ou tombe en panne.
    """
    impacted = []
    for other in monitored_devices(graph):
        if other == device_id:
            continue
        if device_id in probe_path(graph, other)[:-1]:
            impacted.append(other)
    return impacted


def is_on_internet_path(graph: nx.Graph, device_id: str) -> bool:
    """Vrai si l'équipement se trouve sur le chemin entre le LAN et Internet."""
    if device_id not in graph:
        return False
    return device_id in nx.shortest_path(graph, PROBE_ATTACHMENT, INTERNET_NODE)


def device_table(graph: nx.Graph) -> list[dict]:
    """Inventaire des équipements supervisés (pour affichage ou export)."""
    rows = []
    for node in monitored_devices(graph):
        data = graph.nodes[node]
        rows.append({
            "device_id": node,
            "device_name": data["name"],
            "role": data["role"],
            "ip_address": data["ip_address"],
            "criticality": data["criticality"],
            "description": data["description"],
        })
    return rows
