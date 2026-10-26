# 📡 NetOps Copilot

**Supervision intelligente d'un réseau d'entreprise simulé : détection d'anomalies (règles + Isolation Forest), corrélation d'incidents et aide au diagnostic.**

> ⚠️ **Projet pédagogique.** Toutes les mesures sont **simulées**. NetOps Copilot n'est pas un outil de production
> et n'a pas été validé sur un réseau réel. Les causes affichées sont des **hypothèses** destinées à orienter l'investigation.

---

## 1. Problème et objectifs

Une équipe d'exploitation réseau (NOC / NetOps) reçoit des milliers de mesures par heure. Les difficultés classiques sont :

- **repérer** rapidement ce qui sort de l'ordinaire sans être noyé sous les fausses alertes ;
- **distinguer la cause des symptômes** : une panne en amont déclenche des alertes sur tous les équipements situés derrière ;
- **prioriser** les incidents et savoir **par où commencer** l'investigation.

NetOps Copilot reproduit cette chaîne sur un réseau d'entreprise simulé de 10 équipements, entièrement sur un poste
Windows, sans matériel, sans cloud et sans clé d'API.

## 2. Fonctionnalités

| Domaine | Ce qui est implémenté |
|---|---|
| **Infrastructure simulée** | Routeur de bordure, pare-feu, commutateurs (cœur, DMZ, serveurs, accès) et 4 serveurs. Topologie NetworkX affichée de façon interactive, statut opérationnel / dégradé / indisponible, et **rejeu de l'état du réseau à n'importe quel instant**. |
| **Métriques** | Latence, perte de paquets, CPU, mémoire, disponibilité, horodatage. Mesures prises depuis une sonde sur le cœur de réseau : un défaut **se propage** aux équipements en aval. Données reproductibles (graine), trous de collecte et pics bénins simulés. |
| **Scénarios** | Fonctionnement normal · Latence élevée · Perte de paquets · Routeur indisponible · Serveur surchargé. |
| **Détection** | Règles à seuils (avertissement / critique) **et** Isolation Forest (scikit-learn) entraîné sur un historique sans défaut, normalisé par équipement. Chaque alerte est **expliquée** (valeur, seuil, écart à la normale). |
| **Comparaison règles / ML** | Concordance des deux méthodes, et — grâce à la vérité terrain de la simulation — précision, rappel, F1, faux positifs. |
| **Incidents** | Regroupement des anomalies, anti-rebond (les pics isolés n'ouvrent pas d'incident), gravité, priorité P1–P4, état de traitement modifiable, **corrélation topologique (cause racine probable vs symptômes)**, impact, preuves chiffrées, hypothèses et recommandations. |
| **Tableau de bord** | KPI, topologie, graphiques avec seuils et périodes d'incident, filtres (rôle, équipement, gravité), section méthodes & limites, export CSV, import CSV. |

## 3. Architecture

```
netops-copilot/
├── app.py                      # Interface Streamlit (affichage uniquement)
├── requirements.txt
├── pytest.ini
├── README.md · LICENSE · .gitignore
├── .streamlit/config.toml      # Thème de l'interface
├── docs/
│   └── PRESENTATION.md         # Valeur métier, textes GitHub / CV / LinkedIn, pistes v2
├── src/netops_copilot/
│   ├── topology.py             # Équipements, liens, chemins de sonde, équipements en aval
│   ├── simulation.py           # Scénarios et génération des métriques simulées
│   ├── data_sources.py         # Interface MetricsSource, source simulée, source CSV, validation
│   ├── detection.py            # Règles à seuils, Isolation Forest, comparaison des méthodes
│   ├── diagnosis.py            # Incidents, corrélation, hypothèses et recommandations
│   ├── kpi.py                  # KPI globaux, statut des équipements
│   ├── pipeline.py             # Chaîne complète : source → détection → incidents → KPI
│   └── charts.py               # Figures Plotly (topologie, séries temporelles, score ML)
└── tests/                      # 44 tests pytest
```

```
MetricsSource (simulation | CSV | … futur SNMP/API)
        │  DataFrame au format standard (validé)
        ▼
detect_with_rules ──► IsolationForestDetector.score ──► build_incidents ──► device_status / compute_kpis
                                                             │
                                                     topologie (corrélation, impact)
```

La logique métier ne dépend d'aucun composant graphique : `pipeline.run_analysis()` est utilisé à la fois par
l'interface et par les tests.

### Rôle de chaque module

- **`topology.py`** — décrit les équipements (rôle, IP, criticité) et les liens (débit). Calcule le chemin parcouru par
  la sonde jusqu'à chaque équipement et la liste des équipements situés « derrière » un équipement donné.
- **`simulation.py`** — génère une ligne par (instant, équipement). Chaque équipement a un comportement propre (bruit,
  cycle de charge). La latence observée est la somme des contributions des équipements traversés ; la perte combine
  les pertes de chaque saut ; un équipement injoignable n'a plus de mesures. La colonne `ground_truth_fault` indique
  les mesures réellement affectées par le défaut injecté (utile seulement pour évaluer la détection).
- **`data_sources.py`** — contrat d'échange des données : colonnes obligatoires, types, bornes physiques, équipements
  connus. Toute nouvelle source (SNMP, syslog, API) n'a qu'à implémenter `MetricsSource.load()`.
- **`detection.py`** — règles lisibles et Isolation Forest. Le modèle apprend la normalité sur un historique simulé
  sans défaut ; le seuil de décision est calé sur un taux de faux positifs accepté (1 % par défaut). Les mesures
  incomplètes ne sont pas évaluées par le ML (signalé explicitement).
- **`diagnosis.py`** — regroupe les anomalies en épisodes, n'ouvre un incident que s'il persiste, attribue catégorie,
  gravité et priorité, recherche une cause racine probable en amont, puis produit preuves, hypothèses et actions.
- **`kpi.py`** — disponibilité, latence moyenne et P95, perte, statut des équipements (y compris à un instant passé).
- **`pipeline.py`** — orchestre l'ensemble et retourne un objet `AnalysisResult`.
- **`charts.py`** — figures Plotly ; chaque équipement garde une couleur fixe, les couleurs de statut sont réservées
  aux états et toujours accompagnées d'un libellé.

## 4. Technologies

| Outil | Pourquoi |
|---|---|
| **Python** | Langage de référence pour l'automatisation réseau (Netmiko, NAPALM, Nornir…) et la data. |
| **Streamlit** | Tableau de bord interactif en pur Python, idéal pour une démonstration. |
| **Pandas / NumPy** | Manipulation de séries temporelles multi-équipements. |
| **Plotly** | Graphiques interactifs (survol, zoom) et topologie. |
| **NetworkX** | Modèle de graphe : chemins, dépendances, corrélation des alertes. |
| **scikit-learn** | Isolation Forest : détection non supervisée, sans données étiquetées. |
| **pytest** | Tests automatisés des scénarios, KPI, règles, ML et diagnostic. |

## 5. Installation sous Windows

Prérequis : **Python 3.10 ou plus récent** (cocher « Add python.exe to PATH » à l'installation) et Git (facultatif).

```powershell
# 1. Récupérer le projet (ou décompresser l'archive ZIP)
cd C:\Users\<vous>\Documents
git clone https://github.com/<votre-compte>/netops-copilot.git
cd netops-copilot

# 2. Créer et activer un environnement virtuel
py -m venv .venv
.\.venv\Scripts\Activate.ps1
#   Si PowerShell refuse l'exécution du script :
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   (puis relancer la commande d'activation)
#   Sous cmd.exe : .venv\Scripts\activate.bat

# 3. Installer les dépendances
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Lancer l'application

```powershell
streamlit run app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`. Pour arrêter : `Ctrl + C` dans le terminal.

Depuis VS Code : ouvrir le dossier, sélectionner l'interpréteur `.venv` (`Ctrl+Maj+P` → *Python: Select Interpreter*),
puis lancer la commande ci-dessus dans le terminal intégré.

## 7. Lancer les tests

```powershell
pytest            # résumé
pytest -v         # détail de chaque test
```

Résultat obtenu lors du développement (Python 3.13, pandas 3.0, scikit-learn 1.9, NumPy 2.5, pytest 8.3) :
**44 tests réussis en ~8 s**.

## 8. Scénarios de démonstration

Valeurs obtenues avec la graine 42, 6 h simulées, une mesure par minute.

| Scénario | Ce que montre l'application |
|---|---|
| **Fonctionnement normal** | Aucun incident. Quelques mesures isolées signalées par le ML (taux de faux positifs accepté) et par les règles (pics CPU bénins) : **l'anti-rebond empêche qu'elles deviennent des incidents**. |
| **Latence élevée** | Incident P1 sur le pare-feu ; le routeur, joint à travers lui, présente la même latence et est rattaché au pare-feu comme **cause racine probable** (P2). Le ML détecte bien le pare-feu (latence + CPU) mais moins le routeur (latence seule) : bonne illustration de la complémentarité règles / ML. |
| **Perte de paquets** | Incident P1 sur le commutateur serveurs ; les 3 serveurs derrière lui sont identifiés comme symptômes. Hypothèses : câble/SFP, erreurs CRC, duplex. |
| **Routeur indisponible** | Incident critique P1, impact « accès Internet ». Le ML ne peut pas évaluer un équipement sans mesures : seule la règle de disponibilité le détecte. Utiliser le curseur de rejeu pour voir le routeur en rouge pendant la panne. |
| **Serveur surchargé** | Incident P1 *Ouvert* (toujours en cours en fin de période), CPU ~95 % et mémoire ~90 %. Règles et ML concordent. |

Évaluation au niveau des mesures (vérité terrain de la simulation, graine 42) :

| Scénario | Règles (précision / rappel) | Isolation Forest (précision / rappel) |
|---|---|---|
| Latence élevée | 0,99 / 1,00 | 0,75 / 0,56 |
| Perte de paquets | 1,00 / 1,00 | 0,95 / 0,84 |
| Routeur indisponible | 0,96 / 1,00 | non applicable (aucune mesure) |
| Serveur surchargé | 0,95 / 0,99 | 0,85 / 1,00 |

Ces chiffres décrivent le comportement **sur des données simulées construites pour ce projet** : ils ne prédisent pas
les performances sur un réseau réel. Les règles sont avantagées car les défauts simulés franchissent nettement les seuils.

### Importer des données
Onglet **Export** → télécharger les métriques → barre latérale **Importer un CSV** → réimporter le fichier.
Tout CSV respectant le format (`timestamp, device_id, latency_ms, packet_loss_pct, cpu_pct, memory_pct, available`)
et les identifiants d'équipements de la topologie peut être analysé.

## 9. Limites

- Données **simulées** : bruit gaussien, cycles de charge et défauts simplifiés ; pas de trafic réel, pas de protocole réseau émulé.
- Propagation simplifiée : la sonde est unique (sur le cœur), sans redondance de chemins ni routage dynamique.
- Seuils génériques, identiques pour tous les équipements d'un même type.
- L'Isolation Forest est moins sensible aux anomalies portant sur une **seule** métrique, et ne traite pas les mesures incomplètes.
- Les explications du ML indiquent les métriques les plus éloignées de la normale : ce n'est pas une attribution exacte de la décision du modèle.
- Base d'hypothèses et de recommandations rédigée à la main, non exhaustive.
- L'état de traitement des incidents n'est conservé que pendant la session (pas de base de données).

## 10. Pistes d'amélioration

1. Source **SNMP** réelle (lecture seule, sur un équipement de lab ou un routeur virtuel GNS3 / Containerlab) via une classe `SnmpSource(MetricsSource)`.
2. Import de journaux **syslog** et corrélation logs ↔ métriques.
3. Seuils dynamiques (baselines par équipement et par heure de la journée).
4. Détection de tendances (saturation prévisible du CPU ou d'un lien) par prévision de séries temporelles.
5. Persistance des incidents (SQLite) et historique des actions.
6. Notifications (e-mail, Teams/Slack) et export vers un outil de ticketing.
7. Topologie chargée depuis un fichier YAML, liens redondants et chemins multiples.
8. Conteneurisation (Dockerfile) et intégration continue (GitHub Actions lançant `pytest`).

Voir aussi [`docs/PRESENTATION.md`](docs/PRESENTATION.md) pour la valeur métier du projet.

## Licence

MIT — voir [LICENSE](LICENSE).
