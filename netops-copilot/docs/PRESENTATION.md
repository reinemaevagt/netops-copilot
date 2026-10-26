# NetOps Copilot — présentation du projet

## Valeur pour les métiers visés

| Métier | Ce que le projet démontre |
|---|---|
| **Supervision réseau / NOC** | Compréhension des indicateurs de base (latence, perte, disponibilité, CPU, mémoire), des seuils d'alerte, de l'anti-rebond et de la priorisation des incidents (P1–P4, état de traitement). |
| **NetOps / automatisation** | Chaîne de traitement automatisée en Python, code modulaire et testé, format de données normalisé prêt à recevoir des sources réelles (SNMP, syslog, API). |
| **Administration des réseaux IP** | Modélisation d'une architecture d'entreprise (WAN, pare-feu, cœur, DMZ, accès, serveurs), raisonnement sur les chemins et les dépendances, recommandations de dépannage concrètes (compteurs d'interface, duplex, traceroute, OOB, VRRP/HSRP). |
| **Cybersécurité réseau** | Détection d'anomalies non supervisée, gestion des faux positifs, distinction entre signal faible et incident avéré : les mêmes questions se posent en détection d'intrusion. |
| **Gestion de projet télécom** | Projet mené de bout en bout : cahier des charges, architecture, tests, documentation, limites assumées et feuille de route v2. |
| **Data appliquée aux infrastructures** | Séries temporelles multi-équipements, normalisation robuste, évaluation précision / rappel, comparaison objective de deux méthodes. |

Point différenciant : la **corrélation topologique**. Quand un commutateur perd des paquets, NetOps Copilot n'affiche
pas quatre incidents indépendants mais une cause racine probable et trois symptômes, ce qui correspond à la manière
dont un NOC réduit le bruit d'alertes.

---

## Description courte pour GitHub

**About (une ligne) :**
> Supervision intelligente d'un réseau d'entreprise simulé : détection d'anomalies (règles + Isolation Forest), corrélation d'incidents et aide au diagnostic — Python, Streamlit, scikit-learn, NetworkX.

**Topics suggérés :** `netops` `network-monitoring` `anomaly-detection` `isolation-forest` `streamlit` `networkx` `python` `telecom`

---

## Pour le CV (rubrique Projets)

**NetOps Copilot — supervision réseau et détection d'anomalies** *(projet personnel, 2026 · Python)*
- Simulation d'un réseau d'entreprise de 10 équipements (routeur, pare-feu, commutateurs, serveurs) et de 4 scénarios de panne, avec propagation des défauts le long de la topologie.
- Détection d'anomalies par règles à seuils et par Isolation Forest (scikit-learn), avec évaluation comparative (précision, rappel, faux positifs).
- Regroupement des alertes en incidents priorisés, identification de la cause racine probable par corrélation topologique (NetworkX), hypothèses et recommandations de dépannage.
- Tableau de bord Streamlit / Plotly, export CSV, 44 tests automatisés (pytest).

Version courte (une ligne) :
> NetOps Copilot : outil Python de supervision d'un réseau simulé — détection d'anomalies (règles + ML), corrélation d'incidents, tableau de bord Streamlit, 44 tests pytest.

---

## Publication LinkedIn (proposition)

> 📡 **Projet personnel : NetOps Copilot**
>
> Dans un centre de supervision réseau, le plus difficile n'est pas de recevoir des alertes, mais de savoir lesquelles comptent et par où commencer. J'ai voulu explorer ce problème avec un projet en Python.
>
> NetOps Copilot simule le réseau d'une entreprise (routeur, pare-feu, commutateurs, serveurs) et :
> 🔹 génère des métriques réalistes (latence, perte de paquets, CPU, mémoire, disponibilité) pour 5 scénarios : fonctionnement normal, latence élevée, perte de paquets, routeur indisponible, serveur surchargé ;
> 🔹 détecte les anomalies de deux façons : des règles à seuils et un modèle Isolation Forest (scikit-learn) ;
> 🔹 regroupe les alertes en incidents priorisés et utilise la topologie pour proposer une cause racine probable : si un commutateur perd des paquets, les serveurs derrière lui sont signalés comme symptômes, pas comme 4 pannes distinctes ;
> 🔹 affiche le tout dans un tableau de bord Streamlit.
>
> Ce que j'en retiens :
> ✅ règles et machine learning sont complémentaires : le modèle repère bien les combinaisons inhabituelles de métriques, mais il manque une partie des anomalies portant sur une seule métrique, et il ne peut rien dire d'un équipement qui ne répond plus ;
> ✅ sans mécanisme anti-rebond, les pics ponctuels créeraient des fausses alertes en continu ;
> ✅ un diagnostic honnête formule des hypothèses, pas des certitudes.
>
> ⚠️ Les données sont entièrement simulées : c'est un projet d'apprentissage, pas un outil de production. Prochaine étape envisagée : brancher une vraie source SNMP sur un lab virtuel.
>
> Code, documentation et tests (pytest) sur GitHub : [lien]
>
> Je recherche un stage ingénieur de 6 mois à partir de février 2027 en réseaux, télécoms, supervision ou NetOps. N'hésitez pas à me contacter !
>
> #Réseaux #Télécoms #NetOps #Python #MachineLearning #Supervision #Stage

Conseil : joindre une courte vidéo (30–60 s) ou 2–3 captures d'écran : la topologie avec un équipement en rouge,
puis la fiche d'un incident avec sa cause racine probable.

---

## Améliorations envisagées (version 2)

1. Collecte SNMP réelle en lecture seule sur un lab virtuel (GNS3, EVE-NG ou Containerlab) via une nouvelle `MetricsSource`.
2. Ingestion de journaux syslog et corrélation logs ↔ métriques.
3. Seuils dynamiques appris par équipement et par tranche horaire.
4. Prévision de saturation (CPU, liens) sur séries temporelles.
5. Persistance des incidents (SQLite), historique des actions et rapport d'incident exportable.
6. Notifications (e-mail, webhook Teams/Slack).
7. Topologie décrite dans un fichier YAML, liens redondants, chemins multiples.
8. Dockerfile et intégration continue GitHub Actions exécutant les tests à chaque commit.
