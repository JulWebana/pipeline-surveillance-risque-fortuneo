# Pipeline de Surveillance du Risque Boursier — Fortuneo

## Contexte et problématique métier

Fortuneo, filiale de Crédit Mutuel Arkéa, est l'une des premières banques en ligne françaises. En tant que courtier en ligne, Fortuneo met à disposition de ses clients des milliers de valeurs mobilières cotées sur les marchés financiers européens et internationaux.

**Le problème adressé par ce projet** : le Data Lab de Fortuneo doit surveiller en continu les cours des actions les plus échangées par ses clients pour détecter des comportements anormaux de marché (flash crashes, pics de volatilité inhabituels, manipulations de cours) susceptibles d'impacter négativement les portefeuilles clients. Sans détection automatisée, ces événements ne seraient identifiés qu'après coup laissant les clients exposés.

> **Disclaimer** : la problématique décrite ci-dessus est fictive et issue de mon imagination. Ce projet est un exercice personnel. Il n'est en aucun cas affilié à Fortuneo, ne représente pas les pratiques internes de l'entreprise et n'utilise aucune donnée confidentielle. Les données boursières proviennent exclusivement de l'API publique Yahoo Finance.

Ce pipeline répond directement à cette problématique en automatisant l'ingestion, l'enrichissement et l'analyse des données boursières du CAC40 puis en appliquant un modèle de Machine Learning non supervisé pour identifier les séances anormales en quasi-temps réel.

---

## Architecture du pipeline ETL

```
┌─────────────────────────────────────────────────────────────────┐
│               PIPELINE ETL — FORTUNEO DATA LAB                  │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│   EXTRACT    │  TRANSFORM   │    DETECT    │       LOAD         │
│              │              │              │                    │
│ Yahoo Finance│ Indicateurs  │ Isolation    │ SQLite (RDS)       │
│ API          │ techniques   │ Forest ML    │ + Parquet (S3)     │
│              │              │              │                    │
│ • OHLCV      │ • MA 20/50j  │ • Scoring    │ • Table SQL        │
│ • 12 actions │ • RSI 14j    │ • Alertes    │ • Fichiers .parquet│
│ • 1 an hist. │ • Bollinger  │ • Rapport    │ • Logs JSON        │
│              │ • Volatilité │   par titre  │                    │
│              │ • VaR 99%    │              │                    │
└──────────────┴──────────────┴──────────────┴────────────────────┘
         ↕                                          ↕
   [AWS S3 Bronze]                         [AWS S3 Silver]
   [Simulation locale]               [Simulation locale Parquet]
```

### Couche Extract — Ingestion des données
Récupération des données historiques OHLCV (Open, High, Low, Close, Volume) des 12 principales actions du CAC40 disponibles sur la plateforme Fortuneo via l'API Yahoo Finance. 

### Couche Transform — Enrichissement
Calcul des indicateurs techniques de référence pour l'analyse financière : moyennes mobiles (MA 20 et 50 jours), RSI 14 jours, Bandes de Bollinger, volatilité historique annualisée, VaR historique à 99% et amplitude journalière normalisée.

### Couche ML — Détection d'anomalies
Application de l'algorithme Isolation Forest de scikit-learn pour détecter les séances boursières présentant un comportement statistiquement anormal. Les expériences sont tracées dans MLflow.

### Couche Load — Persistance
Chargement des données transformées et des prédictions en base de données SQLite (simulant AWS RDS) et au format Apache Parquet (simulant AWS S3) avec partitionnement par symbole pour optimiser les requêtes analytiques.

---

## Structure du projet

```
pipeline-surveillance-risque-fortuneo/
│
├── config/
│   └── config.yaml               # Configuration centralisée du pipeline
│
├── src/
│   ├── extraction/
│   │   └── extracteur_donnees.py # Couche Extract : API Yahoo Finance
│   ├── transformation/
│   │   └── transformateur_donnees.py  # Couche Transform : indicateurs techniques
│   ├── modeles/
│   │   └── detecteur_anomalies.py     # Modèle ML : Isolation Forest + MLflow
│   ├── chargement/
│   │   └── chargeur_donnees.py        # Couche Load : SQLite + Parquet
│   ├── monitoring/
│   │   └── moniteur_pipeline.py       # Monitoring : logs, métriques, alertes
│   └── pipeline.py                    # Orchestrateur principal du pipeline
│
├── tests/
│   ├── test_extraction.py             # Tests unitaires de la couche Extract
│   ├── test_transformation.py         # Tests unitaires de la couche Transform
│   └── test_modele.py                 # Tests unitaires du modèle ML
│
├── data/
│   ├── brutes/                        # Données brutes extraites (couche Bronze)
│   └── transformees/                  # Données enrichies en Parquet (couche Silver)
│
├── logs/                              # Fichiers de log et rapports JSON des runs
├── mlruns/                            # Artefacts MLflow (modèles, métriques)
├── requirements.txt                   # Dépendances Python du projet
└── README.md                          # Documentation du projet
```

---

## Technologies utilisées

| Technologie | Usage dans le projet | Équivalent production (Fortuneo) |
|---|---|---|
| Python 3.10+ | Langage principal du pipeline | Python 3.10+ |
| yfinance | Récupération des données boursières | Flux Reuters / Bloomberg |
| pandas / numpy | Manipulation et calcul des données | idem |
| scikit-learn | Modèle de détection d'anomalies (Isolation Forest) | idem |
| SQLAlchemy + SQLite | Persistance des données | AWS RDS (PostgreSQL) |
| Apache Parquet + pyarrow | Stockage colonnaire optimisé | AWS S3 + Athena |
| MLflow | Suivi des expériences MLOps | MLflow sur AWS EC2 / SageMaker |
| pytest | Tests unitaires et d'intégration | idem |
| PyYAML | Gestion de la configuration | idem |
| logging | Journalisation du pipeline | AWS CloudWatch |

---

## Installation et lancement

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Connexion internet (pour l'accès à l'API Yahoo Finance)

### Installation des dépendances

```bash
# Clonage du dépôt (si vous etes intéréssé)
git clone https://github.com/<votre-compte>/pipeline-surveillance-risque-fortuneo.git
cd pipeline-surveillance-risque-fortuneo

# Installation des dépendances Python
pip install -r requirements.txt
```

### Exécution du pipeline complet

```bash
# Lancement avec la configuration par défaut
python src/pipeline.py

# Lancement avec un fichier de configuration personnalisé
python src/pipeline.py --config config/config.yaml
```

### Exécution des tests unitaires

```bash
# Lancement de tous les tests avec rapport de couverture
pytest tests/ -v --cov=src --cov-report=term-missing

# Lancement d'un module de tests spécifique
pytest tests/test_transformation.py -v
```

### Consultation des expériences MLflow

```bash
# Démarrage de l'interface web MLflow
mlflow ui --backend-store-uri mlruns

# Ouvrir dans le navigateur : http://localhost:5000
```

---

## Indicateurs techniques calculés

**Moyennes Mobiles Simples (MA)** : lissent les fluctuations de cours pour révéler la tendance sous-jacente d'un titre. La MA 20 jours suit la tendance courte, la MA 50 jours la tendance longue. Un croisement haussier (Golden Cross) est un signal d'achat classique.

**RSI — Relative Strength Index** : oscillateur de momentum mesurant la vitesse et l'amplitude des variations de prix. Un RSI supérieur à 70 signale une zone de surachat, inférieur à 30 une zone de survente.

**Bandes de Bollinger** : encadrent le prix entre une bande haute et une bande basse, définies à + ou -2 écarts-types autour d'une moyenne mobile de 20 jours. Un cours sortant des bandes indique une volatilité anormalement élevée.

**Volatilité historique annualisée** : écart-type des rendements logarithmiques journaliers, multiplié par √252 pour l'annualisation. C'est la mesure de risque de référence en gestion de portefeuille.

**VaR historique à 99%** : Value at Risk, indicateur réglementaire (Bâle III) représentant la perte maximale journalière dépassée dans seulement 1% des cas sur la fenêtre d'observation.

---

## Algorithme de détection d'anomalies — Isolation Forest

L'Isolation Forest est un algorithme non supervisé particulièrement adapté à la détection d'anomalies dans des données financières à haute dimensionnalité. Son principe repose sur le fait que les anomalies étant rares et différentes des observations normales sont plus faciles à "isoler" en peu de divisions dans un arbre de décision aléatoire.

**Paramètres configurables dans `config/config.yaml` :**
- `contamination` : proportion attendue d'anomalies (défaut : 5%)
- `n_estimateurs` : nombre d'arbres dans la forêt (défaut : 100)
- `graine_aleatoire` : graine de reproductibilité (défaut : 42)

---

## Pratiques MLOps implémentées

**Traçabilité des expériences** : chaque entraînement du modèle est enregistré dans MLflow avec ses hyperparamètres, ses métriques et ses artefacts (modèle sérialisé, scaler).

**Versioning du modèle** : le modèle entraîné est sauvegardé avec joblib et peut être rechargé lors des exécutions suivantes évitant un ré-entraînement inutile.

**Monitoring du pipeline** : chaque exécution génère un rapport JSON horodaté contenant les métriques de performance de chaque étape (durée, nombre de lignes, taux de complétude, alertes).

**Configuration externalisée** : tous les hyperparamètres et paramètres du pipeline sont centralisés dans un fichier YAML conformément au principe de séparation code/configuration.

**Tests unitaires** : chaque module dispose de tests unitaires avec pytest couvrant les cas nominaux et les cas limites.

---

## Résultats et sorties du pipeline

Après une exécution complète, le pipeline produit les fichiers suivants :

- `data/brutes/extraction_brute_YYYYMMDD_HHMMSS.csv` : données brutes extraites
- `data/transformees/<symbole>.parquet` : données enrichies par symbole
- `data/fortuneo_pipeline.db` : base de données SQLite avec toutes les tables
- `logs/rapport_pipeline_<run_id>.json` : rapport JSON du monitoring du run
- `logs/pipeline.log` : fichier de log complet du pipeline
- `mlruns/` : artefacts MLflow (modèle, scaler, métriques d'entraînement)

---

## Propositions d' améliorations pour la mise en production

- Remplacer SQLite par une instance AWS RDS PostgreSQL Multi-AZ
- Remplacer les fichiers locaux par un bucket AWS S3 avec AWS Glue Catalog
- Orchestrer le pipeline avec Apache Airflow ou AWS Step Functions
- Ajouter un système d'alertes via AWS SNS (email/SMS) lors de la détection d'anomalies
- Mettre en place une surveillance de la dérive des données (data drift) avec Evidently
- Containeriser le pipeline avec Docker et le déployer sur AWS ECS ou EKS
- Intégrer un pipeline CI/CD avec GitHub Actions pour automatiser les tests

---

## Auteur

Projet réalisé par Julien AGA - Étudiant Data Engineer / Cloud / MLOps
