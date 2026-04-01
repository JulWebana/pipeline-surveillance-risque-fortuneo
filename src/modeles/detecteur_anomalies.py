# =============================================================================
# Module de détection d'anomalies boursières - Couche Machine Learning
# =============================================================================
# Ce module implémente la couche Machine Learning du pipeline Fortuneo.
# Il utilise l'algorithme Isolation Forest pour détecter automatiquement
# les comportements anormaux sur les marchés financiers.
#
# Problématique métier : parmi les milliers de séances boursières quotidiennes,
# certaines présentent des caractéristiques inhabituelles (flash crash, manipulation
# de cours, choc de liquidité) qui peuvent nuire aux portefeuilles des clients.
# La détection automatique de ces anomalies permet à l'équipe de gestion du risque
# de Fortuneo d'intervenir rapidement pour protéger les investisseurs.
#
# Approche MLOps : le modèle est entraîné, évalué et sauvegardé avec MLflow
# pour assurer la traçabilité des expériences et le versioning des modèles.
# Cette approche respecte les pratiques MLOps exigées par l'offre Fortuneo.
# =============================================================================

# Importation des bibliothèques standard Python
import logging   # Gestion de la journalisation des événements
import os        # Opérations sur le système de fichiers
from datetime import datetime   # Manipulation des dates et horodatages
from typing import List, Optional, Tuple   # Annotations de types pour la lisibilité

# Importation des bibliothèques de Machine Learning
import joblib                              # Sérialisation des modèles ML entraînés
import numpy as np                         # Calculs numériques et tableaux multi-dimensionnels
import pandas as pd                        # Manipulation des DataFrames
from sklearn.ensemble import IsolationForest   # Algorithme principal de détection d'anomalies
from sklearn.preprocessing import StandardScaler   # Normalisation des features ML

# Importation de MLflow pour le suivi des expériences (MLOps)
import mlflow                     # Plateforme de traçabilité des expériences ML
import mlflow.sklearn             # Module d'intégration MLflow avec scikit-learn

# Récupération du logger configuré au niveau du pipeline principal
logger = logging.getLogger(__name__)


# =============================================================================
# Liste des features (variables explicatives) utilisées par le modèle ML
# =============================================================================
# Ces colonnes ont été choisies pour leur pertinence dans la caractérisation
# du comportement boursier : elles couvrent le rendement, la volatilité,
# la tendance et le momentum d'une action sur différents horizons temporels.
FEATURES_MODELE: List[str] = [
    "rendement_log",           # Rendement logarithmique journalier (performance du jour)
    "volatilite_annualisee",   # Volatilité historique annualisée (risque structurel)
    "rsi",                     # RSI (indicateur de momentum de surachat/survente)
    "amplitude_journaliere",   # Amplitude haute-basse normalisée (volatilité intra-jour)
    "var_historique_99",       # VaR historique à 99% (perte potentielle maximale)
    "bollinger_haute",         # Bande haute de Bollinger (résistance haute)
    "bollinger_basse",         # Bande basse de Bollinger (support bas)
    "ma_courte",               # Moyenne mobile court terme (tendance récente)
    "ma_longue",               # Moyenne mobile long terme (tendance de fond)
    "volume",                  # Volume de transactions (liquidité du marché)
]


class DetecteurAnomaliesBoursières:
    """
    Classe implémentant la détection d'anomalies boursières par Isolation Forest.

    L'Isolation Forest est un algorithme non supervisé particulièrement adapté
    à la détection d'anomalies dans des données financières à haute dimensionnalité.
    Son principe : les anomalies sont rares et différentes des observations normales,
    donc plus faciles à "isoler" dans un arbre de décision aléatoire.

    Avantages pour Fortuneo :
    - Ne nécessite pas de données étiquetées (anomalies non labellisées)
    - Efficace sur des données à grande dimension (nombreux indicateurs)
    - Temps d'inférence très rapide pour une détection en quasi-temps réel
    - Interprétable : chaque feature contribue au score d'anomalie

    Attributs :
        contamination (float): Proportion estimée d'anomalies dans les données.
        n_estimateurs (int): Nombre d'arbres dans la forêt d'isolation.
        graine_aleatoire (int): Graine pour la reproductibilité des résultats.
        features (List[str]): Liste des colonnes utilisées comme features.
        modele (IsolationForest): Instance du modèle Isolation Forest.
        scaler (StandardScaler): Normaliseur des features.
        est_entraine (bool): Indique si le modèle a été entraîné.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimateurs: int = 100,
        graine_aleatoire: int = 42,
        features: Optional[List[str]] = None,
        chemin_modele: str = "mlruns/modele_detection_anomalies.pkl",
        chemin_scaler: str = "mlruns/scaler_features.pkl",
    ) -> None:
        """
        Initialise le détecteur d'anomalies avec les hyperparamètres du modèle.

        Args:
            contamination: Taux d'anomalies attendu dans les données (défaut : 5%).
                          Ce paramètre est crucial : une valeur trop élevée génère
                          trop de faux positifs, trop faible manque des anomalies réelles.
            n_estimateurs: Nombre d'arbres de décision dans la forêt (défaut : 100).
                          Plus ce nombre est élevé, plus le modèle est précis mais lent.
            graine_aleatoire: Graine aléatoire pour la reproductibilité (défaut : 42).
            features: Liste des colonnes features. Utilise FEATURES_MODELE par défaut.
            chemin_modele: Chemin de sauvegarde du modèle entraîné (format joblib).
            chemin_scaler: Chemin de sauvegarde du scaler de normalisation.
        """
        # Taux de contamination : proportion d'anomalies attendues dans les données
        self.contamination: float = contamination

        # Nombre d'arbres dans la forêt d'isolation
        self.n_estimateurs: int = n_estimateurs

        # Graine aléatoire pour garantir des résultats reproductibles
        self.graine_aleatoire: int = graine_aleatoire

        # Liste des features utilisées par le modèle (défaut si non spécifiée)
        self.features: List[str] = features if features is not None else FEATURES_MODELE

        # Chemin de sauvegarde du modèle entraîné
        self.chemin_modele: str = chemin_modele

        # Chemin de sauvegarde du scaler de normalisation
        self.chemin_scaler: str = chemin_scaler

        # Initialisation de l'algorithme Isolation Forest avec les hyperparamètres
        self.modele: IsolationForest = IsolationForest(
            n_estimators=self.n_estimateurs,   # Nombre d'arbres dans la forêt
            contamination=self.contamination,  # Proportion d'anomalies attendues
            random_state=self.graine_aleatoire, # Reproductibilité
            n_jobs=-1,                          # Utilisation de tous les cœurs CPU disponibles
        )

        # Initialisation du scaler pour normaliser les features avant l'entraînement
        # La normalisation est essentielle pour que les features d'unités différentes
        # (ex: RSI en [0,100] vs VaR en [-0.05, 0]) aient le même poids dans le modèle
        self.scaler: StandardScaler = StandardScaler()

        # Indicateur d'état du modèle : False avant l'entraînement, True après
        self.est_entraine: bool = False

        # Journalisation de l'initialisation du détecteur
        logger.info(
            f"DetecteurAnomaliesBoursières initialisé | "
            f"Contamination : {self.contamination:.1%} | "
            f"N estimateurs : {self.n_estimateurs} | "
            f"Features : {len(self.features)}"
        )

    def _preparer_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Sélectionne et valide les features disponibles dans le DataFrame.

        Cette méthode gère le cas où certaines features configurées ne sont
        pas présentes dans le DataFrame (ex: colonne calculée manquante).
        Elle adapte dynamiquement la liste des features disponibles.

        Args:
            df: DataFrame contenant les données enrichies par la couche Transform.

        Returns:
            Tableau NumPy des features sélectionnées, prêt pour le modèle ML.
        """
        # Identification des features configurées mais absentes du DataFrame
        features_manquantes = [f for f in self.features if f not in df.columns]

        # Alerte si des features configurées sont manquantes dans le DataFrame
        if features_manquantes:
            logger.warning(
                f"Features manquantes dans le DataFrame : {features_manquantes}. "
                f"Ces colonnes seront ignorées par le modèle."
            )

        # Sélection des features disponibles parmi celles configurées
        features_disponibles = [f for f in self.features if f in df.columns]

        # Vérification qu'il reste au moins une feature utilisable
        if not features_disponibles:
            raise ValueError(
                "Aucune feature disponible dans le DataFrame. "
                "Vérifiez que la transformation a bien été appliquée."
            )

        # Journalisation des features effectivement utilisées par le modèle
        logger.info(f"Features utilisées pour le modèle : {features_disponibles}")

        # Extraction du sous-DataFrame contenant uniquement les features sélectionnées
        X = df[features_disponibles].values

        # Vérification de l'absence de valeurs NaN dans les features
        # Les NaN peuvent apparaître si la transformation n'est pas complète
        if np.isnan(X).any():
            nb_nan = np.isnan(X).sum()
            logger.warning(
                f"Valeurs NaN détectées dans les features ({nb_nan} valeurs). "
                f"Remplacement par la médiane de chaque colonne."
            )
            # Remplacement des NaN par la médiane de chaque colonne
            for i in range(X.shape[1]):
                masque_nan = np.isnan(X[:, i])
                if masque_nan.any():
                    X[masque_nan, i] = np.nanmedian(X[:, i])

        return X

    def entrainer(self, df: pd.DataFrame, uri_mlflow: str = "mlruns") -> dict:
        """
        Entraîne le modèle Isolation Forest sur les données historiques.

        L'entraînement est tracé dans MLflow pour garantir la reproductibilité
        et permettre la comparaison entre différentes versions du modèle.
        Cette pratique est au cœur de la culture MLOps que Fortuneo souhaite
        développer au sein de son Data Lab.

        Args:
            df: DataFrame des données transformées contenant les features ML.
            uri_mlflow: URI du serveur de tracking MLflow (local ou AWS).

        Returns:
            Dictionnaire contenant les métriques d'entraînement et les chemins
            des artefacts (modèle sauvegardé, scaler).
        """
        # Vérification que le DataFrame d'entraînement n'est pas vide
        if df.empty:
            logger.error("Données d'entraînement vides. Entraînement annulé.")
            return {"succes": False}

        # Configuration du serveur de tracking MLflow
        # En production : URI d'un serveur MLflow hébergé sur AWS EC2 ou SageMaker
        mlflow.set_tracking_uri(uri_mlflow)

        # Définition de l'expérience MLflow pour regrouper les runs par projet
        mlflow.set_experiment("fortuneo_detection_anomalies")

        # Journalisation du début de l'entraînement
        logger.info(
            f"Début de l'entraînement du modèle | "
            f"{len(df)} observations | "
            f"{df['symbole'].nunique()} symboles"
        )

        # Démarrage d'un run MLflow pour tracer cette session d'entraînement
        with mlflow.start_run(run_name=f"isolation_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            # --- Journalisation des hyperparamètres dans MLflow ---
            # MLflow log_params() enregistre les paramètres du modèle pour la reproductibilité
            mlflow.log_params({
                "algorithme": "IsolationForest",           # Algorithme utilisé
                "contamination": self.contamination,        # Taux d'anomalies attendu
                "n_estimateurs": self.n_estimateurs,        # Nombre d'arbres
                "graine_aleatoire": self.graine_aleatoire,  # Graine de reproductibilité
                "nb_features": len(self.features),          # Nombre de features
                "nb_observations": len(df),                 # Taille du jeu d'entraînement
                "nb_symboles": df["symbole"].nunique(),     # Nombre de symboles
            })

            # --- Préparation des features ---
            # Extraction du tableau NumPy des features depuis le DataFrame
            X = self._preparer_features(df)

            # Normalisation des features : chaque feature est centrée (moyenne = 0)
            # et réduite (écart-type = 1) pour éviter qu'une feature domine les autres
            X_normalise = self.scaler.fit_transform(X)

            # --- Entraînement de l'Isolation Forest ---
            logger.info("Entraînement de l'Isolation Forest en cours...")
            self.modele.fit(X_normalise)

            # Mise à jour du flag indiquant que le modèle est maintenant entraîné
            self.est_entraine = True

            # --- Calcul des métriques d'entraînement ---
            # Scores d'anomalie : valeurs négatives plus basses = plus anormal
            scores_anomalie = self.modele.score_samples(X_normalise)

            # Prédictions : -1 pour les anomalies, +1 pour les observations normales
            predictions = self.modele.predict(X_normalise)

            # Calcul du nombre d'anomalies détectées sur l'ensemble d'entraînement
            nb_anomalies = int((predictions == -1).sum())

            # Calcul du taux d'anomalies réellement détecté
            taux_anomalies_reel = nb_anomalies / len(predictions)

            # Journalisation des métriques dans MLflow pour comparaison ultérieure
            mlflow.log_metrics({
                "nb_anomalies_detectees": nb_anomalies,
                "taux_anomalies_reel": taux_anomalies_reel,
                "score_anomalie_moyen": float(scores_anomalie.mean()),
                "score_anomalie_min": float(scores_anomalie.min()),
                "score_anomalie_max": float(scores_anomalie.max()),
            })

            # --- Sauvegarde du modèle et du scaler sur disque ---
            # Création du dossier de destination si nécessaire
            os.makedirs(os.path.dirname(self.chemin_modele) if os.path.dirname(self.chemin_modele) else ".", exist_ok=True)

            # Sauvegarde du modèle Isolation Forest au format joblib
            # joblib est plus efficace que pickle pour les objets NumPy/scikit-learn
            joblib.dump(self.modele, self.chemin_modele)

            # Sauvegarde du scaler pour normaliser les données lors de l'inférence
            joblib.dump(self.scaler, self.chemin_scaler)

            # Journalisation des artefacts dans MLflow (chemins des fichiers sauvegardés)
            mlflow.log_artifact(self.chemin_modele)
            mlflow.log_artifact(self.chemin_scaler)

            # Enregistrement du modèle scikit-learn directement dans MLflow
            # (permet le déploiement via MLflow Model Registry en production)
            mlflow.sklearn.log_model(
                sk_model=self.modele,
                artifact_path="isolation_forest",
            )

        # Bilan de l'entraînement à retourner au pipeline principal
        metriques = {
            "succes": True,
            "nb_observations": len(df),
            "nb_anomalies": nb_anomalies,
            "taux_anomalies": taux_anomalies_reel,
            "chemin_modele": self.chemin_modele,
            "chemin_scaler": self.chemin_scaler,
        }

        # Journalisation du résumé de l'entraînement
        logger.info(
            f"Entraînement terminé | "
            f"Anomalies détectées : {nb_anomalies} ({taux_anomalies_reel:.2%}) | "
            f"Modèle sauvegardé : {self.chemin_modele}"
        )

        return metriques

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les anomalies boursières sur de nouvelles données.

        Cette méthode est utilisée en production pour analyser les nouvelles
        séances boursières au fil de l'eau et détecter les comportements anormaux
        sans avoir à ré-entraîner le modèle à chaque exécution.

        Args:
            df: DataFrame des données transformées à analyser.

        Returns:
            DataFrame enrichi avec les colonnes :
            - 'score_anomalie' : score d'isolation (plus négatif = plus anormal)
            - 'est_anomalie'   : booléen indiquant si l'observation est anormale
            - 'label_anomalie' : libellé lisible ("Anomalie" ou "Normal")
        """
        # Vérification que le modèle a bien été entraîné avant la prédiction
        if not self.est_entraine:
            logger.error(
                "Le modèle n'a pas encore été entraîné. "
                "Appelez la méthode entrainer() ou charger_modele() avant predire()."
            )
            return df

        # Vérification que le DataFrame n'est pas vide
        if df.empty:
            logger.warning("DataFrame de prédiction vide. Aucune prédiction effectuée.")
            return df

        # Journalisation du début de la prédiction
        logger.info(f"Prédiction des anomalies sur {len(df)} observations...")

        # Copie du DataFrame pour ne pas modifier l'original (immutabilité)
        df_avec_predictions = df.copy()

        # Préparation des features : sélection et gestion des NaN
        X = self._preparer_features(df_avec_predictions)

        # Normalisation des features avec le scaler ajusté lors de l'entraînement
        # IMPORTANT : on utilise transform() et non fit_transform() pour appliquer
        # la même normalisation que lors de l'entraînement
        X_normalise = self.scaler.transform(X)

        # Calcul des scores d'anomalie par l'Isolation Forest
        # score_samples() retourne le score d'isolation normalisé
        # Plus le score est négatif (bas), plus l'observation est anormale
        scores = self.modele.score_samples(X_normalise)

        # Prédiction binaire : -1 pour les anomalies, +1 pour les observations normales
        predictions = self.modele.predict(X_normalise)

        # Ajout de la colonne des scores d'anomalie au DataFrame résultat
        df_avec_predictions["score_anomalie"] = scores

        # Ajout de la colonne booléenne : True si anomalie (-1), False si normal (+1)
        df_avec_predictions["est_anomalie"] = predictions == -1

        # Ajout d'un libellé lisible pour faciliter l'interprétation des résultats
        df_avec_predictions["label_anomalie"] = df_avec_predictions["est_anomalie"].map(
            {True: "Anomalie", False: "Normal"}
        )

        # Calcul du résumé des anomalies détectées
        nb_anomalies = int(df_avec_predictions["est_anomalie"].sum())
        taux_anomalies = nb_anomalies / len(df_avec_predictions)

        # Journalisation du résumé des prédictions
        logger.info(
            f"Prédiction terminée | "
            f"Anomalies détectées : {nb_anomalies}/{len(df_avec_predictions)} "
            f"({taux_anomalies:.2%})"
        )

        # Journalisation des anomalies les plus sévères (top 5 scores les plus bas)
        if nb_anomalies > 0:
            anomalies_top5 = (
                df_avec_predictions[df_avec_predictions["est_anomalie"]]
                .nsmallest(5, "score_anomalie")[["date", "symbole", "score_anomalie", "cloture"]]
            )
            logger.warning(
                f"Top 5 anomalies les plus sévères :\n{anomalies_top5.to_string(index=False)}"
            )

        return df_avec_predictions

    def charger_modele(self) -> bool:
        """
        Charge un modèle Isolation Forest précédemment entraîné et sauvegardé.

        Permet de réutiliser un modèle entraîné lors d'une exécution précédente
        sans avoir à ré-entraîner, ce qui économise du temps de calcul en production.

        Returns:
            True si le chargement a réussi, False si les fichiers sont introuvables.
        """
        try:
            # Vérification de l'existence du fichier modèle
            if not os.path.exists(self.chemin_modele):
                logger.warning(
                    f"Fichier modèle introuvable : {self.chemin_modele}. "
                    f"L'entraînement est nécessaire."
                )
                return False

            # Vérification de l'existence du fichier scaler
            if not os.path.exists(self.chemin_scaler):
                logger.warning(
                    f"Fichier scaler introuvable : {self.chemin_scaler}. "
                    f"L'entraînement est nécessaire."
                )
                return False

            # Chargement du modèle Isolation Forest depuis le fichier joblib
            self.modele = joblib.load(self.chemin_modele)

            # Chargement du scaler depuis le fichier joblib
            self.scaler = joblib.load(self.chemin_scaler)

            # Mise à jour du flag d'état : le modèle est maintenant utilisable
            self.est_entraine = True

            # Journalisation du succès du chargement
            logger.info(
                f"Modèle et scaler chargés avec succès | "
                f"Modèle : {self.chemin_modele} | "
                f"Scaler : {self.chemin_scaler}"
            )
            return True

        except Exception as erreur:
            # Capture de toute erreur de déserialisation ou de compatibilité
            logger.error(
                f"Erreur lors du chargement du modèle : "
                f"{type(erreur).__name__} - {str(erreur)}"
            )
            return False

    def obtenir_rapport_anomalies(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Génère un rapport synthétique des anomalies détectées par symbole.

        Ce rapport permet aux équipes de gestion du risque de Fortuneo d'identifier
        rapidement quels titres présentent le plus d'anomalies et leur sévérité moyenne.

        Args:
            df_predictions: DataFrame retourné par la méthode predire().

        Returns:
            DataFrame de rapport avec une ligne par symbole et les statistiques d'anomalies.
        """
        # Vérification que les colonnes de prédiction sont présentes
        colonnes_requises = ["symbole", "est_anomalie", "score_anomalie"]
        if not all(col in df_predictions.columns for col in colonnes_requises):
            logger.error(
                "Les colonnes de prédiction sont absentes. "
                "Appelez predire() avant obtenir_rapport_anomalies()."
            )
            return pd.DataFrame()

        # Agrégation des statistiques d'anomalies par symbole
        rapport = (
            df_predictions.groupby("symbole")
            .agg(
                nb_observations=("est_anomalie", "count"),       # Total d'observations
                nb_anomalies=("est_anomalie", "sum"),            # Nombre d'anomalies
                taux_anomalies=("est_anomalie", "mean"),         # Taux d'anomalies (%)
                score_moyen=("score_anomalie", "mean"),          # Score moyen d'anomalie
                score_min=("score_anomalie", "min"),             # Score le plus anormal
            )
            .reset_index()
        )

        # Tri par taux d'anomalies décroissant pour mettre en avant les titres à risque
        rapport = rapport.sort_values("taux_anomalies", ascending=False)

        # Formatage du taux d'anomalies en pourcentage pour la lisibilité
        rapport["taux_anomalies_pct"] = (rapport["taux_anomalies"] * 100).round(2)

        # Journalisation du rapport pour la traçabilité
        logger.info(
            f"Rapport d'anomalies généré pour {len(rapport)} symboles | "
            f"Total anomalies : {int(rapport['nb_anomalies'].sum())}"
        )

        return rapport
