# =============================================================================
# Tests unitaires du module de détection d'anomalies (Machine Learning)
# =============================================================================
# Ce fichier valide le comportement du modèle Isolation Forest utilisé pour
# détecter les anomalies boursières dans le pipeline Fortuneo.
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Ajout du répertoire racine du projet au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeles.detecteur_anomalies import DetecteurAnomaliesBoursières, FEATURES_MODELE
from src.transformation.transformateur_donnees import TransformateurDonneesBoursières


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def detecteur():
    """Fixture fournissant un détecteur d'anomalies avec les paramètres de test."""
    return DetecteurAnomaliesBoursières(
        contamination=0.05,
        n_estimateurs=50,  # Moins d'arbres pour accélérer les tests
        graine_aleatoire=42,
        chemin_modele="/tmp/test_modele.pkl",
        chemin_scaler="/tmp/test_scaler.pkl",
    )


@pytest.fixture
def df_entraine():
    """
    Fixture générant un DataFrame transformé complet pour l'entraînement du modèle.
    Inclut toutes les features nécessaires au modèle ML.
    """
    # Génération de données simulées pour l'entraînement
    np.random.seed(42)
    nb_jours = 200

    # Simulation d'une série de prix pour BNP.PA
    rendements = np.random.normal(0.001, 0.015, nb_jours)
    prix = 50 * np.exp(np.cumsum(rendements))

    df_brut = pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=nb_jours, freq="B"),
        "ouverture": prix * 0.999,
        "plus_haut": prix * 1.01,
        "plus_bas": prix * 0.99,
        "cloture": prix,
        "volume": np.random.randint(500000, 3000000, nb_jours),
        "symbole": "BNP.PA",
        "date_extraction": datetime.now(),
    })

    # Application de la transformation pour calculer toutes les features
    transformateur = TransformateurDonneesBoursières()
    return transformateur.transformer(df_brut)


# =============================================================================
# TESTS DE L'INITIALISATION DU DÉTECTEUR
# =============================================================================

class TestInitialisationDetecteur:
    """Tests de l'initialisation correcte du détecteur d'anomalies."""

    def test_etat_initial_non_entraine(self, detecteur):
        """Vérifie que le modèle est marqué comme non entraîné à l'initialisation."""
        assert detecteur.est_entraine is False

    def test_contamination_configuree(self, detecteur):
        """Vérifie que le taux de contamination est correctement configuré."""
        assert detecteur.contamination == 0.05

    def test_features_par_defaut(self, detecteur):
        """Vérifie que la liste de features par défaut est bien assignée."""
        assert detecteur.features == FEATURES_MODELE
        assert len(detecteur.features) > 0


# =============================================================================
# TESTS DE L'ENTRAÎNEMENT
# =============================================================================

class TestEntrainement:
    """Tests du processus d'entraînement du modèle Isolation Forest."""

    def test_modele_marque_entraine_apres_fit(self, detecteur, df_entraine):
        """Vérifie que le flag est_entraine passe à True après l'entraînement."""
        detecteur.entrainer(df_entraine, uri_mlflow="/tmp/test_mlruns")
        assert detecteur.est_entraine is True

    def test_metriques_entrainement_retournees(self, detecteur, df_entraine):
        """Vérifie que l'entraînement retourne un dictionnaire de métriques."""
        metriques = detecteur.entrainer(df_entraine, uri_mlflow="/tmp/test_mlruns")

        # Le bilan doit indiquer le succès de l'entraînement
        assert metriques.get("succes") is True

        # Le nombre d'anomalies détectées doit être présent
        assert "nb_anomalies" in metriques
        assert "taux_anomalies" in metriques

    def test_taux_anomalies_coherent_avec_contamination(self, detecteur, df_entraine):
        """Vérifie que le taux d'anomalies est proche du paramètre de contamination."""
        metriques = detecteur.entrainer(df_entraine, uri_mlflow="/tmp/test_mlruns")

        taux_anomalies = metriques.get("taux_anomalies", 0)

        # Le taux d'anomalies réel doit être proche du taux de contamination configuré
        # On accepte une marge de ±2% autour de la valeur configurée
        assert abs(taux_anomalies - detecteur.contamination) < 0.02, (
            f"Taux d'anomalies réel ({taux_anomalies:.2%}) trop éloigné "
            f"du taux de contamination configuré ({detecteur.contamination:.2%})"
        )

    def test_entrainement_dataframe_vide(self, detecteur):
        """Vérifie que l'entraînement sur un DataFrame vide retourne un échec propre."""
        df_vide = pd.DataFrame()
        metriques = detecteur.entrainer(df_vide, uri_mlflow="/tmp/test_mlruns")

        assert metriques.get("succes") is False
        assert detecteur.est_entraine is False


# =============================================================================
# TESTS DE LA PRÉDICTION
# =============================================================================

class TestPrediction:
    """Tests du processus de prédiction des anomalies."""

    def test_colonnes_predictions_presentes(self, detecteur, df_entraine):
        """Vérifie que les colonnes de prédiction sont ajoutées au DataFrame."""
        detecteur.entrainer(df_entraine, uri_mlflow="/tmp/test_mlruns")
        df_predit = detecteur.predire(df_entraine)

        assert "score_anomalie" in df_predit.columns
        assert "est_anomalie" in df_predit.columns
        assert "label_anomalie" in df_predit.columns

    def test_label_anomalie_valeurs_valides(self, detecteur, df_entraine):
        """Vérifie que les labels d'anomalie ne contiennent que les valeurs attendues."""
        detecteur.entrainer(df_entraine, uri_mlflow="/tmp/test_mlruns")
        df_predit = detecteur.predire(df_entraine)

        valeurs_valides = {"Anomalie", "Normal"}
        valeurs_presentes = set(df_predit["label_anomalie"].unique())

        assert valeurs_presentes.issubset(valeurs_valides), (
            f"Labels inattendus détectés : {valeurs_presentes - valeurs_valides}"
        )

    def test_est_anomalie_type_booleen(self, detecteur, df_entraine):
        """Vérifie que la colonne est_anomalie contient bien des booléens."""
        detecteur.entrainer(df_entraine, uri_mlflow="/tmp/test_mlruns")
        df_predit = detecteur.predire(df_entraine)

        assert df_predit["est_anomalie"].dtype == bool, (
            f"La colonne est_anomalie doit être de type bool, "
            f"pas {df_predit['est_anomalie'].dtype}"
        )

    def test_prediction_sans_entrainement_echoue_proprement(self, detecteur, df_entraine):
        """Vérifie que la prédiction sans entraînement retourne le DataFrame inchangé."""
        # Sans entraînement préalable, la prédiction doit retourner le DataFrame original
        df_result = detecteur.predire(df_entraine)

        # Le DataFrame retourné ne doit pas contenir les colonnes de prédiction
        assert "est_anomalie" not in df_result.columns


# =============================================================================
# TESTS DU RAPPORT D'ANOMALIES
# =============================================================================

class TestRapportAnomalies:
    """Tests de la génération du rapport d'anomalies par symbole."""

    def test_rapport_contient_toutes_colonnes(self, detecteur, df_entraine):
        """Vérifie que le rapport d'anomalies contient toutes les colonnes attendues."""
        detecteur.entrainer(df_entraine, uri_mlflow="/tmp/test_mlruns")
        df_predit = detecteur.predire(df_entraine)
        rapport = detecteur.obtenir_rapport_anomalies(df_predit)

        colonnes_attendues = [
            "symbole", "nb_observations", "nb_anomalies",
            "taux_anomalies", "score_moyen", "score_min"
        ]

        for colonne in colonnes_attendues:
            assert colonne in rapport.columns, (
                f"La colonne '{colonne}' est absente du rapport d'anomalies"
            )

    def test_rapport_un_symbole_par_ligne(self, detecteur, df_entraine):
        """Vérifie que le rapport contient exactement une ligne par symbole."""
        detecteur.entrainer(df_entraine, uri_mlflow="/tmp/test_mlruns")
        df_predit = detecteur.predire(df_entraine)
        rapport = detecteur.obtenir_rapport_anomalies(df_predit)

        nb_symboles_donnees = df_entraine["symbole"].nunique()
        nb_lignes_rapport = len(rapport)

        assert nb_lignes_rapport == nb_symboles_donnees, (
            f"Le rapport doit avoir {nb_symboles_donnees} lignes, "
            f"mais en contient {nb_lignes_rapport}"
        )
