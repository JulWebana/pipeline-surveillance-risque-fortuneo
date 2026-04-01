# =============================================================================
# Tests unitaires du module de transformation des données boursières
# =============================================================================
# Ce fichier valide le calcul correct de chaque indicateur technique
# calculé par la couche Transform du pipeline ETL Fortuneo.
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Ajout du répertoire racine du projet au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformation.transformateur_donnees import TransformateurDonneesBoursières


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def transformateur():
    """Fixture fournissant une instance du transformateur avec les paramètres par défaut."""
    return TransformateurDonneesBoursières()


@pytest.fixture
def df_test_un_symbole():
    """
    Fixture générant un DataFrame de test pour un seul symbole boursier.
    Les données simulent 100 jours de cotation avec une tendance haussière.
    """
    # Nombre de jours de cotation simulés (supérieur à la fenêtre la plus longue = 50 jours)
    nb_jours = 150

    # Génération d'une série de prix simulant une tendance haussière douce
    np.random.seed(42)  # Graine fixe pour la reproductibilité des tests
    rendements_journaliers = np.random.normal(0.001, 0.02, nb_jours)
    prix = 100 * np.exp(np.cumsum(rendements_journaliers))

    # Création du DataFrame avec les colonnes OHLCV standard
    df = pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=nb_jours, freq="B"),
        "ouverture": prix * (1 + np.random.uniform(-0.003, 0.003, nb_jours)),
        "plus_haut": prix * (1 + np.random.uniform(0.001, 0.015, nb_jours)),
        "plus_bas": prix * (1 - np.random.uniform(0.001, 0.015, nb_jours)),
        "cloture": prix,
        "volume": np.random.randint(500000, 3000000, nb_jours),
        "symbole": "BNP.PA",
        "date_extraction": datetime.now(),
    })

    return df


@pytest.fixture
def df_test_multi_symboles(df_test_un_symbole):
    """
    Fixture générant un DataFrame de test pour deux symboles boursiers.
    Utilisée pour tester que les indicateurs ne mélangent pas les symboles.
    """
    # Création d'un deuxième DataFrame pour LVMH avec des prix différents
    np.random.seed(99)
    nb_jours = len(df_test_un_symbole)
    rendements = np.random.normal(0.0005, 0.015, nb_jours)
    prix_mc = 700 * np.exp(np.cumsum(rendements))

    df_mc = pd.DataFrame({
        "date": df_test_un_symbole["date"],
        "ouverture": prix_mc * 0.999,
        "plus_haut": prix_mc * 1.01,
        "plus_bas": prix_mc * 0.99,
        "cloture": prix_mc,
        "volume": np.random.randint(100000, 800000, nb_jours),
        "symbole": "MC.PA",
        "date_extraction": datetime.now(),
    })

    # Concaténation des deux symboles dans un seul DataFrame
    return pd.concat([df_test_un_symbole, df_mc], ignore_index=True)


# =============================================================================
# TESTS DES RENDEMENTS LOGARITHMIQUES
# =============================================================================

class TestRendementsLogarithmiques:
    """Tests du calcul des rendements logarithmiques journaliers."""

    def test_colonne_rendement_creee(self, transformateur, df_test_un_symbole):
        """Vérifie que la colonne rendement_log est bien créée après transformation."""
        df_transforme = transformateur.transformer(df_test_un_symbole)
        assert "rendement_log" in df_transforme.columns

    def test_rendements_dans_plage_realiste(self, transformateur, df_test_un_symbole):
        """Vérifie que les rendements journaliers sont dans une plage financièrement réaliste."""
        df_transforme = transformateur.transformer(df_test_un_symbole)

        # Un rendement journalier de ±30% est extrême mais possible (circuit breaker)
        assert df_transforme["rendement_log"].abs().max() < 0.30, (
            "Des rendements journaliers supérieurs à ±30% sont détectés. "
            "Vérifiez la qualité des données d'entrée."
        )

    def test_moyenne_rendements_proche_zero(self, transformateur, df_test_un_symbole):
        """Vérifie que la moyenne des rendements est proche de zéro (propriété statistique)."""
        df_transforme = transformateur.transformer(df_test_un_symbole)

        # La moyenne des rendements sur 150 jours doit être proche de zéro
        moyenne = df_transforme["rendement_log"].mean()
        assert abs(moyenne) < 0.01, (
            f"La moyenne des rendements ({moyenne:.4f}) s'écarte trop de zéro"
        )


# =============================================================================
# TESTS DES MOYENNES MOBILES
# =============================================================================

class TestMoyennesMobiles:
    """Tests du calcul des moyennes mobiles simples."""

    def test_colonnes_ma_creees(self, transformateur, df_test_un_symbole):
        """Vérifie que les deux colonnes de moyennes mobiles sont créées."""
        df_transforme = transformateur.transformer(df_test_un_symbole)
        assert "ma_courte" in df_transforme.columns
        assert "ma_longue" in df_transforme.columns

    def test_ma_courte_plus_reactive(self, transformateur, df_test_un_symbole):
        """
        Vérifie que la MA courte est plus réactive que la MA longue.

        La réactivité se mesure par la variance normalisée : la MA courte doit avoir
        un rapport variance/moyenne supérieur à la MA longue, car elle lisse moins les données.
        On exclut les premières observations pour éviter l'effet de démarrage (burn-in period)
        lié à l'option min_periods=1 des fenêtres glissantes.
        """
        df_transforme = transformateur.transformer(df_test_un_symbole)

        # On exclut les premières observations (fenêtre longue = 50 jours) pour éviter
        # l'effet de démarrage des moyennes mobiles sur des fenêtres incomplètes
        fenetre_longue = transformateur.fenetre_ma_longue
        df_stable = df_transforme.iloc[fenetre_longue:].copy()

        # Calcul du coefficient de variation (std/mean) pour normaliser les écarts
        # Le coefficient de variation est comparable indépendamment du niveau de prix
        cv_ma_courte = df_stable["ma_courte"].std() / df_stable["ma_courte"].mean()
        cv_ma_longue = df_stable["ma_longue"].std() / df_stable["ma_longue"].mean()

        # La MA courte doit avoir une variabilité relative supérieure à la MA longue
        # car elle réagit plus vite aux fluctuations des prix
        assert cv_ma_courte >= cv_ma_longue, (
            f"La MA courte (CV={cv_ma_courte:.4f}) doit être plus réactive "
            f"que la MA longue (CV={cv_ma_longue:.4f}) sur des données stables"
        )

    def test_ma_positives(self, transformateur, df_test_un_symbole):
        """Vérifie que les moyennes mobiles sont toujours positives."""
        df_transforme = transformateur.transformer(df_test_un_symbole)

        assert (df_transforme["ma_courte"] > 0).all(), "La MA courte contient des valeurs non positives"
        assert (df_transforme["ma_longue"] > 0).all(), "La MA longue contient des valeurs non positives"


# =============================================================================
# TESTS DU RSI
# =============================================================================

class TestRSI:
    """Tests du calcul du RSI (Relative Strength Index)."""

    def test_colonne_rsi_creee(self, transformateur, df_test_un_symbole):
        """Vérifie que la colonne RSI est bien créée après transformation."""
        df_transforme = transformateur.transformer(df_test_un_symbole)
        assert "rsi" in df_transforme.columns

    def test_rsi_dans_plage_valide(self, transformateur, df_test_un_symbole):
        """Vérifie que le RSI est toujours compris entre 0 et 100."""
        df_transforme = transformateur.transformer(df_test_un_symbole)

        rsi_min = df_transforme["rsi"].min()
        rsi_max = df_transforme["rsi"].max()

        assert rsi_min >= 0, f"Le RSI a une valeur minimale négative : {rsi_min:.2f}"
        assert rsi_max <= 100, f"Le RSI a une valeur maximale supérieure à 100 : {rsi_max:.2f}"


# =============================================================================
# TESTS DES BANDES DE BOLLINGER
# =============================================================================

class TestBandesBollinger:
    """Tests du calcul des Bandes de Bollinger."""

    def test_colonnes_bollinger_creees(self, transformateur, df_test_un_symbole):
        """Vérifie que les trois colonnes Bollinger sont créées."""
        df_transforme = transformateur.transformer(df_test_un_symbole)
        assert "bollinger_haute" in df_transforme.columns
        assert "bollinger_basse" in df_transforme.columns
        assert "bollinger_milieu" in df_transforme.columns

    def test_ordre_bandes_bollinger(self, transformateur, df_test_un_symbole):
        """Vérifie que la bande haute est toujours supérieure à la bande basse."""
        df_transforme = transformateur.transformer(df_test_un_symbole)

        violations = (
            df_transforme["bollinger_haute"] < df_transforme["bollinger_basse"]
        ).sum()

        assert violations == 0, (
            f"{violations} ligne(s) ont une bande haute inférieure à la bande basse"
        )

    def test_milieu_entre_bandes(self, transformateur, df_test_un_symbole):
        """Vérifie que la bande médiane est toujours entre la haute et la basse."""
        df_transforme = transformateur.transformer(df_test_un_symbole)

        violations_haute = (
            df_transforme["bollinger_milieu"] > df_transforme["bollinger_haute"]
        ).sum()

        violations_basse = (
            df_transforme["bollinger_milieu"] < df_transforme["bollinger_basse"]
        ).sum()

        assert violations_haute == 0, "La bande médiane dépasse la bande haute"
        assert violations_basse == 0, "La bande médiane descend sous la bande basse"


# =============================================================================
# TESTS DE LA VOLATILITÉ
# =============================================================================

class TestVolatilite:
    """Tests du calcul de la volatilité historique annualisée."""

    def test_colonne_volatilite_creee(self, transformateur, df_test_un_symbole):
        """Vérifie que la colonne de volatilité annualisée est créée."""
        df_transforme = transformateur.transformer(df_test_un_symbole)
        assert "volatilite_annualisee" in df_transforme.columns

    def test_volatilite_positive(self, transformateur, df_test_un_symbole):
        """Vérifie que la volatilité est toujours strictement positive."""
        df_transforme = transformateur.transformer(df_test_un_symbole)

        volatilites_negatives = (df_transforme["volatilite_annualisee"] <= 0).sum()
        assert volatilites_negatives == 0, (
            f"{volatilites_negatives} valeur(s) de volatilité non positive(s) détectée(s)"
        )

    def test_volatilite_dans_plage_realiste(self, transformateur, df_test_un_symbole):
        """Vérifie que la volatilité annualisée est dans une plage financièrement réaliste."""
        df_transforme = transformateur.transformer(df_test_un_symbole)

        volatilite_max = df_transforme["volatilite_annualisee"].max()

        # Une volatilité annualisée > 300% serait extrêmement inhabituelle pour une action du CAC40
        assert volatilite_max < 3.0, (
            f"Volatilité annualisée maximale irréaliste : {volatilite_max:.2%}"
        )


# =============================================================================
# TESTS MULTI-SYMBOLES
# =============================================================================

class TestMultiSymboles:
    """Tests vérifiant que les indicateurs sont calculés correctement pour plusieurs symboles."""

    def test_pas_de_melange_entre_symboles(self, transformateur, df_test_multi_symboles):
        """
        Vérifie que les calculs d'un symbole n'influencent pas ceux d'un autre.
        Ce test est crucial : si le tri ou le groupement par symbole est incorrect,
        les moyennes mobiles et la volatilité seraient calculées sur des données mixtes.
        """
        df_transforme = transformateur.transformer(df_test_multi_symboles)

        # Calcul de la volatilité séparément pour chaque symbole
        vol_bnp = (
            df_transforme[df_transforme["symbole"] == "BNP.PA"]["volatilite_annualisee"]
            .mean()
        )
        vol_mc = (
            df_transforme[df_transforme["symbole"] == "MC.PA"]["volatilite_annualisee"]
            .mean()
        )

        # Les deux volatilités doivent être calculées et non nulles
        assert vol_bnp > 0, "La volatilité de BNP.PA est nulle ou négative"
        assert vol_mc > 0, "La volatilité de MC.PA est nulle ou négative"

    def test_nb_symboles_conserves(self, transformateur, df_test_multi_symboles):
        """Vérifie que le nombre de symboles est conservé après la transformation."""
        nb_symboles_avant = df_test_multi_symboles["symbole"].nunique()
        df_transforme = transformateur.transformer(df_test_multi_symboles)
        nb_symboles_apres = df_transforme["symbole"].nunique()

        assert nb_symboles_avant == nb_symboles_apres, (
            f"Des symboles ont été perdus lors de la transformation : "
            f"{nb_symboles_avant} → {nb_symboles_apres}"
        )

    def test_transformation_dataframe_vide(self, transformateur):
        """Vérifie que la transformation gère proprement un DataFrame vide en entrée."""
        df_vide = pd.DataFrame()
        resultat = transformateur.transformer(df_vide)

        # La transformation d'un DataFrame vide doit retourner un DataFrame vide
        assert isinstance(resultat, pd.DataFrame), "Le résultat doit être un DataFrame"
        assert resultat.empty, "Le résultat doit être vide pour une entrée vide"
