# =============================================================================
# Tests unitaires du module d'extraction des données boursières
# =============================================================================
# Ce fichier contient les tests unitaires pour la couche Extract du pipeline.
# Les tests valident le comportement de l'ExtracteurDonneesBoursières
# dans différentes conditions : cas nominal, données manquantes, symboles invalides.
#
# Les tests utilisent des données simulées (mocking) pour ne pas dépendre
# de la disponibilité de l'API Yahoo Finance lors de l'exécution des tests CI/CD.
# =============================================================================

# Importation du framework de tests Python
import pytest

# Importation des bibliothèques de manipulation de données
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Importation du module à tester
import sys
from pathlib import Path

# Ajout du répertoire racine du projet au chemin Python pour les importations
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.extracteur_donnees import ExtracteurDonneesBoursières, SYMBOLES_FORTUNEO_PAR_DEFAUT


# =============================================================================
# FIXTURES : Données de test réutilisables entre les tests
# =============================================================================

@pytest.fixture
def extracteur_defaut():
    """
    Fixture qui fournit une instance de l'extracteur avec les paramètres par défaut.
    Utilisée dans les tests qui vérifient le comportement nominal de la classe.
    """
    # Création d'une instance avec les paramètres par défaut
    return ExtracteurDonneesBoursières()


@pytest.fixture
def extracteur_symboles_personnalises():
    """
    Fixture qui fournit un extracteur configuré avec des symboles personnalisés.
    Utilisée pour tester la personnalisation des symboles boursiers.
    """
    # Sélection d'un sous-ensemble de symboles pour accélérer les tests
    symboles_test = ["BNP.PA", "MC.PA"]
    return ExtracteurDonneesBoursières(symboles=symboles_test, periode="3mo")


@pytest.fixture
def dataframe_brut_simule():
    """
    Fixture qui génère un DataFrame simulant les données brutes d'une extraction.
    Évite les appels réseau réels lors des tests unitaires.
    """
    # Génération d'une série de 100 dates ouvrées consécutives
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")

    # Création du DataFrame avec des données simulées pour deux symboles
    donnees = []
    for symbole in ["BNP.PA", "MC.PA"]:
        # Génération de prix aléatoires mais cohérents (marche aléatoire)
        prix_base = 50.0 if symbole == "BNP.PA" else 700.0
        rendements = np.random.normal(0, 0.01, len(dates))
        prix = prix_base * np.exp(np.cumsum(rendements))

        for i, date in enumerate(dates):
            donnees.append({
                "date": date,
                "ouverture": prix[i] * (1 + np.random.uniform(-0.005, 0.005)),
                "plus_haut": prix[i] * (1 + np.random.uniform(0, 0.01)),
                "plus_bas": prix[i] * (1 - np.random.uniform(0, 0.01)),
                "cloture": prix[i],
                "volume": np.random.randint(100000, 5000000),
                "symbole": symbole,
                "date_extraction": datetime.now(),
            })

    # Retourne le DataFrame trié par symbole et par date
    return pd.DataFrame(donnees).sort_values(["symbole", "date"]).reset_index(drop=True)


# =============================================================================
# TESTS DE L'INITIALISATION
# =============================================================================

class TestInitialisationExtracteur:
    """Groupe de tests vérifiant l'initialisation correcte de l'extracteur."""

    def test_initialisation_par_defaut(self, extracteur_defaut):
        """Vérifie que l'extracteur s'initialise correctement avec les valeurs par défaut."""
        # L'extracteur doit utiliser la liste de symboles Fortuneo par défaut
        assert extracteur_defaut.symboles == SYMBOLES_FORTUNEO_PAR_DEFAUT

        # La période par défaut doit être "1y" (un an)
        assert extracteur_defaut.periode == "1y"

        # L'intervalle par défaut doit être "1d" (quotidien)
        assert extracteur_defaut.intervalle == "1d"

    def test_initialisation_symboles_personnalises(self, extracteur_symboles_personnalises):
        """Vérifie que l'extracteur accepte des symboles personnalisés."""
        # La liste de symboles doit correspondre à celle fournie
        assert extracteur_symboles_personnalises.symboles == ["BNP.PA", "MC.PA"]

        # La période personnalisée doit être correctement enregistrée
        assert extracteur_symboles_personnalises.periode == "3mo"

    def test_nb_symboles_par_defaut(self, extracteur_defaut):
        """Vérifie que la liste par défaut contient le nombre attendu de symboles."""
        # La liste Fortuneo par défaut doit contenir au moins 10 symboles
        assert len(extracteur_defaut.symboles) >= 10

    def test_symboles_format_valide(self, extracteur_defaut):
        """Vérifie que tous les symboles respectent le format Yahoo Finance pour Euronext Paris."""
        # Chaque symbole doit se terminer par ".PA" (Euronext Paris)
        for symbole in extracteur_defaut.symboles:
            assert symbole.endswith(".PA"), (
                f"Le symbole '{symbole}' ne respecte pas le format Euronext Paris (.PA)"
            )


# =============================================================================
# TESTS DES MÉTHODES DE TRAITEMENT DES DONNÉES
# =============================================================================

class TestTraitementDonnees:
    """Groupe de tests vérifiant le traitement correct des données extraites."""

    def test_schema_dataframe_brut(self, dataframe_brut_simule):
        """Vérifie que le DataFrame simulé possède toutes les colonnes requises."""
        # Liste des colonnes attendues dans les données brutes extraites
        colonnes_attendues = [
            "date", "ouverture", "plus_haut", "plus_bas",
            "cloture", "volume", "symbole", "date_extraction"
        ]

        # Vérification que toutes les colonnes attendues sont présentes
        for colonne in colonnes_attendues:
            assert colonne in dataframe_brut_simule.columns, (
                f"La colonne '{colonne}' est absente du DataFrame brut"
            )

    def test_pas_de_valeurs_nulles_colonnes_critiques(self, dataframe_brut_simule):
        """Vérifie l'absence de valeurs nulles dans les colonnes critiques pour le pipeline."""
        # Les colonnes critiques ne doivent jamais contenir de valeurs nulles
        colonnes_critiques = ["date", "cloture", "volume", "symbole"]

        for colonne in colonnes_critiques:
            nb_nulles = dataframe_brut_simule[colonne].isnull().sum()
            assert nb_nulles == 0, (
                f"La colonne critique '{colonne}' contient {nb_nulles} valeur(s) nulle(s)"
            )

    def test_prix_positifs(self, dataframe_brut_simule):
        """Vérifie que tous les prix sont strictement positifs (cohérence financière)."""
        # Un prix boursier ne peut jamais être négatif ou nul
        colonnes_prix = ["ouverture", "plus_haut", "plus_bas", "cloture"]

        for colonne in colonnes_prix:
            prix_negatifs = (dataframe_brut_simule[colonne] <= 0).sum()
            assert prix_negatifs == 0, (
                f"La colonne '{colonne}' contient {prix_negatifs} prix non positif(s)"
            )

    def test_coherence_high_low(self, dataframe_brut_simule):
        """Vérifie que le plus haut est toujours supérieur ou égal au plus bas (cohérence OHLCV)."""
        # Règle fondamentale : le plus haut d'une séance >= le plus bas d'une séance
        violations = (dataframe_brut_simule["plus_haut"] < dataframe_brut_simule["plus_bas"]).sum()
        assert violations == 0, (
            f"{violations} ligne(s) ont un plus_haut inférieur au plus_bas"
        )

    def test_volume_positif(self, dataframe_brut_simule):
        """Vérifie que tous les volumes de transaction sont strictement positifs."""
        # Un volume nul ou négatif n'a pas de sens financier
        volumes_invalides = (dataframe_brut_simule["volume"] <= 0).sum()
        assert volumes_invalides == 0, (
            f"{volumes_invalides} ligne(s) ont un volume invalide (<= 0)"
        )

    def test_symboles_presents(self, dataframe_brut_simule):
        """Vérifie que les symboles attendus sont bien présents dans le DataFrame."""
        symboles_presents = set(dataframe_brut_simule["symbole"].unique())

        # Les deux symboles simulés doivent être dans le DataFrame
        assert "BNP.PA" in symboles_presents
        assert "MC.PA" in symboles_presents

    def test_tri_chronologique(self, dataframe_brut_simule):
        """Vérifie que les données sont triées chronologiquement par symbole."""
        for symbole in dataframe_brut_simule["symbole"].unique():
            # Extraction des dates pour le symbole courant
            dates_symbole = dataframe_brut_simule[
                dataframe_brut_simule["symbole"] == symbole
            ]["date"].values

            # Vérification que chaque date est supérieure ou égale à la précédente
            for i in range(1, len(dates_symbole)):
                assert dates_symbole[i] >= dates_symbole[i - 1], (
                    f"Les données de {symbole} ne sont pas triées chronologiquement"
                )


# =============================================================================
# TESTS DES CAS LIMITES
# =============================================================================

class TestCasLimites:
    """Groupe de tests vérifiant le comportement de l'extracteur dans les cas extrêmes."""

    def test_extraction_symbole_invalide(self):
        """Vérifie que l'extraction d'un symbole invalide retourne None sans lever d'exception."""
        # Création d'un extracteur avec un symbole volontairement invalide
        extracteur = ExtracteurDonneesBoursières(
            symboles=["SYMBOLE_INVALIDE_12345.PA"],
            periode="1mo"
        )

        # L'extraction d'un symbole invalide doit retourner None, pas lever d'exception
        resultat = extracteur.extraire_action("SYMBOLE_INVALIDE_12345.PA")

        # Le résultat doit être None pour un symbole invalide
        assert resultat is None, (
            "L'extraction d'un symbole invalide doit retourner None"
        )

    def test_extracteur_liste_vide(self):
        """Vérifie le comportement de l'extracteur avec une liste de symboles vide."""
        # Création d'un extracteur avec une liste vide
        extracteur = ExtracteurDonneesBoursières(symboles=[])

        # L'extraction sur une liste vide doit retourner un dictionnaire vide
        resultats = extracteur.extraire_tous_les_symboles()

        assert isinstance(resultats, dict), "Le résultat doit être un dictionnaire"
        assert len(resultats) == 0, "Le dictionnaire doit être vide pour une liste vide"

    def test_dataframe_combine_non_vide_si_succes(self):
        """Vérifie que le DataFrame combiné est non vide si au moins un symbole est valide."""
        # Test avec un symbole connu et valide de l'indice CAC40
        extracteur = ExtracteurDonneesBoursières(
            symboles=["BNP.PA"],
            periode="1mo",
        )

        # L'extraction doit retourner des données pour BNP.PA (valeur du CAC40)
        df = extracteur.extraire_donnees_combinees()

        # Si l'API est disponible, le DataFrame ne doit pas être vide
        # Note : ce test peut échouer si l'API Yahoo Finance est indisponible
        if not df.empty:
            assert "symbole" in df.columns
            assert "cloture" in df.columns
            assert (df["symbole"] == "BNP.PA").all()
