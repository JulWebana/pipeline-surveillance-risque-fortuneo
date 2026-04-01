# =============================================================================
# Orchestrateur principal du pipeline ETL - Point d'entrée du projet
# =============================================================================
# Ce script est le chef d'orchestre du pipeline ETL de surveillance du risque
# boursier de Fortuneo. Il coordonne séquentiellement les quatre grandes étapes :
#
#   1. EXTRACTION  : Récupération des cours boursiers depuis Yahoo Finance
#   2. TRANSFORMATION : Calcul des indicateurs techniques et métriques de risque
#   3. DÉTECTION ML : Identification des anomalies par Isolation Forest
#   4. CHARGEMENT  : Persistance des données en base et au format Parquet
#
# Le monitoring de chaque étape est assuré par le MoniteurPipeline qui enregistre
# les métriques, durées et alertes dans un rapport JSON horodaté.
#
# Usage :
#   python src/pipeline.py                     # Exécution avec configuration par défaut
#   python src/pipeline.py --config config/config.yaml  # Avec fichier de configuration
# =============================================================================

# Importation des bibliothèques standard Python
import argparse   # Gestion des arguments en ligne de commande
import os         # Opérations système (chemins, variables d'environnement)
import sys        # Accès au chemin Python pour les importations locales
from pathlib import Path   # Manipulation des chemins de fichiers

# Importation de la gestion de la configuration YAML
import yaml   # Lecture du fichier de configuration YAML

# --- Résolution du chemin du projet ---
# Ajout du dossier racine au chemin Python pour permettre les importations des modules src/
RACINE_PROJET = Path(__file__).parent.parent
sys.path.insert(0, str(RACINE_PROJET))

# --- Importation des modules du pipeline ---
# Chaque import correspond à une couche du pipeline ETL
from src.monitoring.moniteur_pipeline import configurer_logging, MoniteurPipeline
from src.extraction.extracteur_donnees import ExtracteurDonneesBoursières
from src.transformation.transformateur_donnees import TransformateurDonneesBoursières
from src.modeles.detecteur_anomalies import DetecteurAnomaliesBoursières
from src.chargement.chargeur_donnees import ChargeurDonnees

# Configuration initiale du système de journalisation
# (sera surchargée par la configuration YAML si disponible)
import logging
logger = logging.getLogger(__name__)


def charger_configuration(chemin_config: str = "config/config.yaml") -> dict:
    """
    Charge le fichier de configuration YAML du pipeline.

    La centralisation de la configuration dans un fichier YAML permet de modifier
    les paramètres du pipeline (symboles, périodes, seuils) sans toucher au code.
    C'est une bonne pratique d'ingénierie logicielle pour les pipelines de production.

    Args:
        chemin_config: Chemin vers le fichier de configuration YAML.

    Returns:
        Dictionnaire Python contenant l'ensemble des paramètres de configuration.
    """
    # Vérification que le fichier de configuration existe bien
    if not os.path.exists(chemin_config):
        # Si le fichier n'existe pas, on utilise une configuration minimale par défaut
        logger.warning(
            f"Fichier de configuration introuvable : {chemin_config}. "
            f"Utilisation des paramètres par défaut."
        )
        # Retour d'une configuration vide (les classes utiliseront leurs valeurs par défaut)
        return {}

    # Ouverture et lecture du fichier YAML
    with open(chemin_config, "r", encoding="utf-8") as fichier:
        # safe_load() est utilisé par sécurité (évite l'exécution de code arbitraire)
        configuration = yaml.safe_load(fichier)

    logger.info(f"Configuration chargée depuis : {chemin_config}")
    return configuration or {}


def executer_pipeline(chemin_config: str = "config/config.yaml") -> dict:
    """
    Exécute le pipeline ETL complet de surveillance du risque boursier Fortuneo.

    Cette fonction orchestre toutes les étapes du pipeline dans l'ordre séquentiel
    correct, en s'assurant que chaque étape reçoit les données de la précédente.
    Le monitoring est activé dès le début pour capturer les métriques de chaque étape.

    Args:
        chemin_config: Chemin vers le fichier de configuration du pipeline.

    Returns:
        Dictionnaire contenant les métriques complètes du run et le statut final.
    """
    # =========================================================================
    # INITIALISATION : Configuration et monitoring
    # =========================================================================

    # Chargement de la configuration YAML du pipeline
    config = charger_configuration(chemin_config)

    # Configuration du système de journalisation avec les paramètres du fichier config
    config_logging = config.get("logging", {})
    configurer_logging(
        niveau=config_logging.get("niveau", "INFO"),
        fichier_log=config_logging.get("fichier_log", "logs/pipeline.log"),
        taille_max=config_logging.get("taille_max_log", 10 * 1024 * 1024),
        nb_fichiers=config_logging.get("nb_fichiers_archives", 5),
    )

    # Initialisation du moniteur du pipeline pour suivre toutes les métriques du run
    moniteur = MoniteurPipeline(
        dossier_rapports=config_logging.get("dossier_logs", "logs/")
    )

    logger.info("=" * 60)
    logger.info("  PIPELINE ETL - SURVEILLANCE DU RISQUE BOURSIER FORTUNEO")
    logger.info("=" * 60)

    # Indicateur global de succès du pipeline (devient False si une étape échoue)
    pipeline_en_succes = True

    # =========================================================================
    # ÉTAPE 1 : EXTRACTION DES DONNÉES BOURSIÈRES
    # =========================================================================

    # Démarrage du monitoring pour l'étape d'extraction
    horodatage_extraction = moniteur.demarrer_etape("extraction")

    try:
        # Récupération des paramètres d'extraction depuis le fichier de configuration
        config_extraction = config.get("extraction", {})

        # Instanciation de l'extracteur avec les paramètres configurés
        extracteur = ExtracteurDonneesBoursières(
            symboles=config_extraction.get("symboles", None),     # Liste des symboles
            periode=config_extraction.get("periode", "1y"),       # Fenêtre temporelle
            intervalle=config_extraction.get("intervalle", "1d"), # Granularité
        )

        # Exécution de l'extraction : récupération de toutes les données boursières
        df_brut = extracteur.extraire_donnees_combinees()

        # Vérification que l'extraction a produit des données
        if df_brut.empty:
            raise ValueError("L'extraction n'a produit aucune donnée. Vérifiez la connexion réseau.")

        # Sauvegarde des données brutes dans le dossier Bronze (couche raw data lake)
        config_chargement = config.get("chargement", {})
        extracteur.sauvegarder_donnees_brutes(
            donnees=df_brut,
            dossier=config_chargement.get("dossier_brutes", "data/brutes/"),
        )

        # Enregistrement des métriques de qualité des données extraites
        metriques_extraction = moniteur.enregistrer_metrique_qualite(df_brut, "extraction")

        # Clôture de l'étape d'extraction avec succès
        moniteur.terminer_etape(
            nom_etape="extraction",
            horodatage_debut=horodatage_extraction,
            succes=True,
            metriques_supplementaires=metriques_extraction,
        )

    except Exception as erreur:
        # En cas d'erreur critique lors de l'extraction, le pipeline s'arrête
        logger.critical(f"Erreur fatale lors de l'extraction : {str(erreur)}")
        moniteur.terminer_etape("extraction", horodatage_extraction, succes=False)
        pipeline_en_succes = False
        return moniteur.cloture_pipeline(succes_global=False)

    # =========================================================================
    # ÉTAPE 2 : TRANSFORMATION ET ENRICHISSEMENT DES DONNÉES
    # =========================================================================

    # Démarrage du monitoring pour l'étape de transformation
    horodatage_transformation = moniteur.demarrer_etape("transformation")

    try:
        # Récupération des paramètres de transformation depuis la configuration
        config_transformation = config.get("transformation", {})

        # Instanciation du transformateur avec les paramètres d'indicateurs techniques
        transformateur = TransformateurDonneesBoursières(
            fenetre_ma_courte=config_transformation.get("fenetre_ma_courte", 20),
            fenetre_ma_longue=config_transformation.get("fenetre_ma_longue", 50),
            periode_rsi=config_transformation.get("periode_rsi", 14),
            fenetre_volatilite=config_transformation.get("fenetre_volatilite", 30),
            fenetre_bollinger=config_transformation.get("fenetre_bollinger", 20),
            nb_ecarts_types=config_transformation.get("nb_ecarts_types_bollinger", 2.0),
        )

        # Application de toutes les transformations sur les données brutes
        df_transforme = transformateur.transformer(df_brut)

        # Vérification que la transformation a produit des données
        if df_transforme.empty:
            raise ValueError("La transformation n'a produit aucune donnée valide.")

        # Enregistrement des métriques de qualité des données transformées
        metriques_transformation = moniteur.enregistrer_metrique_qualite(
            df_transforme, "transformation"
        )

        # Clôture de l'étape de transformation avec succès
        moniteur.terminer_etape(
            nom_etape="transformation",
            horodatage_debut=horodatage_transformation,
            succes=True,
            metriques_supplementaires=metriques_transformation,
        )

    except Exception as erreur:
        # En cas d'erreur dans la transformation, le pipeline est arrêté
        logger.critical(f"Erreur fatale lors de la transformation : {str(erreur)}")
        moniteur.terminer_etape("transformation", horodatage_transformation, succes=False)
        pipeline_en_succes = False
        return moniteur.cloture_pipeline(succes_global=False)

    # =========================================================================
    # ÉTAPE 3 : DÉTECTION DES ANOMALIES PAR MACHINE LEARNING
    # =========================================================================

    # Démarrage du monitoring pour l'étape de Machine Learning
    horodatage_ml = moniteur.demarrer_etape("detection_anomalies")

    try:
        # Récupération des paramètres du modèle ML depuis la configuration
        config_modele = config.get("modele", {})
        config_mlflow = config.get("mlflow", {})

        # Instanciation du détecteur d'anomalies avec les hyperparamètres configurés
        detecteur = DetecteurAnomaliesBoursières(
            contamination=config_modele.get("contamination", 0.05),
            n_estimateurs=config_modele.get("n_estimateurs", 100),
            graine_aleatoire=config_modele.get("graine_aleatoire", 42),
            chemin_modele=config_modele.get("chemin_modele", "mlruns/modele_detection_anomalies.pkl"),
            chemin_scaler=config_modele.get("chemin_scaler", "mlruns/scaler_features.pkl"),
        )

        # Tentative de chargement d'un modèle existant pour éviter le ré-entraînement
        modele_charge = detecteur.charger_modele()

        if not modele_charge:
            # Si aucun modèle sauvegardé, entraînement sur les données historiques
            logger.info("Aucun modèle existant. Entraînement du modèle Isolation Forest...")
            metriques_entrainement = detecteur.entrainer(
                df=df_transforme,
                uri_mlflow=config_mlflow.get("tracking_uri", "mlruns"),
            )
        else:
            # Utilisation du modèle préexistant pour l'inférence
            logger.info("Modèle existant chargé. Passage direct à la prédiction.")
            metriques_entrainement = {"succes": True, "modele_precharge": True}

        # Application du modèle pour détecter les anomalies sur toutes les données
        df_avec_anomalies = detecteur.predire(df_transforme)

        # Génération du rapport d'anomalies par symbole
        rapport_anomalies = detecteur.obtenir_rapport_anomalies(df_avec_anomalies)

        # Journalisation du rapport d'anomalies pour information des équipes
        if not rapport_anomalies.empty:
            logger.info(f"Rapport d'anomalies par symbole :\n{rapport_anomalies.to_string(index=False)}")

        # Clôture de l'étape ML avec les métriques d'entraînement
        moniteur.terminer_etape(
            nom_etape="detection_anomalies",
            horodatage_debut=horodatage_ml,
            succes=True,
            metriques_supplementaires={
                "nb_anomalies": int(df_avec_anomalies["est_anomalie"].sum()),
                "taux_anomalies": float(df_avec_anomalies["est_anomalie"].mean()),
                **metriques_entrainement,
            },
        )

    except Exception as erreur:
        # En cas d'erreur ML, le pipeline continue sans les prédictions
        logger.error(f"Erreur lors de la détection des anomalies : {str(erreur)}")
        moniteur.terminer_etape("detection_anomalies", horodatage_ml, succes=False)
        # Le pipeline continue pour charger au moins les données transformées
        df_avec_anomalies = df_transforme

    # =========================================================================
    # ÉTAPE 4 : CHARGEMENT DES DONNÉES EN BASE ET AU FORMAT PARQUET
    # =========================================================================

    # Démarrage du monitoring pour l'étape de chargement
    horodatage_chargement = moniteur.demarrer_etape("chargement")

    try:
        # Récupération des paramètres de chargement depuis la configuration
        config_chargement = config.get("chargement", {})

        # Instanciation du chargeur avec les chemins de stockage configurés
        chargeur = ChargeurDonnees(
            chemin_bdd=config_chargement.get("chemin_base_de_donnees", "data/fortuneo_pipeline.db"),
            dossier_parquet=config_chargement.get("dossier_parquet", "data/transformees/"),
        )

        # Chargement des données dans tous les systèmes de stockage
        bilan_chargement = chargeur.charger(
            df_transforme=df_transforme,
            df_predictions=df_avec_anomalies if "est_anomalie" in df_avec_anomalies.columns else None,
        )

        # Clôture de l'étape de chargement avec le bilan
        moniteur.terminer_etape(
            nom_etape="chargement",
            horodatage_debut=horodatage_chargement,
            succes=bilan_chargement.get("succes", False),
            metriques_supplementaires=bilan_chargement,
        )

        # Mise à jour du statut global du pipeline
        if not bilan_chargement.get("succes", False):
            pipeline_en_succes = False

    except Exception as erreur:
        # En cas d'erreur de chargement, le pipeline est marqué en échec
        logger.critical(f"Erreur fatale lors du chargement : {str(erreur)}")
        moniteur.terminer_etape("chargement", horodatage_chargement, succes=False)
        pipeline_en_succes = False

    # =========================================================================
    # CLÔTURE DU PIPELINE
    # =========================================================================

    # Génération du rapport final de monitoring et sauvegarde JSON
    rapport_final = moniteur.cloture_pipeline(succes_global=pipeline_en_succes)

    return rapport_final


def main() -> None:
    """
    Point d'entrée principal du pipeline ETL Fortuneo.

    Gère les arguments de la ligne de commande et lance l'exécution du pipeline.
    """
    # Définition des arguments acceptés par le script en ligne de commande
    parseur = argparse.ArgumentParser(
        description="Pipeline ETL de surveillance du risque boursier - Fortuneo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python src/pipeline.py
  python src/pipeline.py --config config/config.yaml
        """,
    )

    # Argument optionnel pour spécifier un fichier de configuration personnalisé
    parseur.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration YAML (défaut : config/config.yaml)",
    )

    # Analyse des arguments fournis en ligne de commande
    arguments = parseur.parse_args()

    # Exécution du pipeline avec le fichier de configuration spécifié
    rapport = executer_pipeline(chemin_config=arguments.config)

    # Code de sortie : 0 si succès, 1 si échec (convention Unix)
    code_sortie = 0 if rapport.get("statut") == "succès" else 1
    sys.exit(code_sortie)


# Point d'entrée du script Python
# Ce bloc n'est exécuté que si le script est lancé directement (pas importé)
if __name__ == "__main__":
    main()
