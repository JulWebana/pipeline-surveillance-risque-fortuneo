# =============================================================================
# Module de chargement des données - Étape Load du pipeline ETL
# =============================================================================
# Ce module constitue la troisième et dernière étape du pipeline ETL
# (Extract - Transform - Load). Il est responsable de la persistance des données
# transformées dans deux supports complémentaires :
#
#   1. Une base de données relationnelle SQLite (en local) qui simule une instance
#      AWS RDS (PostgreSQL) utilisée en production chez Fortuneo.
#
#   2. Des fichiers au format Apache Parquet, format colonnaire optimisé pour
#      la lecture analytique et la compatibilité avec AWS S3 + Athena.
#
# Problématique métier : les données enrichies doivent être disponibles aussi bien
# pour des requêtes SQL ad hoc (analyses des équipes métiers) que pour des traitements
# massifs en batch (modèles ML, rapports réglementaires, tableaux de bord).
# =============================================================================

# Importation des bibliothèques standard Python
import logging   # Gestion de la journalisation des événements du pipeline
import os        # Interactions avec le système de fichiers (création de dossiers)
from datetime import datetime   # Manipulation des dates pour l'horodatage des fichiers
from typing import Optional      # Annotation de type pour les valeurs potentiellement absentes

# Importation des bibliothèques tierces
import pandas as pd                          # Manipulation des DataFrames et export Parquet
from sqlalchemy import create_engine, text   # ORM pour la connexion et les requêtes SQL
from sqlalchemy.engine import Engine         # Type de l'objet moteur SQLAlchemy

# Récupération du logger configuré au niveau du pipeline principal
logger = logging.getLogger(__name__)


class ChargeurDonnees:
    """
    Classe responsable du chargement des données transformées dans les systèmes de stockage.

    Cette classe constitue la couche "Load" du pipeline ETL.
    Elle gère deux types de stockage complémentaires :
    - Base de données relationnelle (SQLite / AWS RDS en production) pour les requêtes SQL.
    - Fichiers Parquet (local / AWS S3 en production) pour les traitements analytiques.

    En production chez Fortuneo, le moteur SQLite serait remplacé par une connexion
    sécurisée à une instance AWS RDS PostgreSQL, et les fichiers Parquet seraient
    déposés dans un bucket S3 accessible via AWS Glue et Amazon Athena.

    Attributs :
        chemin_bdd (str): Chemin de la base de données SQLite.
        dossier_parquet (str): Chemin du dossier de stockage des fichiers Parquet.
        moteur (Engine): Moteur SQLAlchemy pour les opérations sur la base de données.
    """

    def __init__(
        self,
        chemin_bdd: str = "data/fortuneo_pipeline.db",
        dossier_parquet: str = "data/transformees/",
    ) -> None:
        """
        Initialise le chargeur avec les connexions aux systèmes de stockage.

        Args:
            chemin_bdd: Chemin complet du fichier de base de données SQLite.
                        En production : chaîne de connexion PostgreSQL AWS RDS.
            dossier_parquet: Chemin du dossier local pour les fichiers Parquet.
                             En production : URI S3 (ex: s3://bucket-fortuneo/silver/).
        """
        # Stockage du chemin de la base de données pour les opérations ultérieures
        self.chemin_bdd: str = chemin_bdd

        # Stockage du chemin du dossier Parquet
        self.dossier_parquet: str = dossier_parquet

        # Création du dossier de la base de données s'il n'existe pas
        os.makedirs(os.path.dirname(chemin_bdd) if os.path.dirname(chemin_bdd) else ".", exist_ok=True)

        # Création du dossier des fichiers Parquet s'il n'existe pas
        os.makedirs(dossier_parquet, exist_ok=True)

        # Initialisation du moteur SQLAlchemy pour la connexion à la base de données
        # create_engine() crée un pool de connexions réutilisables et thread-safe
        # Préfixe "sqlite:///" pour SQLite (3 slashes = chemin relatif)
        self.moteur: Engine = create_engine(
            f"sqlite:///{chemin_bdd}",
            echo=False,   # Désactivation des logs SQL verbeux (mettre True pour déboguer)
        )

        # Journalisation de l'initialisation du chargeur
        logger.info(
            f"ChargeurDonnees initialisé | "
            f"Base de données : {chemin_bdd} | "
            f"Dossier Parquet : {dossier_parquet}"
        )

    # =========================================================================
    # MÉTHODES DE CHARGEMENT EN BASE DE DONNÉES (SIMULATION AWS RDS)
    # =========================================================================

    def charger_en_base(
        self,
        df: pd.DataFrame,
        nom_table: str,
        comportement: str = "replace",
    ) -> bool:
        """
        Charge un DataFrame dans une table de la base de données SQLite.

        En production, cette méthode se connecterait à une instance AWS RDS
        PostgreSQL hautement disponible (Multi-AZ) avec chiffrement au repos.

        Args:
            df: DataFrame pandas contenant les données à charger.
            nom_table: Nom de la table cible dans la base de données.
            comportement: Comportement si la table existe déjà :
                         - "replace" : supprime et recrée la table (défaut)
                         - "append"  : ajoute les lignes sans supprimer l'existant
                         - "fail"    : lève une erreur si la table existe

        Returns:
            True si le chargement a réussi, False en cas d'erreur.
        """
        try:
            # Vérification que le DataFrame n'est pas vide avant de tenter le chargement
            if df.empty:
                logger.warning(f"Le DataFrame est vide. Chargement annulé pour la table '{nom_table}'.")
                return False

            # Chargement du DataFrame dans la table SQL via pandas to_sql()
            # to_sql() gère automatiquement la création du schéma et l'insertion des données
            df.to_sql(
                name=nom_table,         # Nom de la table cible dans la base
                con=self.moteur,        # Moteur SQLAlchemy pour la connexion
                if_exists=comportement, # Comportement si la table existe déjà
                index=False,            # Ne pas inclure l'index pandas comme colonne SQL
                chunksize=1000,         # Insertion par lots de 1000 lignes pour les performances
            )

            # Journalisation du succès du chargement avec les informations clés
            logger.info(
                f"Chargement réussi en base : {len(df)} lignes → table '{nom_table}' "
                f"(comportement : '{comportement}')"
            )
            return True

        except Exception as erreur:
            # Capture des erreurs de connexion, de schéma ou d'écriture
            logger.error(
                f"Erreur lors du chargement en base dans la table '{nom_table}' : "
                f"{type(erreur).__name__} - {str(erreur)}"
            )
            return False

    def lire_depuis_base(
        self,
        requete_sql: str,
    ) -> Optional[pd.DataFrame]:
        """
        Exécute une requête SQL et retourne le résultat sous forme de DataFrame.

        Cette méthode permet aux équipes d'analystes de requêter directement
        les données transformées en SQL pour leurs analyses ad hoc.

        Args:
            requete_sql: Requête SQL SELECT à exécuter sur la base de données.

        Returns:
            DataFrame contenant les résultats de la requête, ou None en cas d'erreur.
        """
        try:
            # Exécution de la requête SQL via pandas read_sql_query()
            # La connexion est automatiquement fermée après l'exécution
            with self.moteur.connect() as connexion:
                df_resultat = pd.read_sql_query(
                    sql=text(requete_sql),   # text() sécurise la requête contre les injections SQL
                    con=connexion,
                )

            # Journalisation du succès de la lecture SQL
            logger.info(
                f"Requête SQL exécutée avec succès : {len(df_resultat)} lignes retournées"
            )
            return df_resultat

        except Exception as erreur:
            # Capture des erreurs de syntaxe SQL ou de connexion
            logger.error(
                f"Erreur lors de l'exécution de la requête SQL : "
                f"{type(erreur).__name__} - {str(erreur)}"
            )
            return None

    def obtenir_statistiques_table(self, nom_table: str) -> Optional[dict]:
        """
        Retourne des statistiques descriptives sur une table de la base de données.

        Args:
            nom_table: Nom de la table à analyser.

        Returns:
            Dictionnaire contenant le nombre de lignes, de symboles et la plage de dates.
        """
        try:
            # Requête SQL pour obtenir les statistiques essentielles de la table
            requete = f"""
                SELECT
                    COUNT(*) AS nb_lignes,
                    COUNT(DISTINCT symbole) AS nb_symboles,
                    MIN(date) AS date_debut,
                    MAX(date) AS date_fin
                FROM {nom_table}
            """

            # Exécution de la requête et récupération du résultat
            resultat = self.lire_depuis_base(requete)

            if resultat is not None and not resultat.empty:
                # Conversion de la première ligne en dictionnaire Python
                statistiques = resultat.iloc[0].to_dict()
                logger.info(f"Statistiques de la table '{nom_table}' : {statistiques}")
                return statistiques

            return None

        except Exception as erreur:
            logger.error(f"Erreur lors du calcul des statistiques : {str(erreur)}")
            return None

    # =========================================================================
    # MÉTHODES DE SAUVEGARDE EN FICHIERS PARQUET (SIMULATION AWS S3)
    # =========================================================================

    def sauvegarder_parquet(
        self,
        df: pd.DataFrame,
        nom_fichier: str,
        partitionner_par_symbole: bool = True,
    ) -> str:
        """
        Sauvegarde un DataFrame au format Apache Parquet.

        Le format Parquet est un format colonnaire optimisé pour les analyses massives.
        Il est nativement supporté par AWS S3, Athena, Spark et Pandas.
        Ses avantages : compression efficace, lecture sélective des colonnes,
        schéma auto-documenté et performances supérieures au CSV pour les grands volumes.

        En production, cette méthode utiliserait boto3 pour déposer les fichiers
        directement dans un bucket AWS S3 avec chiffrement SSE-S3.

        Args:
            df: DataFrame à sauvegarder au format Parquet.
            nom_fichier: Nom de base du fichier (sans extension).
            partitionner_par_symbole: Si True, crée un fichier Parquet par symbole
                                      pour optimiser les requêtes filtrées par symbole.

        Returns:
            Chemin du fichier Parquet créé (ou du dossier si partitionné).
        """
        # Vérification que le DataFrame n'est pas vide
        if df.empty:
            logger.warning("Le DataFrame est vide. Sauvegarde Parquet annulée.")
            return ""

        if partitionner_par_symbole and "symbole" in df.columns:
            # --- Sauvegarde partitionnée par symbole ---
            # Cette stratégie améliore les performances des requêtes filtrées
            # sur un ou plusieurs symboles spécifiques (pattern Hive Partitioning)
            dossier_partition = os.path.join(self.dossier_parquet, nom_fichier)
            os.makedirs(dossier_partition, exist_ok=True)

            # Itération sur chaque symbole pour créer un fichier Parquet dédié
            for symbole in df["symbole"].unique():
                # Filtrage du DataFrame pour ne garder que les données du symbole courant
                df_symbole = df[df["symbole"] == symbole]

                # Nettoyage du nom de symbole pour le nom de fichier (remplacement du "." par "_")
                nom_symbole_nettoye = symbole.replace(".", "_")

                # Construction du chemin complet du fichier Parquet partitionné
                chemin_fichier = os.path.join(
                    dossier_partition, f"{nom_symbole_nettoye}.parquet"
                )

                # Sauvegarde du sous-DataFrame au format Parquet
                # engine='pyarrow' est le moteur le plus performant et compatible AWS
                df_symbole.to_parquet(
                    path=chemin_fichier,
                    engine="pyarrow",    # Moteur Apache Arrow (compatible AWS)
                    index=False,         # Ne pas inclure l'index pandas dans le fichier
                    compression="snappy", # Compression Snappy : bon compromis vitesse/taille
                )

            # Journalisation du succès de la sauvegarde partitionnée
            logger.info(
                f"Sauvegarde Parquet partitionnée : {dossier_partition} "
                f"({df['symbole'].nunique()} fichiers, {len(df)} lignes au total)"
            )
            return dossier_partition

        else:
            # --- Sauvegarde en fichier Parquet unique ---
            # Horodatage du nom de fichier pour garantir l'unicité
            horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
            chemin_fichier = os.path.join(
                self.dossier_parquet, f"{nom_fichier}_{horodatage}.parquet"
            )

            # Sauvegarde du DataFrame complet dans un seul fichier Parquet
            df.to_parquet(
                path=chemin_fichier,
                engine="pyarrow",
                index=False,
                compression="snappy",
            )

            # Journalisation du succès de la sauvegarde
            taille_ko = os.path.getsize(chemin_fichier) / 1024
            logger.info(
                f"Sauvegarde Parquet réussie : {chemin_fichier} "
                f"({len(df)} lignes, {taille_ko:.1f} Ko)"
            )
            return chemin_fichier

    # =========================================================================
    # MÉTHODE PRINCIPALE DE CHARGEMENT
    # =========================================================================

    def charger(
        self,
        df_transforme: pd.DataFrame,
        df_predictions: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Orchestre le chargement complet des données dans tous les systèmes de stockage.

        C'est le point d'entrée principal de la couche Load.
        Elle charge les données transformées et les prédictions ML dans la base
        de données et au format Parquet, et retourne un bilan du chargement.

        Args:
            df_transforme: DataFrame des données enrichies par la couche Transform.
            df_predictions: DataFrame des anomalies détectées par le modèle ML (optionnel).

        Returns:
            Dictionnaire contenant le statut et les chemins des fichiers créés.
        """
        # Initialisation du bilan de chargement
        bilan = {
            "succes": False,
            "nb_lignes_transformees": 0,
            "nb_lignes_predictions": 0,
            "chemin_parquet_transforme": "",
            "chemin_parquet_predictions": "",
        }

        # Vérification que les données transformées ne sont pas vides
        if df_transforme.empty:
            logger.error("Données transformées vides. Chargement impossible.")
            return bilan

        # --- Chargement des données transformées en base de données ---
        succes_bdd_transforme = self.charger_en_base(
            df=df_transforme,
            nom_table="donnees_boursières_transformees",
            comportement="replace",
        )

        # --- Sauvegarde des données transformées au format Parquet ---
        chemin_parquet_transforme = self.sauvegarder_parquet(
            df=df_transforme,
            nom_fichier="donnees_transformees",
            partitionner_par_symbole=True,
        )

        # Mise à jour du bilan avec les résultats du chargement des données transformées
        bilan["nb_lignes_transformees"] = len(df_transforme)
        bilan["chemin_parquet_transforme"] = chemin_parquet_transforme

        # --- Chargement des prédictions ML (si fournies) ---
        if df_predictions is not None and not df_predictions.empty:
            # Chargement des anomalies détectées en base de données
            succes_bdd_predictions = self.charger_en_base(
                df=df_predictions,
                nom_table="anomalies_detectees",
                comportement="replace",
            )

            # Sauvegarde des prédictions au format Parquet
            chemin_parquet_predictions = self.sauvegarder_parquet(
                df=df_predictions,
                nom_fichier="anomalies_detectees",
                partitionner_par_symbole=True,
            )

            # Mise à jour du bilan avec les résultats du chargement des prédictions
            bilan["nb_lignes_predictions"] = len(df_predictions)
            bilan["chemin_parquet_predictions"] = chemin_parquet_predictions

        # Marquage du chargement comme réussi si toutes les étapes se sont bien déroulées
        bilan["succes"] = succes_bdd_transforme and bool(chemin_parquet_transforme)

        # Journalisation du bilan global du chargement
        logger.info(
            f"Chargement terminé | "
            f"Données transformées : {bilan['nb_lignes_transformees']} lignes | "
            f"Anomalies : {bilan['nb_lignes_predictions']} lignes | "
            f"Statut : {'✓ Succès' if bilan['succes'] else '✗ Échec'}"
        )

        return bilan
