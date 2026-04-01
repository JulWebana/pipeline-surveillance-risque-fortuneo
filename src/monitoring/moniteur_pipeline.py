# =============================================================================
# Module de surveillance du pipeline ETL - Monitoring et observabilité
# =============================================================================
# Ce module implémente la couche de monitoring du pipeline Fortuneo.
# Il assure la traçabilité complète de chaque exécution du pipeline,
# enregistre les métriques de performance et génère des alertes en cas
# de comportement anormal du pipeline lui-même ou des données traitées.
#
# Problématique métier : un pipeline de production doit être supervisé en
# permanence pour détecter les pannes, les ralentissements et les dérives
# de qualité des données. Chez Fortuneo, une panne du pipeline signifie
# que les données de risque ne sont plus à jour, ce qui expose les clients.
#
# En production : les métriques seraient envoyées à AWS CloudWatch pour
# la supervision en temps réel et les alertes automatiques (SNS, PagerDuty).
# =============================================================================

# Importation des bibliothèques standard Python
import json      # Sérialisation des métriques au format JSON
import logging   # Gestion de la journalisation des événements
import logging.handlers   # Gestion de la rotation des fichiers de log
import os        # Opérations sur le système de fichiers
import time      # Mesure du temps d'exécution des étapes
from datetime import datetime   # Manipulation des dates et horodatages
from typing import Any, Dict, Optional   # Annotations de types

# Importation des bibliothèques tierces
import pandas as pd   # Manipulation des DataFrames pour le rapport de métriques

# Récupération du logger principal du pipeline
logger = logging.getLogger(__name__)


def configurer_logging(
    niveau: str = "INFO",
    fichier_log: str = "logs/pipeline.log",
    taille_max: int = 10 * 1024 * 1024,  # 10 Mo par défaut
    nb_fichiers: int = 5,
) -> None:
    """
    Configure le système de journalisation centralisé pour l'ensemble du pipeline.

    Met en place deux handlers de journalisation :
    1. Console (StreamHandler) : affiche les logs en temps réel dans le terminal.
    2. Fichier avec rotation (RotatingFileHandler) : sauvegarde les logs dans un
       fichier, avec rotation automatique quand la taille maximale est atteinte.

    Args:
        niveau: Niveau de journalisation (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fichier_log: Chemin du fichier de log principal.
        taille_max: Taille maximale d'un fichier log avant rotation (en octets).
        nb_fichiers: Nombre de fichiers archivés à conserver avant suppression.
    """
    # Création du dossier de logs s'il n'existe pas encore
    os.makedirs(os.path.dirname(fichier_log) if os.path.dirname(fichier_log) else ".", exist_ok=True)

    # Conversion du niveau de log de chaîne en constante numérique Python
    # Ex: "INFO" → logging.INFO (= 20)
    niveau_numerique = getattr(logging, niveau.upper(), logging.INFO)

    # Définition du format des messages de log
    # %(asctime)s     : date et heure de l'événement
    # %(name)s        : nom du logger (module source)
    # %(levelname)s   : niveau de log (INFO, WARNING, etc.)
    # %(message)s     : message de l'événement
    format_log = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    formateur = logging.Formatter(fmt=format_log, datefmt="%Y-%m-%d %H:%M:%S")

    # --- Configuration du logger racine ---
    # Le logger racine (root) capture tous les messages de tous les sous-loggers
    logger_racine = logging.getLogger()
    logger_racine.setLevel(niveau_numerique)

    # Suppression des handlers existants pour éviter les doublons de messages
    # lors des ré-exécutions du pipeline dans la même session Python
    logger_racine.handlers.clear()

    # --- Handler 1 : Affichage en console (StreamHandler) ---
    # Utile pour le suivi en temps réel lors du développement et des tests
    handler_console = logging.StreamHandler()
    handler_console.setLevel(niveau_numerique)
    handler_console.setFormatter(formateur)
    logger_racine.addHandler(handler_console)

    # --- Handler 2 : Fichier avec rotation automatique ---
    # RotatingFileHandler crée un nouveau fichier quand la taille maximale est atteinte
    # Les anciens fichiers sont archivés avec un suffixe numérique (.log.1, .log.2, ...)
    handler_fichier = logging.handlers.RotatingFileHandler(
        filename=fichier_log,          # Chemin du fichier de log principal
        maxBytes=taille_max,            # Taille maximale avant rotation
        backupCount=nb_fichiers,        # Nombre de fichiers archivés à conserver
        encoding="utf-8",              # Encodage pour supporter les caractères français
    )
    handler_fichier.setLevel(niveau_numerique)
    handler_fichier.setFormatter(formateur)
    logger_racine.addHandler(handler_fichier)

    # Confirmation de la configuration du système de logging
    logging.info(
        f"Système de journalisation configuré | "
        f"Niveau : {niveau} | "
        f"Fichier : {fichier_log}"
    )


class MoniteurPipeline:
    """
    Classe de surveillance et de monitoring du pipeline ETL Fortuneo.

    Enregistre les métriques de chaque étape du pipeline, calcule les
    durées d'exécution, détecte les anomalies de qualité des données
    et génère un rapport de synthèse après chaque exécution.

    En production, cette classe communiquerait avec AWS CloudWatch pour
    publier les métriques, et avec AWS SNS pour envoyer des alertes
    par email ou SMS en cas d'anomalie critique du pipeline.

    Attributs :
        dossier_rapports (str): Dossier de stockage des rapports JSON.
        metriques (dict): Dictionnaire accumulant les métriques du run courant.
        horodatage_debut (float): Timestamp du début d'exécution du pipeline.
    """

    def __init__(self, dossier_rapports: str = "logs/") -> None:
        """
        Initialise le moniteur du pipeline.

        Args:
            dossier_rapports: Chemin du dossier de stockage des rapports de monitoring.
        """
        # Création du dossier de rapports si nécessaire
        os.makedirs(dossier_rapports, exist_ok=True)

        # Stockage du chemin du dossier de rapports
        self.dossier_rapports: str = dossier_rapports

        # Dictionnaire principal de collecte des métriques du pipeline
        # Structure : étape → {métriques spécifiques à l'étape}
        self.metriques: Dict[str, Any] = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),  # Identifiant unique du run
            "date_debut": datetime.now().isoformat(),            # Date de début du run
            "date_fin": None,                                     # Date de fin (renseignée à la clôture)
            "duree_totale_secondes": None,                        # Durée totale du run
            "statut": "en_cours",                                 # Statut global du pipeline
            "etapes": {},                                         # Métriques par étape
            "alertes": [],                                        # Liste des alertes générées
        }

        # Horodatage de démarrage du pipeline pour le calcul de la durée totale
        self.horodatage_debut: float = time.time()

        # Journalisation du démarrage du monitoring
        logger.info(
            f"MoniteurPipeline initialisé | "
            f"Run ID : {self.metriques['run_id']}"
        )

    def demarrer_etape(self, nom_etape: str) -> float:
        """
        Enregistre le démarrage d'une étape du pipeline et retourne le timestamp.

        Args:
            nom_etape: Nom de l'étape démarrée (ex: "extraction", "transformation").

        Returns:
            Timestamp de démarrage de l'étape (utilisé pour calculer la durée).
        """
        # Enregistrement de l'heure de début de l'étape
        horodatage_debut_etape = time.time()

        # Initialisation de l'entrée de l'étape dans le dictionnaire de métriques
        self.metriques["etapes"][nom_etape] = {
            "statut": "en_cours",                       # Statut de l'étape
            "date_debut": datetime.now().isoformat(),   # Horodatage de début
            "date_fin": None,                            # Horodatage de fin (à renseigner)
            "duree_secondes": None,                      # Durée (à calculer à la fin)
        }

        # Journalisation du démarrage de l'étape
        logger.info(f"═══ Démarrage de l'étape : [{nom_etape.upper()}] ═══")

        return horodatage_debut_etape

    def terminer_etape(
        self,
        nom_etape: str,
        horodatage_debut: float,
        succes: bool = True,
        metriques_supplementaires: Optional[Dict] = None,
    ) -> None:
        """
        Enregistre la fin d'une étape et calcule sa durée d'exécution.

        Args:
            nom_etape: Nom de l'étape terminée.
            horodatage_debut: Timestamp de début retourné par demarrer_etape().
            succes: True si l'étape s'est terminée sans erreur, False sinon.
            metriques_supplementaires: Dictionnaire de métriques spécifiques à l'étape.
        """
        # Calcul de la durée d'exécution de l'étape
        duree_secondes = time.time() - horodatage_debut

        # Mise à jour des métriques de l'étape avec les informations de fin
        if nom_etape in self.metriques["etapes"]:
            self.metriques["etapes"][nom_etape].update({
                "statut": "succès" if succes else "échec",  # Statut final de l'étape
                "date_fin": datetime.now().isoformat(),     # Horodatage de fin
                "duree_secondes": round(duree_secondes, 2), # Durée arrondie à 2 décimales
            })

            # Intégration des métriques supplémentaires spécifiques à l'étape
            if metriques_supplementaires:
                self.metriques["etapes"][nom_etape].update(metriques_supplementaires)

        # Journalisation du résultat de l'étape
        statut_emoji = "✓" if succes else "✗"
        logger.info(
            f"═══ Fin de l'étape [{nom_etape.upper()}] | "
            f"Statut : {statut_emoji} | "
            f"Durée : {duree_secondes:.2f}s ═══"
        )

        # Génération d'une alerte si l'étape a échoué
        if not succes:
            self._ajouter_alerte(
                niveau="CRITICAL",
                message=f"L'étape '{nom_etape}' a échoué. Le pipeline est compromis.",
                etape=nom_etape,
            )

        # Alerte si l'étape dépasse un seuil de durée acceptable (10 minutes)
        if duree_secondes > 600:
            self._ajouter_alerte(
                niveau="WARNING",
                message=(
                    f"L'étape '{nom_etape}' a duré {duree_secondes:.0f}s "
                    f"(seuil : 600s). Performance dégradée."
                ),
                etape=nom_etape,
            )

    def enregistrer_metrique_qualite(
        self,
        df: pd.DataFrame,
        nom_etape: str,
    ) -> Dict[str, Any]:
        """
        Calcule et enregistre les métriques de qualité des données d'une étape.

        Les métriques de qualité permettent de détecter les dérives de données
        (data drift), les valeurs manquantes excessives ou les ruptures de charge
        qui pourraient indiquer un problème en amont du pipeline.

        Args:
            df: DataFrame à analyser pour les métriques de qualité.
            nom_etape: Nom de l'étape associée à ces métriques.

        Returns:
            Dictionnaire des métriques de qualité calculées.
        """
        # Calcul des métriques de qualité fondamentales sur le DataFrame
        metriques_qualite = {
            "nb_lignes": len(df),                          # Nombre total de lignes
            "nb_colonnes": len(df.columns),                # Nombre total de colonnes
            "nb_valeurs_nulles": int(df.isnull().sum().sum()),  # Valeurs manquantes
            "taux_completude": float(df.notna().mean().mean()), # Taux de données complètes
            "nb_doublons": int(df.duplicated().sum()),     # Lignes dupliquées
        }

        # Ajout des métriques spécifiques aux données boursières si le DataFrame contient
        # la colonne symbole (structure Fortuneo)
        if "symbole" in df.columns:
            metriques_qualite["nb_symboles"] = int(df["symbole"].nunique())

        if "date" in df.columns:
            metriques_qualite["date_min"] = str(df["date"].min())
            metriques_qualite["date_max"] = str(df["date"].max())

        # Stockage des métriques de qualité dans les métriques de l'étape
        if nom_etape in self.metriques["etapes"]:
            self.metriques["etapes"][nom_etape]["qualite_donnees"] = metriques_qualite

        # Génération d'alertes si la qualité des données est insuffisante
        taux_completude = metriques_qualite["taux_completude"]
        if taux_completude < 0.95:
            self._ajouter_alerte(
                niveau="WARNING",
                message=(
                    f"Taux de complétude insuffisant pour l'étape '{nom_etape}' : "
                    f"{taux_completude:.2%} (seuil : 95%)"
                ),
                etape=nom_etape,
            )

        # Journalisation des métriques de qualité
        logger.info(
            f"Qualité des données [{nom_etape}] : "
            f"{metriques_qualite['nb_lignes']} lignes | "
            f"Complétude : {taux_completude:.2%} | "
            f"Doublons : {metriques_qualite['nb_doublons']}"
        )

        return metriques_qualite

    def _ajouter_alerte(
        self,
        niveau: str,
        message: str,
        etape: Optional[str] = None,
    ) -> None:
        """
        Ajoute une alerte à la liste des alertes du run courant.

        En production, cette méthode enverrait les alertes via :
        - AWS SNS (Simple Notification Service) pour les emails/SMS
        - PagerDuty pour les alertes d'astreinte en cas d'incident critique
        - Slack pour les notifications d'équipe en temps réel

        Args:
            niveau: Niveau de sévérité de l'alerte (INFO, WARNING, CRITICAL).
            message: Description de l'alerte.
            etape: Nom de l'étape concernée par l'alerte (optionnel).
        """
        # Construction de l'objet alerte avec toutes les informations nécessaires
        alerte = {
            "horodatage": datetime.now().isoformat(),   # Date et heure de l'alerte
            "niveau": niveau,                            # Niveau de sévérité
            "message": message,                          # Message descriptif
            "etape": etape,                              # Étape concernée
        }

        # Ajout de l'alerte à la liste du run courant
        self.metriques["alertes"].append(alerte)

        # Journalisation de l'alerte avec le niveau approprié
        if niveau == "CRITICAL":
            logger.critical(f"ALERTE CRITIQUE : {message}")
        elif niveau == "WARNING":
            logger.warning(f"ALERTE : {message}")
        else:
            logger.info(f"INFO : {message}")

    def cloture_pipeline(self, succes_global: bool = True) -> Dict[str, Any]:
        """
        Clôture le monitoring du run courant et génère le rapport final.

        Args:
            succes_global: True si toutes les étapes ont réussi, False sinon.

        Returns:
            Dictionnaire complet des métriques du run.
        """
        # Calcul de la durée totale du pipeline
        duree_totale = time.time() - self.horodatage_debut

        # Mise à jour des métriques globales de clôture
        self.metriques["date_fin"] = datetime.now().isoformat()
        self.metriques["duree_totale_secondes"] = round(duree_totale, 2)
        self.metriques["statut"] = "succès" if succes_global else "échec"
        self.metriques["nb_alertes"] = len(self.metriques["alertes"])
        self.metriques["nb_alertes_critiques"] = sum(
            1 for a in self.metriques["alertes"] if a["niveau"] == "CRITICAL"
        )

        # Sauvegarde du rapport JSON dans le dossier de rapports
        nom_rapport = f"rapport_pipeline_{self.metriques['run_id']}.json"
        chemin_rapport = os.path.join(self.dossier_rapports, nom_rapport)

        with open(chemin_rapport, "w", encoding="utf-8") as fichier_rapport:
            # Sérialisation du dictionnaire de métriques en JSON indenté (lisible)
            json.dump(self.metriques, fichier_rapport, ensure_ascii=False, indent=2)

        # Journalisation du bilan final du pipeline
        statut_emoji = "✅" if succes_global else "❌"
        logger.info(
            f"\n{'═' * 60}\n"
            f"  BILAN DU PIPELINE FORTUNEO\n"
            f"{'═' * 60}\n"
            f"  Statut    : {statut_emoji} {self.metriques['statut'].upper()}\n"
            f"  Durée     : {duree_totale:.1f}s\n"
            f"  Étapes    : {len(self.metriques['etapes'])}\n"
            f"  Alertes   : {self.metriques['nb_alertes']} "
            f"({self.metriques['nb_alertes_critiques']} critiques)\n"
            f"  Rapport   : {chemin_rapport}\n"
            f"{'═' * 60}"
        )

        return self.metriques
