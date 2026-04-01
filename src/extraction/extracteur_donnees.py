# =============================================================================
# Module d'extraction des données boursières - Étape Extract du pipeline ETL
# =============================================================================
# Ce module constitue la première étape du pipeline ETL (Extract - Transform - Load).
# Il est responsable de la récupération des cours boursiers historiques depuis
# l'API Yahoo Finance, qui simule les flux de données de marché disponibles
# sur la plateforme de courtage Fortuneo.
#
# Problématique métier : Fortuneo propose à ses clients des milliers d'actions
# disponibles à l'achat. Pour surveiller le risque de leurs portefeuilles,
# le Data Lab doit ingérer quotidiennement les cours de clôture, volumes et
# indicateurs OHLCV (Open, High, Low, Close, Volume) de chaque valeur.
# =============================================================================

# Importation des bibliothèques standard Python
import logging                            # Gestion de la journalisation des événements
import os                                 # Interactions avec le système de fichiers
from datetime import datetime             # Manipulation des dates et heures
from typing import Dict, List, Optional   # Annotations de types pour la lisibilité du code

# Importation des bibliothèques tierces
import pandas as pd     # Manipulation et analyse des données tabulaires
import yfinance as yf   # Accès à l'API Yahoo Finance pour les données de marché

# Récupération du logger configuré au niveau du pipeline principal
# Le point (.) dans __name__ assure la hiérarchie : pipeline -> extraction -> extracteur
logger = logging.getLogger(__name__)


# =============================================================================
# Liste des symboles boursiers suivis par le pipeline Fortuneo
# =============================================================================
# Ces symboles correspondent aux actions françaises du CAC40 les plus échangées
# sur la plateforme Fortuneo. Le suffixe ".PA" indique la cotation sur Euronext Paris.
SYMBOLES_FORTUNEO_PAR_DEFAUT: List[str] = [
    "BNP.PA",   # BNP Paribas - Première banque française par les actifs
    "AIR.PA",   # Airbus - Leader mondial de la construction aéronautique
    "TTE.PA",   # TotalEnergies - Major pétrolière et gazière française
    "SAN.PA",   # Sanofi - Groupe pharmaceutique international
    "OR.PA",    # L'Oréal - Leader mondial des produits cosmétiques
    "MC.PA",    # LVMH - Premier groupe de luxe mondial
    "CS.PA",    # AXA - Groupe d'assurance et de gestion d'actifs
    "DSY.PA",   # Dassault Systèmes - Logiciels 3D et PLM
    "EL.PA",    # EssilorLuxottica - Leader mondial des verres correcteurs
    "DG.PA",    # Vinci - Groupe de concessions et construction
    "KER.PA",   # Kering - Groupe de luxe (Gucci, Balenciaga...)
    "STM.PA",   # STMicroelectronics - Fabricant de semi-conducteurs
]


class ExtracteurDonneesBoursières:
    """
    Classe responsable de l'extraction des données boursières depuis Yahoo Finance.

    Cette classe constitue la couche "Extract" du pipeline ETL.
    Elle encapsule toute la logique de connexion à l'API Yahoo Finance,
    de récupération des données OHLCV et de gestion des erreurs d'extraction.

    En production chez Fortuneo, cette couche serait remplacée par un connecteur
    vers les flux de données Reuters ou Bloomberg, hébergés sur AWS S3.

    Attributs :
        symboles (List[str]): Liste des symboles boursiers à extraire.
        periode (str): Fenêtre temporelle des données historiques (ex: "1y").
        intervalle (str): Granularité des données (ex: "1d" pour quotidien).
    """

    def __init__(
        self,
        symboles: Optional[List[str]] = None,
        periode: str = "1y",
        intervalle: str = "1d",
    ) -> None:
        """
        Initialise l'extracteur avec les paramètres de récupération des données.

        Args:
            symboles: Liste des symboles boursiers. Utilise la liste Fortuneo par défaut si None.
            periode: Durée de l'historique à récupérer. Formats : '1d','5d','1mo','3mo','6mo','1y','2y','5y'.
            intervalle: Fréquence des données. Formats : '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo'.
        """
        # Utilisation de la liste par défaut si aucun symbole n'est fourni par l'utilisateur
        self.symboles: List[str] = symboles if symboles is not None else SYMBOLES_FORTUNEO_PAR_DEFAUT

        # Définition de la fenêtre temporelle pour l'historique des données
        self.periode: str = periode

        # Définition de la granularité temporelle (quotidien, horaire, etc.)
        self.intervalle: str = intervalle

        # Journalisation de l'initialisation pour la traçabilité du pipeline
        logger.info(
            f"ExtracteurDonneesBoursières initialisé | "
            f"Symboles : {len(self.symboles)} | "
            f"Période : {self.periode} | "
            f"Intervalle : {self.intervalle}"
        )

    def extraire_action(self, symbole: str) -> Optional[pd.DataFrame]:
        """
        Extrait les données historiques OHLCV d'une action depuis Yahoo Finance.

        Les données OHLCV contiennent pour chaque période :
        - Open (Ouverture) : prix d'ouverture de la séance
        - High (Plus haut) : cours le plus haut de la séance
        - Low (Plus bas) : cours le plus bas de la séance
        - Close (Clôture) : prix de clôture de la séance
        - Volume : nombre de titres échangés

        Args:
            symbole: Symbole boursier de l'action (ex: "BNP.PA").

        Returns:
            DataFrame pandas contenant les données OHLCV enrichies du symbole,
            ou None si l'extraction échoue (réseau indisponible, symbole invalide).
        """
        try:
            # Journalisation du début de l'extraction pour traçabilité
            logger.info(f"Extraction en cours pour : {symbole}")

            # Utilisation de yf.download() plutôt que ticker.history() car cette méthode
            # est plus robuste face aux changements fréquents de l'API Yahoo Finance.
            # auto_adjust=True applique automatiquement les ajustements pour dividendes et splits.
            # progress=False désactive la barre de progression pour ne pas polluer les logs.
            donnees_brutes: pd.DataFrame = yf.download(
                tickers=symbole,           # Symbole boursier à télécharger
                period=self.periode,       # Fenêtre temporelle (ex: 1 an)
                interval=self.intervalle,  # Granularité (ex: quotidien)
                auto_adjust=True,          # Ajustement automatique pour dividendes et splits
                progress=False,            # Pas de barre de progression dans les logs
            )

            # Vérification que des données ont bien été retournées par l'API
            # Un DataFrame vide indique un symbole invalide ou une absence de cotation
            if donnees_brutes.empty:
                logger.warning(
                    f"Aucune donnée retournée pour {symbole}. "
                    f"Vérifiez le symbole ou la disponibilité de l'API."
                )
                return None

            # yf.download() retourne un DataFrame avec un MultiIndex sur les colonnes
            # quand un seul ticker est demandé — on aplatit les colonnes pour simplifier
            if isinstance(donnees_brutes.columns, pd.MultiIndex):
                # Suppression du niveau "Ticker" du MultiIndex pour obtenir des colonnes simples
                donnees_brutes.columns = donnees_brutes.columns.get_level_values(0)

            # Réinitialisation de l'index pour transformer la date (index) en colonne ordinaire
            # Cette opération facilite les jointures et les filtrages ultérieurs
            donnees_brutes = donnees_brutes.reset_index()

            # Suppression du fuseau horaire sur la colonne Date pour éviter les conflits
            # lors des comparaisons de dates et de la sauvegarde en base de données
            donnees_brutes["Date"] = pd.to_datetime(donnees_brutes["Date"]).dt.tz_localize(None)

            # Ajout du symbole boursier comme colonne pour identifier la source de chaque ligne
            # Cette colonne est essentielle pour les analyses multi-actifs dans le pipeline
            donnees_brutes["symbole"] = symbole

            # Ajout de l'horodatage d'extraction pour garantir la traçabilité des données
            # Permet de savoir quand chaque enregistrement a été ingéré dans le pipeline
            donnees_brutes["date_extraction"] = datetime.now()

            # Sélection et renommage des colonnes utiles pour normaliser le schéma des données
            # On conserve uniquement les colonnes pertinentes pour l'analyse de risque
            colonnes_a_garder = {
                "Date": "date",          # Date de la séance boursière
                "Open": "ouverture",     # Prix d'ouverture de la séance
                "High": "plus_haut",     # Cours le plus haut de la séance
                "Low": "plus_bas",       # Cours le plus bas de la séance
                "Close": "cloture",      # Prix de clôture de la séance (référence principale)
                "Volume": "volume",      # Nombre de titres échangés durant la séance
                "symbole": "symbole",                   # Symbole boursier de l'action
                "date_extraction": "date_extraction",   # Horodatage de l'extraction
            }

            # Application du renommage : on garde uniquement les colonnes du dictionnaire
            donnees_brutes = donnees_brutes[list(colonnes_a_garder.keys())].rename(
                columns=colonnes_a_garder
            )

            # Journalisation du succès de l'extraction avec le volume de données récupéré
            logger.info(
                f"Extraction réussie pour {symbole} : "
                f"{len(donnees_brutes)} enregistrements du "
                f"{donnees_brutes['date'].min().date()} au "
                f"{donnees_brutes['date'].max().date()}"
            )

            return donnees_brutes

        except Exception as erreur:
            # Capture de toute exception imprévue pour éviter l'arrêt du pipeline
            # La journalisation de l'erreur permet le diagnostic sans interruption
            logger.error(
                f"Erreur lors de l'extraction de {symbole} : {type(erreur).__name__} - {str(erreur)}"
            )
            return None

    def extraire_tous_les_symboles(self) -> Dict[str, pd.DataFrame]:
        """
        Extrait les données historiques pour l'ensemble des symboles configurés.

        Cette méthode itère sur tous les symboles de la liste et appelle
        extraire_action() pour chacun d'eux, en gérant les erreurs individuellement
        pour ne pas interrompre le pipeline en cas d'échec partiel.

        Returns:
            Dictionnaire {symbole: DataFrame} contenant les données extraites
            pour chaque symbole. Les symboles en erreur sont absents du dictionnaire.
        """
        # Dictionnaire de résultats qui accumulera les données par symbole
        resultats: Dict[str, pd.DataFrame] = {}

        # Compteur de succès pour le rapport final
        nb_succes: int = 0

        # Compteur d'échecs pour identifier les problèmes d'extraction
        nb_echecs: int = 0

        # Itération sur chaque symbole de la liste configurée
        for index, symbole in enumerate(self.symboles, start=1):
            # Journalisation de la progression pour le suivi en temps réel du pipeline
            logger.info(f"Traitement du symbole {index}/{len(self.symboles)} : {symbole}")

            # Tentative d'extraction des données pour ce symbole
            donnees = self.extraire_action(symbole)

            # Traitement du résultat de l'extraction
            if donnees is not None and not donnees.empty:
                # Stockage du DataFrame dans le dictionnaire résultat
                resultats[symbole] = donnees
                nb_succes += 1
            else:
                # Incrémentation du compteur d'échecs si l'extraction a échoué
                nb_echecs += 1
                logger.warning(f"Symbole ignoré suite à l'échec de l'extraction : {symbole}")

        # Journalisation du bilan global de l'extraction
        logger.info(
            f"Extraction terminée | "
            f"Succès : {nb_succes}/{len(self.symboles)} | "
            f"Échecs : {nb_echecs}/{len(self.symboles)}"
        )

        # Alerte si plus d'un tiers des extractions ont échoué
        taux_echec = nb_echecs / len(self.symboles) if self.symboles else 0
        if taux_echec > 0.33:
            logger.warning(
                f"Taux d'échec élevé détecté : {taux_echec:.1%}. "
                f"Vérifiez la disponibilité de l'API Yahoo Finance."
            )

        return resultats

    def extraire_donnees_combinees(self) -> pd.DataFrame:
        """
        Extrait et consolide les données de tous les symboles dans un DataFrame unique.

        Cette méthode est le point d'entrée principal de la couche Extract.
        Elle retourne un DataFrame normalisé prêt à être transmis à la couche Transform.

        Returns:
            DataFrame consolidé contenant les données de tous les symboles,
            trié par symbole puis par date croissante.
            Retourne un DataFrame vide si aucune extraction n'a réussi.
        """
        # Récupération du dictionnaire de données par symbole
        donnees_par_symbole: Dict[str, pd.DataFrame] = self.extraire_tous_les_symboles()

        # Vérification qu'au moins un symbole a été extrait avec succès
        if not donnees_par_symbole:
            logger.error(
                "Aucune donnée n'a pu être extraite pour aucun symbole. "
                "Le pipeline ETL ne peut pas continuer."
            )
            # Retourne un DataFrame vide pour permettre une gestion propre en aval
            return pd.DataFrame()

        # Concaténation de tous les DataFrames en un seul DataFrame consolidé
        # ignore_index=True réinitialise l'index pour éviter les doublons
        donnees_consolidees: pd.DataFrame = pd.concat(
            donnees_par_symbole.values(),
            ignore_index=True,
        )

        # Tri des données par symbole puis par date pour faciliter les calculs
        # d'indicateurs techniques qui nécessitent des données chronologiques
        donnees_consolidees = donnees_consolidees.sort_values(
            by=["symbole", "date"]
        ).reset_index(drop=True)

        # Journalisation du résumé du DataFrame consolidé pour la traçabilité
        logger.info(
            f"Consolidation terminée : "
            f"{len(donnees_consolidees)} enregistrements au total | "
            f"{donnees_consolidees['symbole'].nunique()} symboles | "
            f"Période : {donnees_consolidees['date'].min().date()} → {donnees_consolidees['date'].max().date()}"
        )

        return donnees_consolidees

    def sauvegarder_donnees_brutes(
        self, donnees: pd.DataFrame, dossier: str = "data/brutes/"
    ) -> str:
        """
        Sauvegarde les données brutes extraites dans un fichier CSV horodaté.

        En production, cette étape correspondrait au dépôt des données brutes
        dans un bucket AWS S3 avant transformation, conformément au pattern
        de data lake "Bronze / Silver / Gold" (couche Bronze).

        Args:
            donnees: DataFrame des données brutes à sauvegarder.
            dossier: Chemin du dossier de destination (simule AWS S3 Bronze).

        Returns:
            Chemin complet du fichier CSV créé.
        """
        # Création du dossier de destination s'il n'existe pas encore
        os.makedirs(dossier, exist_ok=True)

        # Génération d'un nom de fichier horodaté pour identifier chaque extraction
        # Le format YYYYMMDD_HHMMSS garantit l'unicité et le tri chronologique
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        nom_fichier = f"extraction_brute_{horodatage}.csv"
        chemin_complet = os.path.join(dossier, nom_fichier)

        # Sauvegarde du DataFrame dans un fichier CSV sans l'index pandas
        donnees.to_csv(chemin_complet, index=False, encoding="utf-8")

        # Journalisation de la sauvegarde pour la traçabilité du pipeline
        logger.info(
            f"Données brutes sauvegardées : {chemin_complet} "
            f"({len(donnees)} lignes, {os.path.getsize(chemin_complet) / 1024:.1f} Ko)"
        )

        return chemin_complet
