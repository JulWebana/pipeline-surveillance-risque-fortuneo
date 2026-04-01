# =============================================================================
# Module de transformation des données boursières - Étape Transform du pipeline ETL
# =============================================================================
# Ce module constitue la deuxième étape du pipeline ETL (Extract - Transform - Load).
# Il reçoit les données brutes OHLCV extraites et les enrichit avec des indicateurs
# techniques et des métriques de risque utilisés par les analystes de Fortuneo
# pour surveiller les portefeuilles clients.
#
# Problématique métier : les cours bruts ne suffisent pas à évaluer le risque d'un
# portefeuille. Il faut calculer des indicateurs comme la volatilité, le RSI ou les
# Bandes de Bollinger pour détecter des comportements anormaux sur les marchés.
# Ces indicateurs sont ensuite utilisés comme features (variables explicatives)
# par les modèles de Machine Learning de détection d'anomalies.
# =============================================================================

# Importation des bibliothèques standard Python
import logging                  # Gestion de la journalisation des événements du pipeline
from typing import Optional      # Annotation de type pour les valeurs potentiellement absentes

# Importation des bibliothèques tierces de calcul scientifique
import numpy as np   # Calcul numérique vectorisé : racines carrées, logarithmes, etc.
import pandas as pd  # Manipulation des séries temporelles et des DataFrames

# Récupération du logger configuré au niveau du pipeline principal
logger = logging.getLogger(__name__)


class TransformateurDonneesBoursières:
    """
    Classe responsable de la transformation et de l'enrichissement des données boursières.

    Cette classe constitue la couche "Transform" du pipeline ETL.
    Elle calcule les indicateurs techniques et les métriques de risque nécessaires
    à la détection d'anomalies et à l'évaluation du risque de portefeuille.

    Les indicateurs calculés sont classiquement utilisés en analyse financière :
    - Moyennes mobiles (MA) : tendance générale d'un titre
    - RSI : identification des zones de surachat ou survente
    - Bandes de Bollinger : mesure de la volatilité relative au prix
    - VaR historique : perte potentielle maximale sur un horizon donné

    En production, cette couche correspondrait à la transformation
    de la couche Bronze vers la couche Silver dans un data lake AWS.

    Attributs :
        fenetre_ma_courte (int): Fenêtre de la moyenne mobile courte terme (jours).
        fenetre_ma_longue (int): Fenêtre de la moyenne mobile long terme (jours).
        periode_rsi (int): Période de calcul du RSI.
        fenetre_volatilite (int): Fenêtre de calcul de la volatilité historique.
        fenetre_bollinger (int): Fenêtre des Bandes de Bollinger.
        nb_ecarts_types (float): Nombre d'écarts-types pour les Bandes de Bollinger.
    """

    def __init__(
        self,
        fenetre_ma_courte: int = 20,
        fenetre_ma_longue: int = 50,
        periode_rsi: int = 14,
        fenetre_volatilite: int = 30,
        fenetre_bollinger: int = 20,
        nb_ecarts_types: float = 2.0,
    ) -> None:
        """
        Initialise le transformateur avec les paramètres des indicateurs techniques.

        Args:
            fenetre_ma_courte: Nombre de jours pour la moyenne mobile courte (défaut : 20 jours).
            fenetre_ma_longue: Nombre de jours pour la moyenne mobile longue (défaut : 50 jours).
            periode_rsi: Nombre de jours pour le calcul du RSI (défaut : 14 jours).
            fenetre_volatilite: Nombre de jours pour le calcul de la volatilité (défaut : 30 jours).
            fenetre_bollinger: Nombre de jours pour les Bandes de Bollinger (défaut : 20 jours).
            nb_ecarts_types: Multiplicateur de l'écart-type pour les Bandes de Bollinger (défaut : 2).
        """
        # Paramètre de la fenêtre pour la moyenne mobile à court terme (tendance courte)
        self.fenetre_ma_courte: int = fenetre_ma_courte

        # Paramètre de la fenêtre pour la moyenne mobile à long terme (tendance longue)
        self.fenetre_ma_longue: int = fenetre_ma_longue

        # Période de calcul du RSI (Relative Strength Index)
        self.periode_rsi: int = periode_rsi

        # Fenêtre glissante pour estimer la volatilité historique annualisée
        self.fenetre_volatilite: int = fenetre_volatilite

        # Fenêtre glissante pour les Bandes de Bollinger
        self.fenetre_bollinger: int = fenetre_bollinger

        # Nombre d'écarts-types définissant la largeur des Bandes de Bollinger
        self.nb_ecarts_types: float = nb_ecarts_types

        # Journalisation de l'initialisation du transformateur
        logger.info(
            f"TransformateurDonneesBoursières initialisé | "
            f"MA courte : {self.fenetre_ma_courte}j | MA longue : {self.fenetre_ma_longue}j | "
            f"RSI : {self.periode_rsi}j | Volatilité : {self.fenetre_volatilite}j"
        )

    # =========================================================================
    # MÉTHODES DE CALCUL DES INDICATEURS TECHNIQUES
    # =========================================================================

    def _calculer_rendements_logarithmiques(self, groupe: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les rendements logarithmiques journaliers d'une action.

        Le rendement logarithmique est préféré au rendement arithmétique car il est
        symétrique, additif dans le temps et mieux adapté à la modélisation statistique.
        Formule : r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

        Args:
            groupe: DataFrame d'un seul symbole trié par date croissante.

        Returns:
            DataFrame enrichi avec la colonne 'rendement_log'.
        """
        # Calcul du logarithme naturel du prix de clôture pour chaque séance
        # np.log() applique la fonction logarithme élément par élément
        log_prix = np.log(groupe["cloture"])

        # Calcul de la différence première du log : ln(P_t) - ln(P_{t-1})
        # .diff() calcule la différence entre chaque valeur et la précédente
        groupe["rendement_log"] = log_prix.diff()

        return groupe

    def _calculer_moyennes_mobiles(self, groupe: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les moyennes mobiles simples à court et long terme.

        Les moyennes mobiles lissent les fluctuations de prix pour révéler
        la tendance sous-jacente d'un titre. Un croisement de la MA courte
        au-dessus de la MA longue est un signal haussier (Golden Cross).

        Args:
            groupe: DataFrame d'un seul symbole trié par date croissante.

        Returns:
            DataFrame enrichi avec les colonnes 'ma_courte' et 'ma_longue'.
        """
        # Calcul de la moyenne mobile courte (fenêtre de 20 jours par défaut)
        # min_periods=1 permet d'avoir des valeurs dès la première observation
        groupe["ma_courte"] = (
            groupe["cloture"]
            .rolling(window=self.fenetre_ma_courte, min_periods=1)
            .mean()
        )

        # Calcul de la moyenne mobile longue (fenêtre de 50 jours par défaut)
        # Elle représente la tendance de fond sur un horizon plus étendu
        groupe["ma_longue"] = (
            groupe["cloture"]
            .rolling(window=self.fenetre_ma_longue, min_periods=1)
            .mean()
        )

        return groupe

    def _calculer_rsi(self, groupe: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le RSI (Relative Strength Index) sur une période glissante.

        Le RSI est un oscillateur de momentum qui mesure la vitesse et l'amplitude
        des variations de prix. Il oscille entre 0 et 100 :
        - RSI > 70 : zone de surachat (potentiel retournement baissier)
        - RSI < 30 : zone de survente (potentiel retournement haussier)
        - RSI = 50 : neutre

        Args:
            groupe: DataFrame d'un seul symbole trié par date croissante.

        Returns:
            DataFrame enrichi avec la colonne 'rsi'.
        """
        # Calcul de la variation journalière du prix de clôture (delta de prix)
        delta_prix = groupe["cloture"].diff()

        # Séparation des variations positives (hausses) des variations négatives (baisses)
        # clip(lower=0) met à zéro les valeurs négatives pour ne garder que les hausses
        gains = delta_prix.clip(lower=0)

        # -clip(upper=0) transforme les variations négatives en valeurs positives
        pertes = -delta_prix.clip(upper=0)

        # Calcul de la moyenne exponentielle des gains sur la période RSI
        # ewm() (Exponentially Weighted Mean) donne plus de poids aux observations récentes
        moyenne_gains = gains.ewm(
            com=self.periode_rsi - 1,  # com = (période - 1) pour reproduire la formule Wilder
            min_periods=self.periode_rsi,
        ).mean()

        # Calcul de la moyenne exponentielle des pertes sur la même période
        moyenne_pertes = pertes.ewm(
            com=self.periode_rsi - 1,
            min_periods=self.periode_rsi,
        ).mean()

        # Calcul du Relative Strength : rapport entre gains moyens et pertes moyennes
        # On remplace les pertes nulles par 1 pour éviter la division par zéro
        rs = moyenne_gains / moyenne_pertes.replace(0, 1)

        # Transformation du RS en RSI bornée entre 0 et 100
        # Formule standard : RSI = 100 - (100 / (1 + RS))
        groupe["rsi"] = 100 - (100 / (1 + rs))

        return groupe

    def _calculer_bandes_bollinger(self, groupe: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les Bandes de Bollinger pour mesurer la volatilité relative du prix.

        Les Bandes de Bollinger encadrent le prix entre une bande haute et une bande basse,
        définies à partir de la moyenne mobile et d'un multiple de l'écart-type.
        Quand le prix sort des bandes, cela indique une volatilité anormalement élevée,
        signal potentiel d'un événement de marché significatif.

        Args:
            groupe: DataFrame d'un seul symbole trié par date croissante.

        Returns:
            DataFrame enrichi avec les colonnes 'bollinger_haute', 'bollinger_basse'
            et 'bollinger_milieu' (la moyenne mobile centrale).
        """
        # Calcul de la bande médiane (moyenne mobile simple sur la fenêtre Bollinger)
        # Elle constitue la "ligne centrale" des Bandes de Bollinger
        groupe["bollinger_milieu"] = (
            groupe["cloture"]
            .rolling(window=self.fenetre_bollinger, min_periods=1)
            .mean()
        )

        # Calcul de l'écart-type glissant du prix de clôture sur la même fenêtre
        # ddof=0 utilise la formule de la population (et non de l'échantillon)
        ecart_type_glissant = (
            groupe["cloture"]
            .rolling(window=self.fenetre_bollinger, min_periods=1)
            .std(ddof=0)
        )

        # Calcul de la bande supérieure : moyenne + (n * écart-type)
        # Un prix au-delà de cette bande indique une volatilité excessive
        groupe["bollinger_haute"] = (
            groupe["bollinger_milieu"] + self.nb_ecarts_types * ecart_type_glissant
        )

        # Calcul de la bande inférieure : moyenne - (n * écart-type)
        groupe["bollinger_basse"] = (
            groupe["bollinger_milieu"] - self.nb_ecarts_types * ecart_type_glissant
        )

        return groupe

    def _calculer_volatilite(self, groupe: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule la volatilité historique annualisée à partir des rendements logarithmiques.

        La volatilité est l'indicateur clé du risque en finance. Elle mesure la dispersion
        des rendements autour de leur moyenne. Une volatilité élevée indique un titre
        plus risqué. Elle est annualisée en multipliant par racine carrée de 252
        (nombre de jours de bourse dans une année).

        Args:
            groupe: DataFrame d'un seul symbole avec les rendements logarithmiques calculés.

        Returns:
            DataFrame enrichi avec la colonne 'volatilite_annualisee' (en pourcentage).
        """
        # Calcul de l'écart-type glissant des rendements logarithmiques
        # ddof=1 utilise la formule de l'écart-type de l'échantillon (correction de Bessel)
        ecart_type_rendements = (
            groupe["rendement_log"]
            .rolling(window=self.fenetre_volatilite, min_periods=2)
            .std(ddof=1)
        )

        # Annualisation de la volatilité : multiplication par racine carrée de 252
        # 252 est le nombre conventionnel de jours de bourse dans une année civile
        groupe["volatilite_annualisee"] = ecart_type_rendements * np.sqrt(252)

        return groupe

    def _calculer_var_historique(
        self, groupe: pd.DataFrame, niveau_confiance: float = 0.99
    ) -> pd.DataFrame:
        """
        Calcule la VaR (Value at Risk) historique par fenêtre glissante.

        La VaR représente la perte potentielle maximale sur un horizon donné,
        avec un certain niveau de confiance. Par exemple, une VaR à 99% de -3%
        signifie qu'on ne perd pas plus de 3% dans 99% des cas sur la période.

        C'est un indicateur réglementaire clé (Bâle III) que les banques comme
        Fortuneo doivent surveiller pour leurs activités de courtage.

        Args:
            groupe: DataFrame d'un seul symbole avec les rendements logarithmiques calculés.
            niveau_confiance: Niveau de confiance de la VaR (défaut : 0.99 = 99%).

        Returns:
            DataFrame enrichi avec la colonne 'var_historique_99'.
        """
        # Calcul du quantile glissant des rendements pour estimer la VaR historique
        # Le quantile (1 - niveau_confiance) donne la perte dans le pire cas
        # Par exemple pour 99% : on prend le 1er percentile des rendements
        groupe["var_historique_99"] = (
            groupe["rendement_log"]
            .rolling(window=self.fenetre_volatilite, min_periods=2)
            .quantile(1 - niveau_confiance)
        )

        return groupe

    def _calculer_amplitude_journaliere(self, groupe: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule l'amplitude journalière normalisée (ATR simplifié).

        L'amplitude journalière mesure l'écart entre le plus haut et le plus bas
        d'une séance, rapporté au prix de clôture. Un pic d'amplitude indique
        une forte volatilité intra-journalière, souvent associée à des nouvelles
        importantes ou des mouvements spéculatifs.

        Args:
            groupe: DataFrame d'un seul symbole.

        Returns:
            DataFrame enrichi avec la colonne 'amplitude_journaliere'.
        """
        # Calcul de l'écart absolu entre le plus haut et le plus bas de la séance
        ecart_absolu = groupe["plus_haut"] - groupe["plus_bas"]

        # Normalisation par le prix de clôture pour rendre la métrique comparable
        # entre des actions de niveaux de prix très différents (ex : LVMH vs STM)
        groupe["amplitude_journaliere"] = ecart_absolu / groupe["cloture"]

        return groupe

    def _nettoyer_donnees(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame en supprimant les lignes avec des valeurs invalides.

        Les valeurs manquantes (NaN) apparaissent naturellement lors du calcul
        des indicateurs techniques sur fenêtre glissante. Par exemple, une MA à 50 jours
        ne peut pas être calculée pour les 49 premières observations.

        Args:
            df: DataFrame contenant les données enrichies avec des NaN potentiels.

        Returns:
            DataFrame nettoyé sans valeurs manquantes.
        """
        # Comptage des lignes avant nettoyage pour quantifier les données supprimées
        nb_lignes_avant = len(df)

        # Suppression de toutes les lignes contenant au moins une valeur NaN
        # Ces lignes correspondent aux premières observations de chaque symbole
        # où les indicateurs sur fenêtre glissante ne sont pas encore calculables
        df_nettoye = df.dropna().reset_index(drop=True)

        # Calcul du nombre de lignes supprimées pour la journalisation
        nb_lignes_supprimees = nb_lignes_avant - len(df_nettoye)

        # Journalisation du résultat du nettoyage
        logger.info(
            f"Nettoyage des données : {nb_lignes_supprimees} lignes supprimées "
            f"({nb_lignes_avant} → {len(df_nettoye)} lignes conservées)"
        )

        return df_nettoye

    # =========================================================================
    # MÉTHODE PRINCIPALE DE TRANSFORMATION
    # =========================================================================

    def transformer(self, df_brut: pd.DataFrame) -> pd.DataFrame:
        """
        Applique l'ensemble des transformations sur le DataFrame brut.

        C'est le point d'entrée principal de la couche Transform.
        Elle orchestre l'application de tous les indicateurs techniques
        symbole par symbole, en s'assurant que les calculs sur fenêtre glissante
        ne mélangent pas les données de deux symboles différents.

        Args:
            df_brut: DataFrame brut issu de la couche Extract, contenant les
                     colonnes : date, ouverture, plus_haut, plus_bas, cloture, volume, symbole.

        Returns:
            DataFrame enrichi avec tous les indicateurs techniques et métriques de risque,
            prêt à être chargé en base de données et utilisé par les modèles ML.
        """
        # Vérification que le DataFrame d'entrée n'est pas vide
        if df_brut.empty:
            logger.error("Le DataFrame d'entrée est vide. Transformation annulée.")
            return pd.DataFrame()

        # Journalisation du début de la transformation
        logger.info(
            f"Début de la transformation : {len(df_brut)} lignes | "
            f"{df_brut['symbole'].nunique()} symboles"
        )

        # Liste des groupes transformés (un groupe = un symbole boursier)
        groupes_transformes = []

        # Traitement de chaque symbole séparément pour éviter de mélanger
        # les séries temporelles de deux actions différentes dans les calculs glissants
        for symbole in df_brut["symbole"].unique():
            # Extraction du sous-DataFrame correspondant au symbole courant
            groupe = df_brut[df_brut["symbole"] == symbole].copy()

            # Tri chronologique indispensable pour les calculs sur fenêtre glissante
            groupe = groupe.sort_values("date").reset_index(drop=True)

            # Journalisation du traitement du symbole courant
            logger.debug(f"Transformation du symbole : {symbole} ({len(groupe)} lignes)")

            # --- Application séquentielle de chaque transformation ---

            # 1. Calcul des rendements logarithmiques (base de nombreux indicateurs)
            groupe = self._calculer_rendements_logarithmiques(groupe)

            # 2. Calcul des moyennes mobiles (tendance court terme et long terme)
            groupe = self._calculer_moyennes_mobiles(groupe)

            # 3. Calcul du RSI (indicateur de momentum de surachat/survente)
            groupe = self._calculer_rsi(groupe)

            # 4. Calcul des Bandes de Bollinger (volatilité relative)
            groupe = self._calculer_bandes_bollinger(groupe)

            # 5. Calcul de la volatilité historique annualisée (mesure de risque principale)
            groupe = self._calculer_volatilite(groupe)

            # 6. Calcul de la VaR historique à 99% (indicateur réglementaire)
            groupe = self._calculer_var_historique(groupe)

            # 7. Calcul de l'amplitude journalière normalisée (détection de pics intra-jour)
            groupe = self._calculer_amplitude_journaliere(groupe)

            # Ajout du groupe transformé à la liste des résultats
            groupes_transformes.append(groupe)

        # Vérification qu'au moins un groupe a été transformé avec succès
        if not groupes_transformes:
            logger.error("Aucun groupe n'a pu être transformé.")
            return pd.DataFrame()

        # Concaténation de tous les groupes transformés en un DataFrame unique
        df_transforme = pd.concat(groupes_transformes, ignore_index=True)

        # Nettoyage final des valeurs NaN dues aux périodes de démarrage des fenêtres glissantes
        df_transforme = self._nettoyer_donnees(df_transforme)

        # Journalisation du résumé final de la transformation
        logger.info(
            f"Transformation terminée : {len(df_transforme)} lignes | "
            f"Nouvelles colonnes : {[c for c in df_transforme.columns if c not in df_brut.columns]}"
        )

        return df_transforme
