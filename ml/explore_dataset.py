
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

# Récupérer le chemin absolu de la racine du projet.
# parents[0] correspond au dossier ml.
# parents[1] correspond au dossier FraudDetect.
BASE_DIR = Path(__file__).resolve().parents[1]

# Définir le chemin vers notre dataset.
DATASET_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

# Définir le dossier dans lequel enregistrer les graphiques.
REPORTS_DIR = BASE_DIR / "reports"

# Créer le dossier reports s'il n'existe pas.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. ANALYSE GENERALE DU DATASET
# ==========================================

def analyze_general(df):

    print("\n===== GENERAL INFORMATION =====")

    # shape retourne le nombre de lignes et de colonnes.
    print("\nDataset dimensions:")
    print(df.shape)

    # Récupérer les noms de toutes les colonnes.
    print("\nColumn names:")
    print(df.columns.tolist())

    # Afficher le type de données de chaque colonne.
    print("\nData types:")
    print(df.dtypes)

    # Compter toutes les valeurs manquantes du dataset.
    print("\nMissing values:")
    print(df.isnull().sum().sum())

    # Compter les lignes entièrement identiques.
    # Nous ne supprimons encore aucune donnée.
    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # Calculer les statistiques descriptives :
    # moyenne, écart-type, minimum, maximum, quartiles...
    print("\nDescriptive statistics:")
    print(df.describe())


# ==========================================
# 3. ANALYSE DES CLASSES
# ==========================================

def analyze_classes(df):

    print("\n===== CLASS DISTRIBUTION =====")

    # Compter les transactions de chaque classe.
    # Class 0 : transaction légitime.
    # Class 1 : transaction frauduleuse.
    class_counts = df["Class"].value_counts()

    # normalize=True retourne les proportions.
    # On multiplie par 100 pour obtenir les pourcentages.
    class_percentages = (
        df["Class"].value_counts(normalize=True) * 100
    )

    print("\nTransaction counts:")
    print(class_counts)

    print("\nTransaction percentages:")
    print(class_percentages)

    # Créer une figure de 8 pouces sur 5.
    plt.figure(figsize=(8, 5))

    # Créer un graphique représentant le nombre
    # de transactions appartenant à chaque classe.
    sns.countplot(data=df, x="Class")

    plt.title("Transaction Class Distribution")
    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")

    # Remplacer les valeurs 0 et 1 par des noms lisibles.
    plt.xticks([0, 1], ["Legitimate", "Fraud"])

    # Ajuster automatiquement les marges du graphique.
    plt.tight_layout()

    # Enregistrer le graphique avec une bonne résolution.
    plt.savefig(
        REPORTS_DIR / "class_distribution.png",
        dpi=300
    )

    # Fermer la figure pour libérer la mémoire.
    plt.close()


# ==========================================
# 4. ANALYSE DES MONTANTS
# ==========================================

def analyze_amounts(df):

    print("\n===== TRANSACTION AMOUNTS =====")

    # Sélectionner uniquement les transactions légitimes.
    legitimate = df[df["Class"] == 0]

    # Sélectionner uniquement les transactions frauduleuses.
    fraud = df[df["Class"] == 1]

    # Afficher les statistiques des montants légitimes.
    print("\nLegitimate transaction amounts:")
    print(legitimate["Amount"].describe())

    # Afficher les statistiques des montants frauduleux.
    print("\nFraudulent transaction amounts:")
    print(fraud["Amount"].describe())

    plt.figure(figsize=(8, 5))

    # Créer une boîte à moustaches.
    # showfliers=False masque les valeurs atypiques
    # sur le graphique, mais ne les supprime pas du dataset.
    sns.boxplot(
        data=df,
        x="Class",
        y="Amount",
        showfliers=False
    )

    plt.title("Transaction Amounts by Class")
    plt.xlabel("Transaction Class")
    plt.ylabel("Transaction Amount")

    plt.xticks([0, 1], ["Legitimate", "Fraud"])

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "amount_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 5. ANALYSE DES CORRELATIONS
# ==========================================

def analyze_correlations(df):

    print("\n===== CORRELATION ANALYSIS =====")

    # Calculer la corrélation de Pearson entre
    # toutes les colonnes numériques du dataset.
    #
    # Notre dataset possède 31 colonnes.
    # Le résultat est donc une matrice de 31 x 31.
    correlation_matrix = df.corr(method="pearson")

    print("\nCorrelation matrix:")
    print(correlation_matrix)

    # Extraire uniquement les corrélations entre
    # chaque variable et notre variable cible Class.
    #
    # On obtient une Series Pandas.
    target_correlations = correlation_matrix["Class"]

    # Supprimer Class de cette Series.
    # Sa corrélation avec elle-même vaut toujours 1
    # puisque Class n'est pas une variable constante.
    target_correlations = target_correlations.drop("Class")

    # Trier les variables selon la valeur absolue
    # de leur coefficient de corrélation.
    #
    # Exemple :
    # -0.8 représente une relation plus forte que +0.2.
    # abs() permet de comparer leur intensité.
    #
    # ascending=False : tri du plus grand au plus petit.
    sorted_features = (
        target_correlations
        .abs()
        .sort_values(ascending=False)
        .index
    )

    # Réorganiser les corrélations selon ce classement,
    # tout en conservant leurs signes positifs ou négatifs.
    sorted_correlations = target_correlations.loc[
        sorted_features
    ]

    print("\nFeatures correlated with fraud:")
    print(sorted_correlations)

    # Afficher uniquement les 10 variables possédant
    # les corrélations les plus fortes avec Class.
    print("\nTop 10 correlated features:")
    print(sorted_correlations.head(10))

    # ------------------------------------------
    # GRAPHIQUE 1 : MATRICE DE CORRELATION
    # ------------------------------------------

    # Une grande figure est nécessaire pour afficher
    # correctement les 31 colonnes du dataset.
    plt.figure(figsize=(16, 12))

    # Créer une heatmap, ou carte thermique.
    #
    # cmap : palette de couleurs.
    # center=0 : couleur neutre pour une corrélation nulle.
    # vmin=-1 : valeur minimale de l'échelle.
    # vmax=1 : valeur maximale de l'échelle.
    # annot=False : ne pas afficher les 961 coefficients.
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False
    )

    plt.title("Transaction Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "correlation_matrix.png",
        dpi=300
    )

    plt.close()

    # ------------------------------------------
    # GRAPHIQUE 2 : CORRELATIONS AVEC LA FRAUDE
    # ------------------------------------------

    plt.figure(figsize=(10, 12))

    # Attribuer une couleur à chaque coefficient.
    # Rouge pour les corrélations positives.
    # Bleu pour les corrélations négatives.
    colors = [
        "red" if value > 0 else "steelblue"
        for value in sorted_correlations
    ]

    # Créer un graphique à barres horizontales.
    #
    # index contient les noms des variables.
    # values contient leurs coefficients de corrélation.
    plt.barh(
        sorted_correlations.index,
        sorted_correlations.values,
        color=colors
    )

    # Placer les variables les plus corrélées en haut.
    plt.gca().invert_yaxis()

    # Ajouter une ligne verticale au niveau de zéro.
    plt.axvline(
        x=0,
        color="black",
        linewidth=1
    )

    plt.title("Feature Correlations with Fraud")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("Features")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "fraud_correlations.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 6. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 50)
    print("FRAUDDETECT AI - Exploratory Data Analysis")
    print("=" * 50)

    # Vérifier que le fichier CSV existe.
    if not DATASET_PATH.exists():
        print("ERROR: Dataset not found.")
        print(f"Expected path: {DATASET_PATH}")
        return

    # Charger le fichier CSV dans un DataFrame Pandas.
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset loaded successfully!")

    # Exécuter les différentes fonctions d'analyse.
    analyze_general(df)

    analyze_classes(df)

    analyze_amounts(df)

    # Exécuter notre nouvelle fonction.
    analyze_correlations(df)

    print("\n" + "=" * 50)
    print("Analysis completed successfully!")
    print(f"Reports saved in: {REPORTS_DIR}")
    print("=" * 50)


# Exécuter main uniquement lorsque ce fichier
# est lancé directement depuis le terminal.
if __name__ == "__main__":
    main()
