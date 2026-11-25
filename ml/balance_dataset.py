
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

# Récupérer le dossier racine du projet.
BASE_DIR = Path(__file__).resolve().parents[1]

# Dossier contenant les données préparées.
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Dossier contenant les graphiques.
REPORTS_DIR = BASE_DIR / "reports"

# Créer le dossier reports si nécessaire.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Garantir la reproductibilité des expériences.
RANDOM_STATE = 42

# Ratio souhaité entre la classe minoritaire
# et la classe majoritaire après rééquilibrage.
#
# 0.1 signifie une fraude pour 10 transactions
# légitimes.
SAMPLING_RATIO = 0.1


# ==========================================
# 2. CHARGEMENT DES DONNEES D'ENTRAINEMENT
# ==========================================

def load_training_data():

    # Charger uniquement le Training Set.
    #
    # Nous ne modifions jamais les données
    # de validation et de test.

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    # Sélectionner directement la colonne Class
    # afin d'obtenir une Series Pandas.
    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    )["Class"]

    print("\nTraining data loaded successfully!")

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    return X_train, y_train


# ==========================================
# 3. AFFICHAGE DE LA DISTRIBUTION
# ==========================================

def display_distribution(name, y):

    print(f"\n===== {name} =====")

    # Compter les observations de chaque classe.
    counts = y.value_counts()

    print(f"Legitimate transactions: {counts.get(0, 0)}")
    print(f"Fraudulent transactions: {counts.get(1, 0)}")

    print(f"Total transactions: {len(y)}")

    # Calculer le pourcentage de fraudes.
    fraud_percentage = (y == 1).mean() * 100

    print(f"Fraud percentage: {fraud_percentage:.4f}%")


# ==========================================
# 4. RANDOM UNDERSAMPLING
# ==========================================

def apply_undersampling(X_train, y_train):

    # Créer un objet RandomUnderSampler.
    #
    # Cette méthode réduit le nombre
    # de transactions légitimes.
    sampler = RandomUnderSampler(
        sampling_strategy=SAMPLING_RATIO,
        random_state=RANDOM_STATE
    )

    # fit_resample() analyse la distribution
    # des classes puis rééquilibre les données.
    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    display_distribution(
        "RANDOM UNDERSAMPLING",
        y_resampled
    )

    return X_resampled, y_resampled


# ==========================================
# 5. RANDOM OVERSAMPLING
# ==========================================

def apply_oversampling(X_train, y_train):

    # Augmenter le nombre de transactions
    # frauduleuses en dupliquant aléatoirement
    # des observations existantes.
    sampler = RandomOverSampler(
        sampling_strategy=SAMPLING_RATIO,
        random_state=RANDOM_STATE
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    display_distribution(
        "RANDOM OVERSAMPLING",
        y_resampled
    )

    return X_resampled, y_resampled


# ==========================================
# 6. SMOTE
# ==========================================

def apply_smote(X_train, y_train):

    # Créer des transactions frauduleuses
    # synthétiques à partir des observations
    # minoritaires existantes.
    #
    # k_neighbors=5 signifie que SMOTE recherche
    # les cinq voisins minoritaires les plus proches.
    sampler = SMOTE(
        sampling_strategy=SAMPLING_RATIO,
        random_state=RANDOM_STATE,
        k_neighbors=5
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    display_distribution(
        "SMOTE",
        y_resampled
    )

    return X_resampled, y_resampled


# ==========================================
# 7. SAUVEGARDE DES DATASETS
# ==========================================

def save_dataset(X, y, name):

    # Construire les noms des fichiers
    # dynamiquement selon la méthode utilisée.

    X_path = PROCESSED_DIR / f"X_train_{name}.csv"
    y_path = PROCESSED_DIR / f"y_train_{name}.csv"

    # Sauvegarder les variables d'entrée.
    X.to_csv(X_path, index=False)

    # Sauvegarder les réponses attendues.
    y.to_csv(y_path, index=False)

    print(f"\nDataset '{name}' saved successfully!")


# ==========================================
# 8. COMPARAISON GRAPHIQUE
# ==========================================

def compare_distributions(datasets):

    # Créer une liste pour stocker les résultats.
    results = []

    # Parcourir le dictionnaire contenant
    # les différentes versions de y_train.
    for name, y in datasets.items():

        counts = y.value_counts()

        results.append({
            "Method": name,
            "Legitimate": counts.get(0, 0),
            "Fraud": counts.get(1, 0)
        })

    # Transformer les résultats en DataFrame.
    comparison = pd.DataFrame(results)

    print("\n===== SAMPLING COMPARISON =====")
    print(comparison.to_string(index=False))

    # Utiliser les noms des méthodes comme index.
    comparison = comparison.set_index("Method")

    # Créer un graphique à barres comparatif.
    ax = comparison.plot(
        kind="bar",
        figsize=(12, 6),
        color=["steelblue", "firebrick"]
    )

    # Utiliser une échelle logarithmique
    # pour mieux visualiser les petites classes.
    ax.set_yscale("log")

    ax.set_title("Class Balancing Methods Comparison")
    ax.set_xlabel("Sampling Method")
    ax.set_ylabel("Number of Transactions (log scale)")

    plt.xticks(rotation=15)
    plt.tight_layout()

    # Enregistrer le graphique.
    plt.savefig(
        REPORTS_DIR / "balancing_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nComparison chart saved successfully!")


# ==========================================
# 9. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 55)
    print("FRAUDDETECT AI - Class Imbalance Handling")
    print("=" * 55)

    # Etape 1 : charger les données d'entraînement.
    X_train, y_train = load_training_data()

    # Etape 2 : afficher la distribution originale.
    display_distribution("ORIGINAL DATASET", y_train)

    # Etape 3 : appliquer Random Undersampling.
    X_under, y_under = apply_undersampling(
        X_train,
        y_train
    )

    save_dataset(
        X_under,
        y_under,
        "undersampled"
    )

    # Etape 4 : appliquer Random Oversampling.
    X_over, y_over = apply_oversampling(
        X_train,
        y_train
    )

    save_dataset(
        X_over,
        y_over,
        "oversampled"
    )

    # Etape 5 : appliquer SMOTE.
    X_smote, y_smote = apply_smote(
        X_train,
        y_train
    )

    save_dataset(
        X_smote,
        y_smote,
        "smote"
    )

    # Etape 6 : comparer toutes les distributions.
    datasets = {
        "Original": y_train,
        "Undersampling": y_under,
        "Oversampling": y_over,
        "SMOTE": y_smote
    }

    compare_distributions(datasets)

    print("\n" + "=" * 55)
    print("Class balancing completed successfully!")
    print("=" * 55)


# Exécuter la fonction principale uniquement
# lorsque le fichier est lancé directement.
if __name__ == "__main__":
    main()
