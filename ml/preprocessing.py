
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

# Récupérer le dossier racine FraudDetect.
BASE_DIR = Path(__file__).resolve().parents[1]

# Chemin vers le dataset original.
DATASET_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

# Dossier contenant les données préparées.
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Dossier contenant les objets et modèles sauvegardés.
MODELS_DIR = BASE_DIR / "models"

# Créer les dossiers s'ils n'existent pas.
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Fixer une graine aléatoire pour obtenir
# la même séparation des données à chaque exécution.
RANDOM_STATE = 42


# ==========================================
# 2. CHARGEMENT DU DATASET
# ==========================================

def load_dataset():

    # Vérifier que le fichier existe.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Lire le fichier CSV avec Pandas.
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset loaded successfully!")
    print(f"Original shape: {df.shape}")

    # Retourner le DataFrame.
    return df


# ==========================================
# 3. NETTOYAGE DES DONNEES
# ==========================================

def clean_dataset(df):

    print("\n===== DATA CLEANING =====")

    # Compter les valeurs manquantes.
    missing_values = df.isnull().sum().sum()

    print(f"Missing values: {missing_values}")

    # Notre dataset ne devrait contenir aucune
    # valeur manquante.
    #
    # Si des valeurs manquantes apparaissent,
    # arrêter le programme pour les analyser.
    if missing_values > 0:
        raise ValueError(
            "Missing values detected in dataset."
        )

    # Compter les lignes entièrement identiques.
    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")

    # Supprimer les lignes entièrement identiques.
    #
    # Cela réduit notamment le risque qu'une même
    # observation apparaisse dans le Train Set
    # et dans le Test Set.
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Cleaned shape: {df.shape}")

    # Vérifier la distribution des classes
    # après la suppression des doublons.
    print("\nClass distribution after cleaning:")
    print(df["Class"].value_counts())

    return df


# ==========================================
# 4. SEPARATION DES FEATURES ET DE LA TARGET
# ==========================================

def separate_features_target(df):

    print("\n===== FEATURES AND TARGET =====")

    # X contient toutes les colonnes sauf Class.
    #
    # axis=1 signifie que nous supprimons
    # une colonne et non une ligne.
    X = df.drop("Class", axis=1)

    # y contient uniquement les réponses attendues.
    #
    # 0 : transaction légitime.
    # 1 : transaction frauduleuse.
    y = df["Class"]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nInput features:")
    print(X.columns.tolist())

    return X, y


# ==========================================
# 5. SEPARATION TRAIN / VALIDATION / TEST
# ==========================================

def split_dataset(X, y):

    print("\n===== DATASET SPLITTING =====")

    # Première séparation :
    #
    # 70 % pour l'entraînement.
    # 30 % pour la validation et le test.
    #
    # stratify=y conserve approximativement
    # la proportion des différentes classes.

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Deuxième séparation :
    #
    # Diviser les 30 % restants en deux parties.
    #
    # 50 % de 30 % = 15 % du dataset total.
    #
    # Validation : 15 %
    # Test       : 15 %

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # Afficher les dimensions des trois ensembles.
    print(f"Training set:   {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set:       {X_test.shape}")

    # Vérifier la répartition des fraudes.
    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nValidation class distribution:")
    print(y_val.value_counts())

    print("\nTest class distribution:")
    print(y_test.value_counts())

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ==========================================
# 6. STANDARDISATION DES VARIABLES
# ==========================================

def scale_features(X_train, X_val, X_test):

    print("\n===== FEATURE SCALING =====")

    # Créer des copies indépendantes des DataFrames
    # pour éviter de modifier les objets originaux.
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Sélectionner les variables à standardiser.
    #
    # V1 à V28 restent inchangées dans cette version.
    columns_to_scale = ["Time", "Amount"]

    # Créer un objet StandardScaler.
    scaler = StandardScaler()

    # FIT :
    # Calculer la moyenne et l'écart-type
    # uniquement sur les données d'entraînement.
    #
    # TRANSFORM :
    # Appliquer la formule de standardisation.

    X_train[columns_to_scale] = scaler.fit_transform(
        X_train[columns_to_scale]
    )

    # Appliquer exactement la même transformation
    # aux données de validation.
    #
    # Ne surtout pas recalculer la moyenne
    # et l'écart-type sur le Validation Set.
    X_val[columns_to_scale] = scaler.transform(
        X_val[columns_to_scale]
    )

    # Même principe pour les données de test.
    X_test[columns_to_scale] = scaler.transform(
        X_test[columns_to_scale]
    )

    # Sauvegarder le scaler pour pouvoir transformer
    # les futures transactions de notre API.
    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.joblib"
    )

    print(f"Scaled columns: {columns_to_scale}")

    print("\nTraining statistics after scaling:")

    print(
        X_train[columns_to_scale]
        .describe()
    )

    print("\nScaler saved successfully!")

    return X_train, X_val, X_test


# ==========================================
# 7. SAUVEGARDE DES DONNEES PREPAREES
# ==========================================

def save_datasets(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    print("\n===== SAVING DATASETS =====")

    # Sauvegarder les variables d'entrée.
    #
    # index=False évite d'ajouter une colonne
    # contenant les indices du DataFrame.

    X_train.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False
    )

    X_val.to_csv(
        PROCESSED_DIR / "X_val.csv",
        index=False
    )

    X_test.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False
    )

    # Sauvegarder les réponses attendues.
    y_train.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    y_val.to_csv(
        PROCESSED_DIR / "y_val.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    print("All datasets saved successfully!")
    print(f"Output directory: {PROCESSED_DIR}")


# ==========================================
# 8. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 55)
    print("FRAUDDETECT AI - Data Preprocessing")
    print("=" * 55)

    # Etape 1 : charger les données.
    df = load_dataset()

    # Etape 2 : nettoyer les données.
    df = clean_dataset(df)

    # Etape 3 : séparer les entrées et les réponses.
    X, y = separate_features_target(df)

    # Etape 4 : créer les trois ensembles.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(X, y)

    # Etape 5 : standardiser les variables.
    X_train, X_val, X_test = scale_features(
        X_train,
        X_val,
        X_test
    )

    # Etape 6 : enregistrer les données préparées.
    save_datasets(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    print("\n" + "=" * 55)
    print("Preprocessing completed successfully!")
    print("=" * 55)


# Lancer main() uniquement lorsque ce fichier
# est exécuté directement.
if __name__ == "__main__":
    main()
