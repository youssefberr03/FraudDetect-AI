
from pathlib import Path
from time import perf_counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Importer notre troisième algorithme de Machine Learning.
from sklearn.neighbors import KNeighborsClassifier

# Importer les outils de standardisation.
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Importer les métriques d'évaluation.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODELS_DIR = BASE_DIR / "models" / "knn"

REPORTS_DIR = BASE_DIR / "reports" / "knn"

# Créer les dossiers de sauvegarde.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Nombre de voisins utilisés pour la classification.
N_NEIGHBORS = 5

# Seuil de classification.
THRESHOLD = 0.5

# Nombre de transactions traitées par lot.
BATCH_SIZE = 2000


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

def load_dataset(suffix=""):

    # Charger les variables d'entrée.
    X_train = pd.read_csv(
        PROCESSED_DIR / f"X_train{suffix}.csv"
    )

    # Charger les réponses attendues.
    y_train = pd.read_csv(
        PROCESSED_DIR / f"y_train{suffix}.csv"
    )["Class"]

    return X_train, y_train


def load_validation_data():

    # Charger le jeu de validation non rééquilibré.
    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    return X_val, y_val


# ==========================================
# 3. STANDARDISATION SUPPLEMENTAIRE
# ==========================================

def create_preprocessor(X_original):

    # Dans preprocessing.py, nous avons déjà
    # standardisé les colonnes Time et Amount.
    #
    # KNN utilise des distances mathématiques.
    # Il est donc important que les autres
    # variables aient aussi des échelles similaires.

    pca_columns = [
        f"V{i}" for i in range(1, 29)
    ]

    # ColumnTransformer permet d'appliquer
    # une transformation à certaines colonnes
    # sans modifier les autres.
    #
    # remainder="passthrough" conserve Time
    # et Amount sans les standardiser à nouveau.

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "pca_scaler",
                StandardScaler(),
                pca_columns
            )
        ],
        remainder="passthrough"
    )

    # Calculer les moyennes et écarts-types
    # uniquement sur l'entraînement original.
    preprocessor.fit(X_original)

    print("KNN preprocessor fitted successfully!")

    return preprocessor


# ==========================================
# 4. ENTRAINEMENT DU MODELE
# ==========================================

def train_model(X_train, y_train, preprocessor):

    # Créer un modèle KNN.
    #
    # n_neighbors=5 :
    # Examiner les cinq voisins les plus proches.
    #
    # weights="uniform" :
    # Tous les voisins ont la même importance.
    #
    # metric="euclidean" :
    # Utiliser la distance euclidienne.
    #
    # algorithm="brute" :
    # Effectuer une recherche exhaustive des voisins.
    #
    # n_jobs=-1 :
    # Utiliser les ressources CPU disponibles.

    knn = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        weights="uniform",
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1
    )

    start_time = perf_counter()

    # Transformer les données avant de les
    # transmettre au modèle.
    X_scaled = preprocessor.transform(X_train)

    # ======================================
    # APPRENTISSAGE DU MODELE
    # ======================================
    #
    # Contrairement à Logistic Regression,
    # KNN mémorise les observations et leurs classes.
    #
    # Il ne cherche pas des poids ou des seuils
    # comme les algorithmes précédents.

    knn.fit(X_scaled, y_train)

    # Regrouper le préprocesseur déjà ajusté
    # et le modèle déjà entraîné dans un pipeline.
    #
    # Ne pas appeler fit() sur ce pipeline,
    # car cela réajusterait le préprocesseur.

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", knn)
    ])

    training_time = perf_counter() - start_time

    print(f"Training time: {training_time:.4f} seconds")

    print(f"Number of neighbors: {N_NEIGHBORS}")

    print(f"Stored observations: {len(X_train)}")

    return model, training_time


# ==========================================
# 5. PREDICTIONS PAR LOTS
# ==========================================

def predict_in_batches(model, X_val):

    # KNN peut être coûteux lorsque le dataset
    # contient plusieurs centaines de milliers
    # d'observations.
    #
    # Nous divisons donc les prédictions
    # en plusieurs petits lots pour limiter
    # la consommation de mémoire.

    probabilities = []

    total = len(X_val)

    for start in range(0, total, BATCH_SIZE):

        # Calculer la fin du lot actuel.
        end = min(start + BATCH_SIZE, total)

        # Sélectionner une partie du DataFrame.
        batch = X_val.iloc[start:end]

        # Calculer les scores de fraude.
        batch_probabilities = model.predict_proba(
            batch
        )[:, 1]

        # Ajouter les résultats à notre liste.
        probabilities.append(
            batch_probabilities
        )

        # Afficher la progression.
        progress = (end / total) * 100

        print(
            f"\rPrediction progress: "
            f"{end}/{total} "
            f"({progress:.1f}%)",
            end="",
            flush=True
        )

    print()

    # Regrouper tous les tableaux NumPy
    # dans un seul tableau de scores.
    return np.concatenate(probabilities)


# ==========================================
# 6. EVALUATION DU MODELE
# ==========================================

def evaluate_model(model, X_val, y_val, name):

    start_time = perf_counter()

    # Effectuer les prédictions par lots.
    probabilities = predict_in_batches(
        model,
        X_val
    )

    # Transformer les scores en classes 0 ou 1.
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    prediction_time = perf_counter() - start_time

    # Construire la matrice de confusion.
    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    # Calculer les mêmes métriques que
    # pour nos deux algorithmes précédents.

    results = {
        "Model": name,

        "Accuracy": accuracy_score(
            y_val, predictions
        ),

        "Precision": precision_score(
            y_val,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_val,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_val,
            predictions,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_val,
            probabilities
        ),

        "Average_Precision": average_precision_score(
            y_val,
            probabilities
        ),

        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),

        "Prediction_Time": prediction_time,

        "K": N_NEIGHBORS
    }

    print("\n===== EVALUATION RESULTS =====")

    for metric, value in results.items():

        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")

        else:
            print(f"{metric}: {value}")

    return results, matrix


# ==========================================
# 7. MATRICE DE CONFUSION
# ==========================================

def save_confusion_matrix(matrix, name):

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )

    plt.title(f"KNN - {name}")

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"confusion_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. SAUVEGARDE DU MODELE
# ==========================================

def save_model(model, name):

    model_path = MODELS_DIR / f"{name}.joblib"

    # Sauvegarder le pipeline complet.
    #
    # Il contient :
    # - Le préprocesseur supplémentaire.
    # - Le modèle KNN.
    # - Les observations mémorisées.

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    print(f"\nModel saved: {model_path}")

    # Calculer la taille du modèle sur le disque.
    model_size = (
        model_path.stat().st_size / (1024 ** 2)
    )

    print(f"Model size: {model_size:.2f} MB")

    return model_size


# ==========================================
# 9. COMPARAISON DES MODELES
# ==========================================

def compare_models(all_results):

    comparison = pd.DataFrame(all_results)

    # Classer les modèles selon Average Precision.
    comparison = comparison.sort_values(
        "Average_Precision",
        ascending=False
    )

    columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Average_Precision",
        "Training_Time",
        "Prediction_Time",
        "Model_Size_MB"
    ]

    print("\n===== MODEL COMPARISON =====")

    print(
        comparison[columns]
        .round(4)
        .to_string(index=False)
    )

    # Sauvegarder les résultats dans un CSV.
    comparison.to_csv(
        REPORTS_DIR / "comparison.csv",
        index=False
    )

    # --------------------------------------
    # GRAPHIQUE 1 : PERFORMANCES
    # --------------------------------------

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "Average_Precision"
    ]

    ax = comparison.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(13, 7)
    )

    ax.set_title("KNN - Sampling Comparison")
    ax.set_xlabel("Training Strategy")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "model_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 2 : TEMPS DE PREDICTION
    # --------------------------------------

    ax = comparison.set_index("Model")[
        "Prediction_Time"
    ].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("KNN - Prediction Time Comparison")
    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("Training Strategy")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "prediction_time.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 3 : TAILLE DES MODELES
    # --------------------------------------

    ax = comparison.set_index("Model")[
        "Model_Size_MB"
    ].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("KNN - Model Size Comparison")
    ax.set_ylabel("Size (MB)")
    ax.set_xlabel("Training Strategy")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "model_size.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # SELECTION DU MEILLEUR MODELE
    # --------------------------------------

    best_model = comparison.iloc[0]

    print("\n===== BEST KNN MODEL =====")

    print(f"Strategy: {best_model['Model']}")

    print(
        f"Average Precision: "
        f"{best_model['Average_Precision']:.6f}"
    )

    print(f"F1-score: {best_model['F1']:.6f}")

    selected_model = joblib.load(
        MODELS_DIR / f"{best_model['Model']}.joblib"
    )

    joblib.dump(
        selected_model,
        MODELS_DIR / "best_model.joblib",
        compress=3
    )


# ==========================================
# 10. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)
    print("FRAUDDETECT AI - KNN Training")
    print("=" * 60)

    # Charger le jeu de validation.
    X_val, y_val = load_validation_data()

    # Charger les données originales pour
    # ajuster notre préprocesseur supplémentaire.
    X_original, _ = load_dataset()

    preprocessor = create_preprocessor(
        X_original
    )

    # Libérer les données devenues inutiles.
    del X_original

    # Définir les quatre stratégies.
    #
    # Nous commençons par Undersampling,
    # car son dataset est beaucoup plus petit.

    experiments = [
        {
            "name": "undersampling",
            "suffix": "_undersampled"
        },
        {
            "name": "original",
            "suffix": ""
        },
        {
            "name": "oversampling",
            "suffix": "_oversampled"
        },
        {
            "name": "smote",
            "suffix": "_smote"
        }
    ]

    all_results = []

    # Entraîner les quatre modèles.
    for experiment in experiments:

        name = experiment["name"]

        print("\n" + "=" * 60)
        print(f"TRAINING: {name.upper()}")
        print("=" * 60)

        # Charger le dataset correspondant.
        X_train, y_train = load_dataset(
            experiment["suffix"]
        )

        print(f"Training shape: {X_train.shape}")

        print("\nClass distribution:")
        print(y_train.value_counts())

        # Entraîner KNN.
        model, training_time = train_model(
            X_train,
            y_train,
            preprocessor
        )

        # Evaluer le modèle.
        results, matrix = evaluate_model(
            model,
            X_val,
            y_val,
            name
        )

        results["Training_Time"] = training_time

        # Sauvegarder le modèle et récupérer sa taille.
        model_size = save_model(
            model,
            name
        )

        results["Model_Size_MB"] = model_size

        # Sauvegarder la matrice de confusion.
        save_confusion_matrix(
            matrix,
            name
        )

        # Conserver les résultats.
        all_results.append(results)

        # Libérer les objets avant l'expérience suivante.
        del model, X_train, y_train

    # Comparer les quatre modèles.
    compare_models(all_results)

    print("\n" + "=" * 60)
    print("KNN training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
