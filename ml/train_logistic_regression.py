
from pathlib import Path
from time import perf_counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Importer notre premier algorithme de Machine Learning.
from sklearn.linear_model import LogisticRegression

# Importer les métriques nécessaires à l'évaluation.
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

# Récupérer le dossier racine du projet.
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Créer des dossiers spécifiques à la régression
# logistique pour organiser les résultats.
MODELS_DIR = BASE_DIR / "models" / "logistic_regression"

REPORTS_DIR = BASE_DIR / "reports" / "logistic_regression"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Seuil utilisé pour classifier les transactions.
THRESHOLD = 0.5


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

def load_dataset(suffix=""):

    # Le suffixe permet de sélectionner la version
    # du dataset que nous souhaitons utiliser.
    #
    # Exemple :
    # suffix=""             -> X_train.csv
    # suffix="_smote"       -> X_train_smote.csv
    # suffix="_oversampled" -> X_train_oversampled.csv

    X_train = pd.read_csv(
        PROCESSED_DIR / f"X_train{suffix}.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / f"y_train{suffix}.csv"
    )["Class"]

    return X_train, y_train


def load_validation_data():

    # Charger le jeu de validation original.
    #
    # Il ne doit jamais être rééquilibré,
    # car il doit conserver une distribution
    # représentative des transactions.

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    return X_val, y_val


# ==========================================
# 3. ENTRAINEMENT DU MODELE
# ==========================================

def train_model(X_train, y_train, class_weight=None):

    # Créer un modèle de régression logistique.
    #
    # max_iter :
    # Nombre maximal d'itérations d'optimisation.
    #
    # solver :
    # Algorithme mathématique utilisé pour optimiser
    # les paramètres de la régression logistique.
    #
    # class_weight :
    # None -> toutes les classes ont le même poids.
    # balanced -> donner davantage d'importance
    # aux observations de la classe minoritaire.
    #
    # C :
    # Inverse de la force de régularisation.
    # Une petite valeur impose une régularisation
    # plus importante.

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight=class_weight,
        C=1.0,
        random_state=RANDOM_STATE
    )

    # Démarrer le chronomètre.
    start_time = perf_counter()

    # ======================================
    # VERITABLE APPRENTISSAGE DU MODELE
    # ======================================
    #
    # Le modèle apprend les poids et le biais
    # à partir des données d'entraînement
    # et des véritables réponses.

    model.fit(X_train, y_train)

    # Calculer la durée de l'entraînement.
    training_time = perf_counter() - start_time

    print(f"Training time: {training_time:.4f} seconds")

    # Afficher le nombre d'itérations réalisées.
    print(f"Iterations: {model.n_iter_[0]}")

    return model, training_time


# ==========================================
# 4. EVALUATION DU MODELE
# ==========================================

def evaluate_model(model, X_val, y_val, name):

    # Démarrer le chronomètre des prédictions.
    start_time = perf_counter()

    # predict_proba() retourne deux scores :
    #
    # Colonne 0 : classe légitime.
    # Colonne 1 : classe frauduleuse.
    #
    # Nous récupérons uniquement le score
    # associé à la classe frauduleuse.

    probabilities = model.predict_proba(X_val)[:, 1]

    # Transformer les scores en classes.
    #
    # Score >= 0.5 -> fraude.
    # Score < 0.5  -> transaction légitime.
    #
    # astype(int) transforme True et False
    # en 1 et 0.

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    prediction_time = perf_counter() - start_time

    # Calculer la matrice de confusion.
    #
    # labels=[0, 1] garantit l'ordre des classes.

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    # Extraire les quatre résultats de la matrice.
    tn, fp, fn, tp = matrix.ravel()

    # Calculer toutes les métriques.
    #
    # zero_division=0 évite un avertissement
    # lorsqu'aucune fraude n'est prédite.

    results = {
        "Model": name,

        "Accuracy": accuracy_score(
            y_val, predictions
        ),

        "Precision": precision_score(
            y_val, predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_val, predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_val, predictions,
            zero_division=0
        ),

        # ROC-AUC et Average Precision utilisent
        # les scores continus et non les classes
        # finales 0 et 1.

        "ROC_AUC": roc_auc_score(
            y_val, probabilities
        ),

        "Average_Precision": average_precision_score(
            y_val, probabilities
        ),

        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),

        "Prediction_Time": prediction_time
    }

    # Afficher les performances dans le terminal.
    print("\n===== EVALUATION RESULTS =====")

    for metric, value in results.items():

        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")

    return results, matrix


# ==========================================
# 5. MATRICE DE CONFUSION
# ==========================================

def save_confusion_matrix(matrix, name):

    plt.figure(figsize=(7, 5))

    # annot=True affiche les nombres dans les cases.
    # fmt="d" affiche des nombres entiers.
    # cmap="Blues" définit les couleurs.

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )

    plt.title(f"Confusion Matrix - {name}")

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"confusion_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 6. SAUVEGARDE DU MODELE
# ==========================================

def save_model(model, name):

    # Sauvegarder l'objet entraîné avec Joblib.
    #
    # Nous pourrons le recharger plus tard
    # sans avoir besoin de refaire l'entraînement.

    model_path = MODELS_DIR / f"{name}.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved: {model_path}")


# ==========================================
# 7. SAUVEGARDE DES POIDS APPRIS
# ==========================================

def save_coefficients(model, columns, name):

    # coef_ contient les poids appris par le modèle.
    # intercept_ contient le biais.

    coefficients = pd.DataFrame({
        "Feature": columns,
        "Coefficient": model.coef_[0]
    })

    # Calculer la valeur absolue des coefficients.
    coefficients["Absolute_Coefficient"] = (
        coefficients["Coefficient"].abs()
    )

    # Trier selon la magnitude des poids.
    coefficients = coefficients.sort_values(
        "Absolute_Coefficient",
        ascending=False
    )

    # Sauvegarder les coefficients dans un CSV.
    coefficients.to_csv(
        REPORTS_DIR / f"coefficients_{name}.csv",
        index=False
    )

    print(f"Model bias: {model.intercept_[0]:.6f}")


# ==========================================
# 8. COMPARAISON DES MODELES
# ==========================================

def compare_models(all_results):

    # Transformer la liste des résultats
    # en un tableau Pandas.

    comparison = pd.DataFrame(all_results)

    # Trier les modèles selon Average Precision.
    #
    # Cette métrique est particulièrement utile
    # pour notre dataset fortement déséquilibré.

    comparison = comparison.sort_values(
        "Average_Precision",
        ascending=False
    )

    print("\n===== MODEL COMPARISON =====")

    columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Average_Precision",
        "Training_Time"
    ]

    print(
        comparison[columns]
        .round(4)
        .to_string(index=False)
    )

    # Sauvegarder tous les résultats.
    comparison.to_csv(
        REPORTS_DIR / "comparison.csv",
        index=False
    )

    # Comparer graphiquement les métriques.
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

    ax.set_title(
        "Logistic Regression - Sampling Comparison"
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("Training Strategy")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "model_comparison.png",
        dpi=300
    )

    plt.close()

    # Identifier le meilleur modèle selon
    # la métrique de sélection choisie.
    best_model = comparison.iloc[0]

    print("\n===== BEST LOGISTIC REGRESSION =====")

    print(f"Strategy: {best_model['Model']}")

    print(
        f"Average Precision: "
        f"{best_model['Average_Precision']:.6f}"
    )

    print(f"F1-score: {best_model['F1']:.6f}")

    # Copier le meilleur modèle sous un nom commun.
    selected_model = joblib.load(
        MODELS_DIR / f"{best_model['Model']}.joblib"
    )

    joblib.dump(
        selected_model,
        MODELS_DIR / "best_model.joblib"
    )


# ==========================================
# 9. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)
    print("FRAUDDETECT AI - Logistic Regression Training")
    print("=" * 60)

    # Charger une seule fois le jeu de validation.
    X_val, y_val = load_validation_data()

    # Définir les cinq expériences.
    #
    # name : nom de l'expérience.
    # suffix : version des données d'entraînement.
    # class_weight : stratégie de pondération.

    experiments = [
        {
            "name": "original",
            "suffix": "",
            "class_weight": None
        },
        {
            "name": "balanced",
            "suffix": "",
            "class_weight": "balanced"
        },
        {
            "name": "undersampling",
            "suffix": "_undersampled",
            "class_weight": None
        },
        {
            "name": "oversampling",
            "suffix": "_oversampled",
            "class_weight": None
        },
        {
            "name": "smote",
            "suffix": "_smote",
            "class_weight": None
        }
    ]

    # Liste qui contiendra les résultats
    # de nos cinq expériences.
    all_results = []

    # Entraîner chaque modèle indépendamment.
    for experiment in experiments:

        name = experiment["name"]

        print("\n" + "=" * 60)
        print(f"TRAINING: {name.upper()}")
        print("=" * 60)

        # Etape 1 : charger le dataset correspondant.
        X_train, y_train = load_dataset(
            experiment["suffix"]
        )

        print(f"Training shape: {X_train.shape}")

        print("\nClass distribution:")
        print(y_train.value_counts())

        # Etape 2 : entraîner le modèle.
        model, training_time = train_model(
            X_train,
            y_train,
            experiment["class_weight"]
        )

        # Etape 3 : évaluer le modèle.
        results, matrix = evaluate_model(
            model,
            X_val,
            y_val,
            name
        )

        # Ajouter la durée d'entraînement.
        results["Training_Time"] = training_time

        # Etape 4 : sauvegarder le modèle.
        save_model(model, name)

        # Etape 5 : sauvegarder la matrice.
        save_confusion_matrix(matrix, name)

        # Etape 6 : sauvegarder les poids appris.
        save_coefficients(
            model,
            X_train.columns,
            name
        )

        # Etape 7 : conserver les résultats.
        all_results.append(results)

    # Comparer les cinq modèles entraînés.
    compare_models(all_results)

    print("\n" + "=" * 60)
    print("Logistic Regression training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
