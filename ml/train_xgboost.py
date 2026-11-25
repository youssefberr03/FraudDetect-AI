
from pathlib import Path
from time import perf_counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Importer notre cinquième algorithme.
from xgboost import XGBClassifier

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

# Récupérer le dossier racine FraudDetect.
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODELS_DIR = BASE_DIR / "models" / "xgboost"

REPORTS_DIR = BASE_DIR / "reports" / "xgboost"

# Créer les dossiers de sauvegarde.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Garantir la reproductibilité des expériences.
RANDOM_STATE = 42

# Nombre d'arbres construits successivement.
N_ESTIMATORS = 200

# Taux d'apprentissage.
LEARNING_RATE = 0.1

# Seuil de classification.
THRESHOLD = 0.5


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

def load_dataset(suffix=""):

    # Charger la version du dataset correspondant
    # à la stratégie d'entraînement sélectionnée.

    X_train = pd.read_csv(
        PROCESSED_DIR / f"X_train{suffix}.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / f"y_train{suffix}.csv"
    )["Class"]

    return X_train, y_train


def load_validation_data():

    # Utiliser le même jeu de validation
    # que pour les algorithmes précédents.
    #
    # Le jeu de validation n'est pas rééquilibré.

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    return X_val, y_val


# ==========================================
# 3. CALCUL DU POIDS DES FRAUDES
# ==========================================

def calculate_positive_weight(y_train):

    # Compter les transactions légitimes.
    legitimate = (y_train == 0).sum()

    # Compter les transactions frauduleuses.
    fraud = (y_train == 1).sum()

    # Vérifier que le dataset contient des fraudes.
    if fraud == 0:
        raise ValueError(
            "No fraudulent transactions found."
        )

    # Calculer le rapport entre les deux classes.
    #
    # Ce rapport permet de donner davantage
    # d'importance aux erreurs sur les fraudes.

    positive_weight = legitimate / fraud

    return float(positive_weight)


# ==========================================
# 4. ENTRAINEMENT DU MODELE
# ==========================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    balanced=False
):

    # Par défaut, les deux classes ont
    # le même poids individuel.
    positive_weight = 1.0

    # Si la stratégie Balanced est sélectionnée,
    # calculer automatiquement le poids des fraudes.
    if balanced:

        positive_weight = calculate_positive_weight(
            y_train
        )

    print(
        f"Scale positive weight: "
        f"{positive_weight:.4f}"
    )

    # Créer notre modèle XGBoost.
    #
    # n_estimators :
    # Nombre d'arbres à construire.
    #
    # learning_rate :
    # Contrôle la contribution de chaque arbre.
    #
    # max_depth :
    # Profondeur maximale des arbres.
    #
    # subsample :
    # Utiliser 80 % des observations par itération.
    #
    # colsample_bytree :
    # Utiliser 80 % des variables par arbre.
    #
    # reg_alpha :
    # Régularisation L1 sur les poids des feuilles.
    #
    # reg_lambda :
    # Régularisation L2 sur les poids des feuilles.
    #
    # tree_method="hist" :
    # Utiliser une méthode optimisée par histogrammes.
    #
    # objective="binary:logistic" :
    # Effectuer une classification binaire.

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=positive_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nStarting XGBoost training...")

    # Démarrer le chronomètre.
    start_time = perf_counter()

    # ======================================
    # VERITABLE APPRENTISSAGE DU MODELE
    # ======================================
    #
    # XGBoost construit les arbres successivement.
    #
    # eval_set permet de suivre la fonction
    # de perte sur les données d'entraînement
    # et de validation après chaque arbre.
    #
    # Le jeu de validation ne sert pas ici
    # à ajuster directement les arbres.
    #
    # verbose=False désactive l'affichage
    # des résultats de chaque itération.

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        verbose=False
    )

    # Calculer la durée totale d'entraînement.
    training_time = perf_counter() - start_time

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    # Récupérer le nombre de tours de boosting.
    number_of_trees = (
        model.get_booster().num_boosted_rounds()
    )

    print(f"Number of trees: {number_of_trees}")

    return model, training_time, positive_weight


# ==========================================
# 5. EVALUATION DU MODELE
# ==========================================

def evaluate_model(model, X_val, y_val, name):

    start_time = perf_counter()

    # Récupérer les scores de la classe fraude.
    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # Convertir les scores en classes 0 ou 1.
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

    # Calculer les mêmes métriques
    # que pour les précédents algorithmes.
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

        "Number_of_Trees": (
            model.get_booster().num_boosted_rounds()
        )
    }

    print("\n===== EVALUATION RESULTS =====")

    for metric, value in results.items():

        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")

    return results, matrix


# ==========================================
# 6. MATRICE DE CONFUSION
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

    plt.title(f"XGBoost - {name}")

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"confusion_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 7. SAUVEGARDE DU MODELE
# ==========================================

def save_model(model, name):

    # Sauvegarder le modèle avec Joblib.
    #
    # Ce format sera compatible avec
    # notre futur backend Python.

    model_path = MODELS_DIR / f"{name}.joblib"

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    # XGBoost possède également son propre
    # format de sauvegarde.
    #
    # Le format JSON conserve la structure
    # des arbres et les paramètres du modèle
    # pris en charge par la sauvegarde native.

    model.save_model(
        str(MODELS_DIR / f"{name}.json")
    )

    # Calculer la taille du fichier Joblib.
    model_size = (
        model_path.stat().st_size / (1024 ** 2)
    )

    print(f"\nModel saved: {model_path}")
    print(f"Model size: {model_size:.2f} MB")

    return model_size


# ==========================================
# 8. IMPORTANCE DES VARIABLES
# ==========================================

def save_feature_importance(model, columns, name):

    # Récupérer l'importance des variables
    # calculée par XGBoost.
    #
    # Pour les arbres de boosting, cette mesure
    # repose par défaut sur le gain moyen
    # associé aux séparations utilisant la variable.

    importance = pd.DataFrame({
        "Feature": columns,
        "Importance": model.feature_importances_
    })

    # Trier les variables selon leur importance.
    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n===== TOP 10 IMPORTANT FEATURES =====")

    print(
        importance.head(10).to_string(index=False)
    )

    # Sauvegarder les importances.
    importance.to_csv(
        REPORTS_DIR / f"importance_{name}.csv",
        index=False
    )

    # Sélectionner les 15 principales variables.
    top_features = importance.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
        color="steelblue"
    )

    plt.gca().invert_yaxis()

    plt.title(f"XGBoost Feature Importance - {name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"importance_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 9. COURBES D'APPRENTISSAGE
# ==========================================

def save_learning_curves(model, name):

    # evals_result() retourne l'historique
    # des métriques calculées après chaque arbre.

    history = model.evals_result()

    # validation_0 correspond aux données
    # d'entraînement transmises dans eval_set.

    train_loss = history[
        "validation_0"
    ]["logloss"]

    # validation_1 correspond aux données
    # de validation.

    validation_loss = history[
        "validation_1"
    ]["logloss"]

    # Créer les numéros des itérations.
    iterations = range(
        1,
        len(train_loss) + 1
    )

    # Sauvegarder l'historique dans un CSV.
    learning_history = pd.DataFrame({
        "Iteration": iterations,
        "Training_Loss": train_loss,
        "Validation_Loss": validation_loss
    })

    learning_history.to_csv(
        REPORTS_DIR / f"learning_history_{name}.csv",
        index=False
    )

    # Créer les courbes d'apprentissage.
    plt.figure(figsize=(11, 6))

    plt.plot(
        iterations,
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        iterations,
        validation_loss,
        label="Validation Loss"
    )

    plt.title(f"XGBoost Learning Curves - {name}")

    plt.xlabel("Boosting Iteration")
    plt.ylabel("Log Loss")

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"learning_curves_{name}.png",
        dpi=300
    )

    plt.close()

    print("Learning curves saved successfully!")


# ==========================================
# 10. COMPARAISON DES MODELES
# ==========================================

def compare_models(all_results):

    # Transformer les résultats en DataFrame.
    comparison = pd.DataFrame(all_results)

    # Trier les modèles selon Average Precision.
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

    # Sauvegarder les résultats complets.
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

    ax.set_title("XGBoost - Sampling Comparison")
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
    # GRAPHIQUES TECHNIQUES
    # --------------------------------------

    # Utiliser une boucle pour éviter de répéter
    # trois fois le même code de visualisation.

    technical_metrics = [
        (
            "Training_Time",
            "XGBoost - Training Time",
            "Time (seconds)",
            "training_time.png"
        ),
        (
            "Prediction_Time",
            "XGBoost - Prediction Time",
            "Time (seconds)",
            "prediction_time.png"
        ),
        (
            "Model_Size_MB",
            "XGBoost - Model Size",
            "Size (MB)",
            "model_size.png"
        )
    ]

    for (
        metric,
        title,
        ylabel,
        filename
    ) in technical_metrics:

        ax = comparison.set_index("Model")[
            metric
        ].plot(
            kind="bar",
            figsize=(10, 6)
        )

        ax.set_title(title)
        ax.set_xlabel("Training Strategy")
        ax.set_ylabel(ylabel)

        plt.xticks(rotation=15)
        plt.tight_layout()

        plt.savefig(
            REPORTS_DIR / filename,
            dpi=300
        )

        plt.close()

    # --------------------------------------
    # SELECTION DU MEILLEUR MODELE
    # --------------------------------------

    # La première ligne contient le modèle
    # ayant obtenu la meilleure Average Precision.

    best_model = comparison.iloc[0]

    print("\n===== BEST XGBOOST MODEL =====")

    print(f"Strategy: {best_model['Model']}")

    print(
        f"Average Precision: "
        f"{best_model['Average_Precision']:.6f}"
    )

    print(f"Precision: {best_model['Precision']:.6f}")
    print(f"Recall: {best_model['Recall']:.6f}")
    print(f"F1-score: {best_model['F1']:.6f}")

    # Charger le meilleur modèle entraîné.
    selected_model = joblib.load(
        MODELS_DIR / f"{best_model['Model']}.joblib"
    )

    # Sauvegarder le meilleur modèle
    # dans les deux formats.

    joblib.dump(
        selected_model,
        MODELS_DIR / "best_model.joblib",
        compress=3
    )

    selected_model.save_model(
        str(MODELS_DIR / "best_model.json")
    )


# ==========================================
# 11. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)
    print("FRAUDDETECT AI - XGBoost Training")
    print("=" * 60)

    # Charger le jeu de validation.
    X_val, y_val = load_validation_data()

    # Définir les cinq expériences.
    #
    # balanced=True active scale_pos_weight.
    #
    # Les autres méthodes ne nécessitent
    # pas de pondération supplémentaire
    # pour cette première comparaison.

    experiments = [
        {
            "name": "undersampling",
            "suffix": "_undersampled",
            "balanced": False
        },
        {
            "name": "original",
            "suffix": "",
            "balanced": False
        },
        {
            "name": "balanced",
            "suffix": "",
            "balanced": True
        },
        {
            "name": "oversampling",
            "suffix": "_oversampled",
            "balanced": False
        },
        {
            "name": "smote",
            "suffix": "_smote",
            "balanced": False
        }
    ]

    # Liste contenant les résultats
    # de toutes les expériences.
    all_results = []

    # Entraîner les cinq variantes.
    for index, experiment in enumerate(
        experiments,
        start=1
    ):

        name = experiment["name"]

        print("\n" + "=" * 60)

        print(
            f"EXPERIMENT {index}/{len(experiments)} "
            f"- {name.upper()}"
        )

        print("=" * 60)

        # Etape 1 : charger les données.
        X_train, y_train = load_dataset(
            experiment["suffix"]
        )

        print(f"Training shape: {X_train.shape}")

        print("\nClass distribution:")
        print(y_train.value_counts())

        # Etape 2 : entraîner le modèle.
        (
            model,
            training_time,
            positive_weight
        ) = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            experiment["balanced"]
        )

        # Etape 3 : évaluer le modèle.
        results, matrix = evaluate_model(
            model,
            X_val,
            y_val,
            name
        )

        results["Training_Time"] = training_time

        results["Scale_Pos_Weight"] = positive_weight

        # Etape 4 : sauvegarder le modèle.
        model_size = save_model(
            model,
            name
        )

        results["Model_Size_MB"] = model_size

        # Etape 5 : générer la matrice.
        save_confusion_matrix(
            matrix,
            name
        )

        # Etape 6 : analyser les variables.
        save_feature_importance(
            model,
            X_train.columns,
            name
        )

        # Etape 7 : générer les courbes
        # d'apprentissage.
        save_learning_curves(
            model,
            name
        )

        # Etape 8 : conserver les résultats.
        all_results.append(results)

        # Libérer les objets devenus inutiles.
        del model, X_train, y_train

    # Comparer toutes les variantes.
    compare_models(all_results)

    print("\n" + "=" * 60)
    print("XGBoost training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
