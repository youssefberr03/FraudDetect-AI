
from pathlib import Path
from time import perf_counter

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importer notre algorithme.
from sklearn.ensemble import RandomForestClassifier

# Importer les outils d'optimisation.
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold
)

# Importer les métriques d'évaluation.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

BASELINE_DIR = BASE_DIR / "models" / "random_forest"

MODELS_DIR = (
    BASE_DIR / "models" / "optimized" / "random_forest"
)

REPORTS_DIR = (
    BASE_DIR / "reports" / "optimization" / "random_forest"
)

# Créer automatiquement les dossiers.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Garantir la reproductibilité.
RANDOM_STATE = 42

# Nombre de configurations aléatoires.
N_ITERATIONS = 20

# Nombre de folds de validation croisée.
N_FOLDS = 3

# Seuil de classification actuel.
THRESHOLD = 0.5


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

def load_data():

    # Charger les données d'entraînement originales.
    #
    # Nous ne chargeons pas les datasets SMOTE
    # ou Oversampling, afin de rechercher ici
    # la meilleure pondération des classes.

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    )["Class"]

    # Charger le jeu de validation externe.
    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    # Vérifier la cohérence des données.
    if len(X_train) != len(y_train):
        raise ValueError("Invalid training size.")

    if len(X_val) != len(y_val):
        raise ValueError("Invalid validation size.")

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Feature columns do not match.")

    if X_train.shape[1] != 30:
        raise ValueError("Expected 30 features.")

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Invalid training classes.")

    if set(y_val.unique()) != {0, 1}:
        raise ValueError("Invalid validation classes.")

    print("\n===== DATA LOADED =====")

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    return X_train, y_train, X_val, y_val


# ==========================================
# 3. CHARGEMENT DES MODELES INITIAUX
# ==========================================

def load_baseline_models():

    # Charger les deux modèles déjà entraînés.
    #
    # Nous allons les utiliser comme références
    # pour mesurer les gains de l'optimisation.

    balanced_model = joblib.load(
        BASELINE_DIR / "balanced.joblib"
    )

    original_model = joblib.load(
        BASELINE_DIR / "original.joblib"
    )

    print("\nBaseline models loaded successfully!")

    return {
        "Balanced": balanced_model,
        "Original": original_model
    }


# ==========================================
# 4. ESPACE DE RECHERCHE
# ==========================================

def create_search_space():

    # Définir les valeurs que notre programme
    # pourra sélectionner aléatoirement.
    #
    # RandomizedSearchCV ne testera pas
    # toutes les combinaisons possibles.

    search_space = {

        # Nombre d'arbres dans la forêt.
        "n_estimators": [
            50, 100, 150, 200, 300
        ],

        # Profondeur maximale des arbres.
        # None signifie aucune limite explicite.
        "max_depth": [
            5, 8, 10, 15, 20, None
        ],

        # Nombre minimal d'observations
        # nécessaires pour diviser un noeud.
        "min_samples_split": [
            2, 5, 10, 20
        ],

        # Nombre minimal d'observations
        # nécessaires dans une feuille.
        "min_samples_leaf": [
            1, 2, 4, 8
        ],

        # Nombre de variables candidates
        # à chaque séparation.
        "max_features": [
            "sqrt", "log2", 0.3, 0.5
        ],

        # Méthode de pondération des classes.
        "class_weight": [
            None,
            "balanced",
            "balanced_subsample"
        ],

        # Fonction utilisée pour mesurer
        # la qualité d'une séparation.
        "criterion": [
            "gini",
            "entropy"
        ],

        # Utiliser ou non un échantillonnage
        # avec remise pour construire les arbres.
        "bootstrap": [
            True,
            False
        ]
    }

    return search_space


# ==========================================
# 5. VALIDATION CROISEE
# ==========================================

def create_cross_validation():

    # StratifiedKFold permet de conserver
    # approximativement la proportion
    # de fraudes dans chaque fold.

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    return cv


# ==========================================
# 6. RANDOMIZED SEARCH
# ==========================================

def run_randomized_search(
    X_train,
    y_train,
    cv
):

    # Créer une forêt non entraînée.
    #
    # n_jobs=-1 permet à Random Forest
    # d'utiliser tous les coeurs disponibles.

    estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Définir plusieurs métriques.
    #
    # Average Precision reste notre critère
    # principal pour sélectionner le modèle.

    scoring = {
        "AP": "average_precision",
        "F1": "f1",
        "Precision": "precision",
        "Recall": "recall",
        "ROC_AUC": "roc_auc"
    }

    # Créer l'objet responsable
    # de la recherche d'hyperparamètres.

    search = RandomizedSearchCV(

        # Algorithme à optimiser.
        estimator=estimator,

        # Hyperparamètres à explorer.
        param_distributions=create_search_space(),

        # Nombre de configurations.
        n_iter=N_ITERATIONS,

        # Métriques calculées.
        scoring=scoring,

        # Ne pas réentraîner automatiquement.
        # Nous effectuerons cette étape après
        # comparaison avec les baselines.
        refit=False,

        # Validation croisée stratifiée.
        cv=cv,

        # Exécuter les expériences une par une.
        #
        # Chaque Random Forest utilise déjà
        # tous les coeurs du processeur.
        # Cela limite le parallélisme imbriqué.
        n_jobs=1,

        # Afficher la progression.
        verbose=2,

        random_state=RANDOM_STATE,

        # Arrêter en cas d'erreur réelle.
        error_score="raise",

        # Inutile de conserver les scores
        # sur les folds d'entraînement.
        return_train_score=False
    )

    print("\n===== RANDOMIZED SEARCH =====")

    start_time = perf_counter()

    # ======================================
    # VERITABLE OPTIMISATION DU MODELE
    # ======================================
    #
    # Cette instruction lance automatiquement :
    #
    # 1. La génération des configurations.
    # 2. La validation croisée.
    # 3. Les entraînements.
    # 4. Les prédictions.
    # 5. Le calcul des métriques.

    search.fit(
        X_train,
        y_train
    )

    search_time = (
        perf_counter() - start_time
    )

    print(
        f"\nRandomized Search time: "
        f"{search_time:.2f} seconds"
    )

    # Récupérer tous les résultats.
    results = pd.DataFrame(
        search.cv_results_
    )

    results["Source"] = "Random Search"

    results["Trial"] = [
        f"Trial {i}"
        for i in range(1, len(results) + 1)
    ]

    return results, search_time


# ==========================================
# 7. EVALUATION DES CONFIGURATIONS INITIALES
# ==========================================

def evaluate_baseline_configurations(
    baseline_models,
    X_train,
    y_train,
    cv
):

    print("\n===== BASELINE CROSS VALIDATION =====")

    # Récupérer automatiquement les véritables
    # hyperparamètres de nos anciens modèles.
    #
    # Nous évitons ainsi de devoir les recopier
    # manuellement ou de les deviner.

    parameter_grid = []

    for name, model in baseline_models.items():

        # get_params() retourne les paramètres
        # utilisés lors de la création du modèle.

        params = model.get_params(
            deep=False
        )

        # GridSearchCV exige une liste
        # de valeurs pour chaque paramètre.
        #
        # Nous plaçons une seule valeur par clé :
        # celle du véritable modèle initial.

        grid = {
            key: [value]
            for key, value in params.items()
        }

        parameter_grid.append(grid)

        print(f"\nBaseline: {name}")

        for key, value in params.items():
            print(f"{key}: {value}")

    scoring = {
        "AP": "average_precision",
        "F1": "f1",
        "Precision": "precision",
        "Recall": "recall",
        "ROC_AUC": "roc_auc"
    }

    # Tester exactement les deux configurations
    # historiques sur les mêmes folds.

    baseline_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=parameter_grid,
        scoring=scoring,
        refit=False,
        cv=cv,
        n_jobs=1,
        verbose=2,
        error_score="raise",
        return_train_score=False
    )

    start_time = perf_counter()

    baseline_search.fit(
        X_train,
        y_train
    )

    baseline_time = (
        perf_counter() - start_time
    )

    results = pd.DataFrame(
        baseline_search.cv_results_
    )

    results["Source"] = "Baseline"

    results["Trial"] = [
        f"Baseline {name}"
        for name in baseline_models.keys()
    ]

    print(
        f"\nBaseline CV time: "
        f"{baseline_time:.2f} seconds"
    )

    return results, baseline_time


# ==========================================
# 8. CLASSEMENT DES CONFIGURATIONS
# ==========================================

def rank_configurations(
    random_results,
    baseline_results
):

    # Fusionner les configurations aléatoires
    # et les deux configurations initiales.

    results = pd.concat(
        [
            random_results,
            baseline_results
        ],
        ignore_index=True
    )

    # Classer selon Average Precision moyenne.
    #
    # En cas d'égalité, favoriser la
    # configuration ayant le plus faible
    # écart-type.

    results = results.sort_values(
        by=[
            "mean_test_AP",
            "std_test_AP"
        ],
        ascending=[
            False,
            True
        ]
    ).reset_index(drop=True)

    # Créer un classement global.
    # Les classements internes de chaque
    # objet SearchCV ne sont pas comparables.

    results.insert(
        0,
        "Global_Rank",
        range(1, len(results) + 1)
    )

    columns = [
        "Global_Rank",
        "Trial",
        "mean_test_AP",
        "std_test_AP",
        "mean_test_F1",
        "mean_test_Precision",
        "mean_test_Recall",
        "mean_fit_time"
    ]

    print("\n===== GLOBAL CV RANKING =====")

    print(
        results[columns]
        .round(5)
        .to_string(index=False)
    )

    results.to_csv(
        REPORTS_DIR / "search_results.csv",
        index=False
    )

    # Récupérer la configuration gagnante.
    best_trial = results.iloc[0]

    best_parameters = dict(
        best_trial["params"]
    )

    print("\n===== BEST CONFIGURATION =====")

    print(f"Trial: {best_trial['Trial']}")

    print(
        f"Mean CV AP: "
        f"{best_trial['mean_test_AP']:.6f}"
    )

    print(
        f"Std CV AP: "
        f"{best_trial['std_test_AP']:.6f}"
    )

    for key, value in best_parameters.items():
        print(f"{key}: {value}")

    return results, best_parameters


# ==========================================
# 9. REENTRAINEMENT DU MEILLEUR MODELE
# ==========================================

def train_final_model(
    X_train,
    y_train,
    best_parameters
):

    # Copier les paramètres pour ne pas
    # modifier le dictionnaire original.

    params = best_parameters.copy()

    params.setdefault(
        "random_state",
        RANDOM_STATE
    )

    params["n_jobs"] = -1

    # Créer une nouvelle forêt avec la
    # configuration sélectionnée.

    model = RandomForestClassifier(
        **params
    )

    print("\n===== FINAL MODEL TRAINING =====")

    start_time = perf_counter()

    # Réentraîner sur la totalité du Training Set.
    #
    # Nous n'utilisons toujours pas
    # le Validation Set ou le Test Set.

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        perf_counter() - start_time
    )

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Number of trees: "
        f"{len(model.estimators_)}"
    )

    return model, training_time


# ==========================================
# 10. EVALUATION DES MODELES
# ==========================================

def evaluate_model(
    model,
    X_val,
    y_val,
    name
):

    start_time = perf_counter()

    # Récupérer les scores de la classe fraude.
    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # Appliquer le seuil de classification.
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    prediction_time = (
        perf_counter() - start_time
    )

    # Construire la matrice de confusion.
    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    results = {

        "Model": name,

        "Accuracy": accuracy_score(
            y_val,
            predictions
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

        "True_Positives": int(tp),

        "True_Negatives": int(tn),

        "False_Positives": int(fp),

        "False_Negatives": int(fn),

        "Prediction_Time": prediction_time,

        "Number_of_Trees": len(
            model.estimators_
        )
    }

    print(f"\n===== {name.upper()} =====")

    for metric, value in results.items():

        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")

        else:
            print(f"{metric}: {value}")

    return results, matrix, probabilities


# ==========================================
# 11. SAUVEGARDE DU MODELE
# ==========================================

def save_model(
    model,
    search_results,
    training_time,
    search_time
):

    model_path = (
        MODELS_DIR / "optimized_model.joblib"
    )

    # Sauvegarder les arbres réellement
    # construits pendant l'entraînement.

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    best_trial = search_results.iloc[0]

    # Sauvegarder également la configuration
    # sélectionnée dans un fichier JSON.

    configuration = {

        "parameters": model.get_params(
            deep=False
        ),

        "selected_trial": str(
            best_trial["Trial"]
        ),

        "mean_cv_ap": float(
            best_trial["mean_test_AP"]
        ),

        "std_cv_ap": float(
            best_trial["std_test_AP"]
        ),

        "training_time": training_time,

        "total_search_time": search_time,

        "classification_threshold": THRESHOLD
    }

    with open(
        MODELS_DIR / "best_parameters.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4,
            default=float
        )

    model_size = (
        model_path.stat().st_size
        / (1024 ** 2)
    )

    print(f"\nModel saved: {model_path}")

    print(
        f"Model size: "
        f"{model_size:.4f} MiB"
    )

    return model_size


# ==========================================
# 12. GRAPHIQUES DE LA RECHERCHE
# ==========================================

def plot_search_results(results):

    # --------------------------------------
    # GRAPHIQUE 1 : TOP CONFIGURATIONS
    # --------------------------------------

    top_results = results.head(10)

    colors = [
        "darkorange" if source == "Baseline"
        else "steelblue"
        for source in top_results["Source"]
    ]

    plt.figure(figsize=(12, 7))

    plt.barh(
        top_results["Trial"],
        top_results["mean_test_AP"],
        xerr=top_results["std_test_AP"],
        color=colors,
        capsize=4
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Random Forest - Top 10 Configurations"
    )

    plt.xlabel("Mean Cross-Validation AP")
    plt.ylabel("Configuration")
    plt.xlim(0, 1.05)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "top_configurations.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 2 : PROGRESSION
    # --------------------------------------

    random_results = results[
        results["Source"] == "Random Search"
    ].copy()

    # Extraire le numéro de chaque essai.
    random_results["Trial_Number"] = (
        random_results["Trial"]
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    random_results = random_results.sort_values(
        "Trial_Number"
    )

    random_results["Best_AP_So_Far"] = (
        random_results["mean_test_AP"].cummax()
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        random_results["Trial_Number"],
        random_results["mean_test_AP"],
        marker="o",
        label="Configuration AP"
    )

    plt.plot(
        random_results["Trial_Number"],
        random_results["Best_AP_So_Far"],
        linewidth=2,
        label="Best AP So Far"
    )

    # Ajouter les deux références évaluées
    # sur les mêmes folds de validation croisée.

    baseline_results = results[
        results["Source"] == "Baseline"
    ]

    for _, row in baseline_results.iterrows():

        plt.axhline(
            row["mean_test_AP"],
            linestyle="--",
            label=row["Trial"]
        )

    plt.title(
        "Random Forest - Search Progress"
    )

    plt.xlabel("Trial")
    plt.ylabel("Mean Average Precision")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "search_progress.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 3 : NOMBRE D'ARBRES
    # --------------------------------------

    plot_data = results.copy()

    plot_data["Trees"] = (
        plot_data["params"]
        .apply(lambda p: p["n_estimators"])
    )

    # Convertir None en texte pour éviter
    # que Seaborn ignore cette catégorie.

    plot_data["Class_Weight"] = (
        plot_data["params"]
        .apply(lambda p: str(p["class_weight"]))
    )

    plot_data["Max_Depth"] = (
        plot_data["params"]
        .apply(lambda p: str(p["max_depth"]))
    )

    plt.figure(figsize=(11, 7))

    sns.scatterplot(
        data=plot_data,
        x="Trees",
        y="mean_test_AP",
        hue="Class_Weight",
        style="Source",
        s=110
    )

    plt.title(
        "Number of Trees vs Cross-Validation AP"
    )

    plt.xlabel("Number of Trees")
    plt.ylabel("Mean Average Precision")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "number_of_trees_impact.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 4 : PROFONDEUR DES ARBRES
    # --------------------------------------

    plt.figure(figsize=(11, 7))

    sns.stripplot(
        data=plot_data,
        x="Max_Depth",
        y="mean_test_AP",
        hue="Class_Weight",
        dodge=True,
        size=8
    )

    plt.title(
        "Tree Depth vs Cross-Validation AP"
    )

    plt.xlabel("Maximum Depth")
    plt.ylabel("Mean Average Precision")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "max_depth_impact.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 5 : PONDERATION DES CLASSES
    # --------------------------------------

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=plot_data,
        x="Class_Weight",
        y="mean_test_AP"
    )

    sns.stripplot(
        data=plot_data,
        x="Class_Weight",
        y="mean_test_AP",
        color="black",
        alpha=0.6
    )

    plt.title(
        "Class Weight vs Cross-Validation AP"
    )

    plt.xlabel("Class Weight Strategy")
    plt.ylabel("Mean Average Precision")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "class_weight_impact.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 13. COMPARAISON AVANT / APRES
# ==========================================

def compare_models(
    baseline_models,
    optimized_model,
    X_val,
    y_val,
    training_time,
    model_size
):

    models = {

        "Baseline Balanced":
            baseline_models["Balanced"],

        "Baseline Original":
            baseline_models["Original"],

        "Optimized Random Forest":
            optimized_model
    }

    # Charger les temps d'entraînement
    # des deux anciens modèles.

    baseline_report = pd.read_csv(
        BASE_DIR
        / "reports"
        / "random_forest"
        / "comparison.csv"
    )

    all_results = []
    matrices = {}
    scores = {}

    # Evaluer les trois modèles sur
    # les mêmes transactions de validation.

    for name, model in models.items():

        results, matrix, probabilities = (
            evaluate_model(
                model,
                X_val,
                y_val,
                name
            )
        )

        if name == "Optimized Random Forest":

            results["Training_Time"] = (
                training_time
            )

            results["Model_Size_MB"] = (
                model_size
            )

        else:

            strategy = (
                "balanced"
                if name == "Baseline Balanced"
                else "original"
            )

            baseline_row = baseline_report[
                baseline_report["Model"]
                == strategy
            ].iloc[0]

            results["Training_Time"] = float(
                baseline_row["Training_Time"]
            )

            model_path = (
                BASELINE_DIR
                / f"{strategy}.joblib"
            )

            results["Model_Size_MB"] = (
                model_path.stat().st_size
                / (1024 ** 2)
            )

        all_results.append(results)

        matrices[name] = matrix

        scores[name] = probabilities

    comparison = pd.DataFrame(
        all_results
    )

    comparison.to_csv(
        REPORTS_DIR / "before_after_comparison.csv",
        index=False
    )

    # --------------------------------------
    # GRAPHIQUE 1 : COMPARAISON DES SCORES
    # --------------------------------------

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "Average_Precision"
    ]

    ax = comparison.set_index(
        "Model"
    )[metrics].T.plot(
        kind="bar",
        figsize=(13, 7)
    )

    ax.set_title(
        "Random Forest - Before and After Optimization"
    )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "before_after_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 2 : PRECISION-RECALL
    # --------------------------------------

    plt.figure(figsize=(10, 7))

    for name, probabilities in scores.items():

        precision, recall, _ = (
            precision_recall_curve(
                y_val,
                probabilities
            )
        )

        ap = average_precision_score(
            y_val,
            probabilities
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} - AP: {ap:.4f}"
        )

    # Référence d'un classement non informatif.
    plt.axhline(
        y_val.mean(),
        linestyle="--",
        color="gray",
        label="Fraud prevalence"
    )

    plt.title(
        "Random Forest - Precision Recall Curves"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.xlim(0, 1.02)
    plt.ylim(0, 1.05)

    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "precision_recall_curves.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 3 : MATRICES DE CONFUSION
    # --------------------------------------

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5)
    )

    for ax, (name, matrix) in zip(
        axes,
        matrices.items()
    ):

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Legitimate",
                "Fraud"
            ],
            yticklabels=[
                "Legitimate",
                "Fraud"
            ],
            cbar=False,
            ax=ax
        )

        ax.set_title(name)

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "confusion_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUES TECHNIQUES
    # --------------------------------------

    technical_metrics = [

        (
            "Training_Time",
            "Training Time",
            "Seconds",
            "training_time.png"
        ),

        (
            "Prediction_Time",
            "Prediction Time",
            "Seconds",
            "prediction_time.png"
        ),

        (
            "Model_Size_MB",
            "Model Size",
            "MiB",
            "model_size.png"
        )
    ]

    for metric, title, ylabel, filename in technical_metrics:

        ax = comparison.set_index(
            "Model"
        )[metric].plot(
            kind="bar",
            figsize=(10, 6)
        )

        ax.set_title(
            f"Random Forest - {title}"
        )

        ax.set_ylabel(ylabel)

        plt.xticks(rotation=15)
        plt.tight_layout()

        plt.savefig(
            REPORTS_DIR / filename,
            dpi=300
        )

        plt.close()

    # --------------------------------------
    # BILAN DE L'OPTIMISATION
    # --------------------------------------

    baseline_ap = comparison.loc[
        comparison["Model"] == "Baseline Balanced",
        "Average_Precision"
    ].iloc[0]

    optimized_ap = comparison.loc[
        comparison["Model"] == "Optimized Random Forest",
        "Average_Precision"
    ].iloc[0]

    improvement = optimized_ap - baseline_ap

    print("\n===== OPTIMIZATION SUMMARY =====")

    print(
        f"Baseline AP: "
        f"{baseline_ap:.6f}"
    )

    print(
        f"Optimized AP: "
        f"{optimized_ap:.6f}"
    )

    print(
        f"AP improvement: "
        f"{improvement * 100:+.4f} percentage points"
    )

    if improvement > 0:

        print(
            "Optimization improved validation AP!"
        )

    elif improvement < 0:

        print(
            "The baseline remains better."
        )

    else:

        print(
            "No change in validation AP."
        )

    print(
        comparison[
            [
                "Model",
                "Precision",
                "Recall",
                "F1",
                "Average_Precision"
            ]
        ].round(5).to_string(index=False)
    )

    return comparison


# ==========================================
# 14. IMPORTANCE DES VARIABLES
# ==========================================

def save_feature_importance(
    model,
    columns
):

    importance = pd.DataFrame({

        "Feature": columns,

        "Importance":
            model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    importance.to_csv(
        REPORTS_DIR / "feature_importance.csv",
        index=False
    )

    top_features = importance.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
        color="steelblue"
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Optimized Random Forest - Feature Importance"
    )

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "feature_importance.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 15. ANALYSE DE LA COMPLEXITE DES ARBRES
# ==========================================

def save_tree_statistics(model):

    # Random Forest contient une collection
    # d'arbres dans l'attribut estimators_.
    #
    # Nous allons analyser chaque arbre
    # individuellement.

    statistics = []

    for index, tree in enumerate(
        model.estimators_,
        start=1
    ):

        statistics.append({

            "Tree": index,

            # Profondeur réellement atteinte.
            "Depth": tree.get_depth(),

            # Nombre total de noeuds.
            "Nodes": tree.tree_.node_count,

            # Nombre de feuilles terminales.
            "Leaves": tree.get_n_leaves()
        })

    tree_stats = pd.DataFrame(
        statistics
    )

    tree_stats.to_csv(
        REPORTS_DIR / "tree_statistics.csv",
        index=False
    )

    print("\n===== TREE STATISTICS =====")

    print(
        tree_stats[
            ["Depth", "Nodes", "Leaves"]
        ].describe().round(2)
    )

    # Visualiser la distribution
    # des profondeurs des arbres.

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=tree_stats,
        x="Depth",
        discrete=True
    )

    plt.title(
        "Optimized Random Forest - Tree Depth Distribution"
    )

    plt.xlabel("Tree Depth")
    plt.ylabel("Number of Trees")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "tree_depth_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 16. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)

    print(
        "FRAUDDETECT AI - Random Forest Optimization"
    )

    print("=" * 60)

    # Etape 1 : charger les données.
    X_train, y_train, X_val, y_val = (
        load_data()
    )

    # Etape 2 : charger les anciens modèles.
    baseline_models = load_baseline_models()

    # Etape 3 : préparer la validation croisée.
    cv = create_cross_validation()

    # Etape 4 : rechercher les hyperparamètres.
    random_results, random_time = (
        run_randomized_search(
            X_train,
            y_train,
            cv
        )
    )

    # Sauvegarder les premiers résultats.
    random_results.to_csv(
        REPORTS_DIR / "random_search_results.csv",
        index=False
    )

    # Etape 5 : évaluer les configurations
    # initiales sur les mêmes folds.

    baseline_results, baseline_time = (
        evaluate_baseline_configurations(
            baseline_models,
            X_train,
            y_train,
            cv
        )
    )

    baseline_results.to_csv(
        REPORTS_DIR / "baseline_cv_results.csv",
        index=False
    )

    # Etape 6 : classer toutes les expériences.
    search_results, best_parameters = (
        rank_configurations(
            random_results,
            baseline_results
        )
    )

    # Etape 7 : générer les graphiques.
    plot_search_results(
        search_results
    )

    # Etape 8 : réentraîner le meilleur modèle.
    optimized_model, training_time = (
        train_final_model(
            X_train,
            y_train,
            best_parameters
        )
    )

    # Etape 9 : sauvegarder le modèle.
    total_search_time = (
        random_time + baseline_time
    )

    model_size = save_model(
        optimized_model,
        search_results,
        training_time,
        total_search_time
    )

    # Etape 10 : comparer les performances.
    compare_models(
        baseline_models,
        optimized_model,
        X_val,
        y_val,
        training_time,
        model_size
    )

    # Etape 11 : analyser les variables.
    save_feature_importance(
        optimized_model,
        X_train.columns
    )

    # Etape 12 : analyser les arbres.
    save_tree_statistics(
        optimized_model
    )

    print("\n" + "=" * 60)

    print(
        "Random Forest optimization completed successfully!"
    )

    print(
        f"Total search time: "
        f"{total_search_time:.2f}s"
    )

    print(f"Reports saved in: {REPORTS_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()
