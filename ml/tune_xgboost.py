
from pathlib import Path
from time import perf_counter

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importer notre algorithme XGBoost.
from xgboost import XGBClassifier

# Importer les outils de recherche
# d'hyperparamètres et de validation croisée.
from sklearn.model_selection import (
    ParameterSampler,
    StratifiedKFold,
    train_test_split
)

# Importer les distributions statistiques
# utilisées pendant la recherche aléatoire.
from scipy.stats import uniform, loguniform

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

MODELS_DIR = (
    BASE_DIR / "models" / "optimized" / "xgboost"
)

REPORTS_DIR = (
    BASE_DIR / "reports" / "optimization" / "xgboost"
)

BASELINE_DIR = BASE_DIR / "models" / "xgboost"

# Créer les dossiers de sauvegarde.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Garantir la reproductibilité.
RANDOM_STATE = 42

# Nombre de configurations aléatoires.
N_ITERATIONS = 20

# Nombre de folds de validation croisée.
N_FOLDS = 3

# Nombre maximal d'arbres autorisés.
MAX_ESTIMATORS = 800

# Nombre d'itérations sans amélioration
# avant l'arrêt automatique.
EARLY_STOPPING_ROUNDS = 25

# Part réservée à l'Early Stopping
# à l'intérieur de chaque fold d'entraînement.
EARLY_STOPPING_SIZE = 0.15

# Seuil de classification actuel.
THRESHOLD = 0.5


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

def load_data():

    # Charger uniquement les données originales.
    #
    # Nous allons gérer le déséquilibre avec
    # scale_pos_weight et non avec SMOTE.

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    )["Class"]

    # Charger notre jeu de validation externe.
    #
    # Il ne sera pas utilisé pour la recherche
    # des configurations par validation croisée.

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    # Vérifier la cohérence des données.
    if len(X_train) != len(y_train):
        raise ValueError("Invalid training data size.")

    if len(X_val) != len(y_val):
        raise ValueError("Invalid validation data size.")

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Feature columns do not match.")

    if X_train.shape[1] != 30:
        raise ValueError("Expected 30 input features.")

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
# 3. ESPACE DE RECHERCHE
# ==========================================

def create_search_space():

    # Chaque clé correspond à un hyperparamètre.
    #
    # Les listes représentent des valeurs discrètes.
    #
    # uniform() représente une distribution uniforme.
    #
    # loguniform() est particulièrement utile pour
    # explorer des valeurs sur plusieurs échelles.

    search_space = {

        "max_depth": [3, 4, 5, 6, 8],

        "learning_rate": loguniform(
            0.01, 0.2
        ),

        "min_child_weight": [1, 3, 5, 10],

        "subsample": uniform(
            0.6, 0.4
        ),

        "colsample_bytree": uniform(
            0.6, 0.4
        ),

        "gamma": [0, 0.1, 0.5, 1, 3],

        "reg_alpha": [
            0, 0.01, 0.1, 0.5, 1, 5
        ],

        "reg_lambda": loguniform(
            0.5, 10
        ),

        # Multiplicateur appliqué au rapport
        # entre transactions légitimes et fraudes.
        #
        # 1.0 correspond au ratio complet.
        # 0.1 correspond à 10 % de ce ratio.

        "weight_multiplier": [
            0.01, 0.05, 0.1, 0.25,
            0.5, 0.75, 1.0
        ]
    }

    return search_space


# ==========================================
# 4. CALCUL DU POIDS DES FRAUDES
# ==========================================

def calculate_positive_weight(
    y_train,
    multiplier
):

    legitimate = int((y_train == 0).sum())
    fraud = int((y_train == 1).sum())

    if fraud == 0:
        raise ValueError("No fraud samples found.")

    # Calculer le ratio uniquement à partir
    # des observations utilisées pour apprendre.

    ratio = legitimate / fraud

    positive_weight = ratio * multiplier

    return float(positive_weight)


# ==========================================
# 5. CREATION D'UN MODELE XGBOOST
# ==========================================

def create_model(
    parameters,
    y_train,
    n_estimators=MAX_ESTIMATORS,
    early_stopping=True
):

    # Copier le dictionnaire pour ne pas
    # modifier la configuration originale.
    params = parameters.copy()

    # Extraire notre multiplicateur personnalisé.
    #
    # XGBoost ne possède pas directement
    # un hyperparamètre weight_multiplier.

    multiplier = params.pop(
        "weight_multiplier"
    )

    # Calculer le poids réel des fraudes.
    positive_weight = calculate_positive_weight(
        y_train,
        multiplier
    )

    # Définir les paramètres communs.
    model_params = {
        **params,

        "n_estimators": n_estimators,

        "scale_pos_weight": positive_weight,

        "objective": "binary:logistic",

        # Métrique suivie par Early Stopping.
        "eval_metric": "aucpr",

        "tree_method": "hist",

        "random_state": RANDOM_STATE,

        "n_jobs": -1
    }

    # Activer Early Stopping uniquement
    # pendant la validation croisée.
    if early_stopping:

        model_params["early_stopping_rounds"] = (
            EARLY_STOPPING_ROUNDS
        )

    model = XGBClassifier(
        **model_params
    )

    return model


# ==========================================
# 6. PREPARATION DES FOLDS
# ==========================================

def create_cv_splits(X_train, y_train):

    # StratifiedKFold conserve approximativement
    # les proportions de chaque classe.

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    splits = []

    for fold, (train_idx, score_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1
    ):

        # Diviser les indices d'entraînement
        # du fold en deux sous-ensembles :
        #
        # fit_idx : apprentissage des arbres.
        # stop_idx : suivi de l'Early Stopping.
        #
        # score_idx reste indépendant et permet
        # d'évaluer la configuration.

        fit_idx, stop_idx = train_test_split(
            train_idx,
            test_size=EARLY_STOPPING_SIZE,
            stratify=y_train.iloc[train_idx],
            random_state=RANDOM_STATE + fold
        )

        splits.append({
            "fit": fit_idx,
            "stop": stop_idx,
            "score": score_idx
        })

        print(
            f"Fold {fold} | "
            f"Fit: {len(fit_idx)} | "
            f"Early stopping: {len(stop_idx)} | "
            f"Evaluation: {len(score_idx)}"
        )

    return splits


# ==========================================
# 7. EVALUATION D'UNE CONFIGURATION
# ==========================================

def evaluate_configuration(
    parameters,
    X_train,
    y_train,
    splits
):

    fold_scores = []
    best_iterations = []

    start_time = perf_counter()

    # Evaluer la même configuration
    # sur les trois folds.

    for fold_number, split in enumerate(
        splits,
        start=1
    ):

        # Données utilisées pour apprendre.
        X_fit = X_train.iloc[split["fit"]]
        y_fit = y_train.iloc[split["fit"]]

        # Données utilisées pour Early Stopping.
        X_stop = X_train.iloc[split["stop"]]
        y_stop = y_train.iloc[split["stop"]]

        # Données indépendantes utilisées
        # pour mesurer les performances.
        X_score = X_train.iloc[split["score"]]
        y_score = y_train.iloc[split["score"]]

        # Créer un nouveau modèle indépendant
        # pour chaque fold.

        model = create_model(
            parameters,
            y_fit,
            early_stopping=True
        )

        # ==================================
        # VERITABLE APPRENTISSAGE DU MODELE
        # ==================================
        #
        # X_stop permet de déterminer
        # quand arrêter la construction
        # des arbres.

        model.fit(
            X_fit,
            y_fit,
            eval_set=[
                (X_stop, y_stop)
            ],
            verbose=False
        )

        # predict_proba utilise automatiquement
        # la meilleure itération sélectionnée
        # par Early Stopping.

        probabilities = model.predict_proba(
            X_score
        )[:, 1]

        # Calculer notre métrique principale
        # sur les observations indépendantes.

        ap = average_precision_score(
            y_score,
            probabilities
        )

        # best_iteration commence à zéro.
        #
        # Ajouter 1 pour obtenir le nombre
        # réel de tours de boosting retenus.

        best_rounds = int(
            model.best_iteration + 1
        )

        fold_scores.append(
            float(ap)
        )

        best_iterations.append(
            best_rounds
        )

        print(
            f"  Fold {fold_number}/{N_FOLDS} | "
            f"AP: {ap:.6f} | "
            f"Best trees: {best_rounds}"
        )

    elapsed_time = perf_counter() - start_time

    # Calculer la performance moyenne.
    mean_ap = float(
        np.mean(fold_scores)
    )

    # Mesurer la variabilité des résultats.
    std_ap = float(
        np.std(fold_scores)
    )

    # Utiliser la médiane du nombre optimal
    # d'arbres obtenu sur les trois folds.
    #
    # La médiane est moins sensible
    # aux valeurs extrêmes.

    recommended_trees = int(
        np.median(best_iterations)
    )

    results = {
        **parameters,

        "Mean_AP": mean_ap,

        "Std_AP": std_ap,

        "Recommended_Trees": recommended_trees,

        "Search_Time": elapsed_time
    }

    # Conserver les résultats de chaque fold.
    for index, (score, trees) in enumerate(
        zip(fold_scores, best_iterations),
        start=1
    ):

        results[f"Fold_{index}_AP"] = score
        results[f"Fold_{index}_Trees"] = trees

    return results


# ==========================================
# 8. RANDOMIZED SEARCH
# ==========================================

def run_randomized_search(
    X_train,
    y_train
):

    search_space = create_search_space()

    # ParameterSampler génère automatiquement
    # des combinaisons aléatoires.

    configurations = list(
        ParameterSampler(
            param_distributions=search_space,
            n_iter=N_ITERATIONS,
            random_state=RANDOM_STATE
        )
    )

    # Préparer les mêmes partitions pour
    # toutes les configurations.
    #
    # Cela rend leur comparaison plus cohérente.

    print("\n===== CROSS VALIDATION SPLITS =====")

    splits = create_cv_splits(
        X_train,
        y_train
    )

    all_results = []

    print("\n===== RANDOMIZED SEARCH =====")

    search_start_time = perf_counter()

    # Tester successivement les configurations.

    for index, parameters in enumerate(
        configurations,
        start=1
    ):

        print("\n" + "=" * 60)

        print(
            f"CONFIGURATION "
            f"{index}/{N_ITERATIONS}"
        )

        print("=" * 60)

        for parameter, value in parameters.items():

            if isinstance(value, float):

                print(
                    f"{parameter}: {value:.6f}"
                )

            else:

                print(
                    f"{parameter}: {value}"
                )

        # Evaluer la configuration
        # avec la validation croisée.

        results = evaluate_configuration(
            parameters,
            X_train,
            y_train,
            splits
        )

        results["Trial"] = index

        all_results.append(results)

        print(
            f"Mean AP: "
            f"{results['Mean_AP']:.6f}"
        )

        print(
            f"Std AP: "
            f"{results['Std_AP']:.6f}"
        )

        print(
            f"Recommended trees: "
            f"{results['Recommended_Trees']}"
        )

        # Sauvegarder les résultats
        # après chaque configuration.
        #
        # Les expériences déjà terminées
        # restent disponibles si le programme
        # est interrompu.

        pd.DataFrame(all_results).to_csv(
            REPORTS_DIR / "search_results.csv",
            index=False
        )

    total_time = (
        perf_counter() - search_start_time
    )

    # Classer toutes les configurations.
    search_results = pd.DataFrame(
        all_results
    )

    search_results = search_results.sort_values(
        by=["Mean_AP", "Std_AP"],
        ascending=[False, True]
    ).reset_index(drop=True)

    search_results.to_csv(
        REPORTS_DIR / "search_results.csv",
        index=False
    )

    print("\n===== RANDOMIZED SEARCH COMPLETED =====")

    print(
        f"Total search time: "
        f"{total_time:.2f} seconds"
    )

    print(
        search_results[
            [
                "Trial",
                "Mean_AP",
                "Std_AP",
                "Recommended_Trees"
            ]
        ].head(10).to_string(index=False)
    )

    # Récupérer la meilleure configuration.

    best_trial = search_results.iloc[0]

    best_index = int(
        best_trial["Trial"]
    ) - 1

    best_parameters = configurations[
        best_index
    ]

    best_n_estimators = int(
        best_trial["Recommended_Trees"]
    )

    return (
        search_results,
        best_parameters,
        best_n_estimators
    )


# ==========================================
# 9. REENTRAINEMENT DU MEILLEUR MODELE
# ==========================================

def train_final_model(
    X_train,
    y_train,
    best_parameters,
    best_n_estimators
):

    print("\n===== FINAL MODEL TRAINING =====")

    # Nous connaissons maintenant :
    #
    # - Les meilleurs hyperparamètres.
    # - Le nombre d'arbres recommandé.
    #
    # Nous pouvons utiliser la totalité
    # du Training Set pour apprendre.

    model = create_model(
        best_parameters,
        y_train,
        n_estimators=best_n_estimators,
        early_stopping=False
    )

    start_time = perf_counter()

    # Aucun Early Stopping ici.
    #
    # Le nombre d'arbres a déjà été estimé
    # grâce aux expériences précédentes.

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        perf_counter() - start_time
    )

    print(
        f"Number of trees: "
        f"{best_n_estimators}"
    )

    print(
        f"Training time: "
        f"{training_time:.4f}s"
    )

    return model, training_time


# ==========================================
# 10. EVALUATION SUR LA VALIDATION
# ==========================================

def evaluate_model(
    model,
    X_val,
    y_val,
    name
):

    start_time = perf_counter()

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    prediction_time = (
        perf_counter() - start_time
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

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

        "True_Positives": int(tp),

        "True_Negatives": int(tn),

        "False_Positives": int(fp),

        "False_Negatives": int(fn),

        "Prediction_Time": prediction_time
    }

    print(f"\n===== {name.upper()} =====")

    for metric, value in results.items():

        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")

        else:
            print(f"{metric}: {value}")

    return results, matrix, probabilities


# ==========================================
# 11. SAUVEGARDE DU MODELE OPTIMISE
# ==========================================

def save_model(
    model,
    best_parameters,
    best_n_estimators,
    search_results,
    training_time
):

    model_path = (
        MODELS_DIR / "optimized_model.joblib"
    )

    # Sauvegarder avec Joblib.
    joblib.dump(
        model,
        model_path,
        compress=3
    )

    # Sauvegarder dans le format natif XGBoost.
    model.save_model(
        str(
            MODELS_DIR / "optimized_model.json"
        )
    )

    # Sauvegarder également les hyperparamètres
    # sous une forme lisible.

    configuration = {
        "parameters": best_parameters,

        "n_estimators": best_n_estimators,

        "scale_pos_weight": float(
            model.get_params()["scale_pos_weight"]
        ),

        "mean_cv_ap": float(
            search_results.iloc[0]["Mean_AP"]
        ),

        "std_cv_ap": float(
            search_results.iloc[0]["Std_AP"]
        ),

        "training_time": training_time,

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
    print(f"Model size: {model_size:.4f} MiB")

    return model_size


# ==========================================
# 12. GRAPHIQUES DE LA RECHERCHE
# ==========================================

def plot_search_results(search_results):

    # --------------------------------------
    # GRAPHIQUE 1 : TOP 10 CONFIGURATIONS
    # --------------------------------------

    top_results = search_results.head(10)

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_results["Trial"].astype(str),
        top_results["Mean_AP"],
        xerr=top_results["Std_AP"],
        color="steelblue",
        capsize=4
    )

    plt.gca().invert_yaxis()

    plt.title(
        "XGBoost - Top 10 Configurations"
    )

    plt.xlabel("Mean Cross-Validation AP")
    plt.ylabel("Trial")
    plt.xlim(0, 1.05)

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "top_configurations.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 2 : PROGRESSION DE LA RECHERCHE
    # --------------------------------------

    chronological = search_results.sort_values(
        "Trial"
    ).copy()

    # cummax retourne le meilleur score
    # rencontré jusqu'à la configuration actuelle.

    chronological["Best_AP_So_Far"] = (
        chronological["Mean_AP"].cummax()
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        chronological["Trial"],
        chronological["Mean_AP"],
        marker="o",
        label="Configuration AP"
    )

    plt.plot(
        chronological["Trial"],
        chronological["Best_AP_So_Far"],
        linewidth=2,
        label="Best AP So Far"
    )

    plt.title(
        "XGBoost - Search Progress"
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
    # GRAPHIQUE 3 : LEARNING RATE
    # --------------------------------------

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=search_results,
        x="learning_rate",
        y="Mean_AP",
        hue="max_depth",
        size="Recommended_Trees",
        sizes=(50, 250)
    )

    plt.xscale("log")

    plt.title(
        "Learning Rate vs Cross-Validation AP"
    )

    plt.xlabel("Learning Rate")
    plt.ylabel("Mean Average Precision")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "learning_rate_impact.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 4 : POIDS DES FRAUDES
    # --------------------------------------

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=search_results,
        x="weight_multiplier",
        y="Mean_AP",
        hue="max_depth",
        size="Recommended_Trees",
        sizes=(50, 250)
    )

    plt.xscale("log")

    plt.title(
        "Fraud Weight vs Cross-Validation AP"
    )

    plt.xlabel("Positive Weight Multiplier")
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

def compare_with_baseline(
    optimized_model,
    X_val,
    y_val,
    training_time,
    model_size
):

    # Charger notre modèle initial.
    baseline_model = joblib.load(
        BASELINE_DIR / "balanced.joblib"
    )

    # Réévaluer le modèle initial et le nouveau
    # modèle sur les mêmes observations.

    (
        baseline_results,
        baseline_matrix,
        baseline_probabilities
    ) = evaluate_model(
        baseline_model,
        X_val,
        y_val,
        "Baseline XGBoost"
    )

    (
        optimized_results,
        optimized_matrix,
        optimized_probabilities
    ) = evaluate_model(
        optimized_model,
        X_val,
        y_val,
        "Optimized XGBoost"
    )

    # Récupérer les informations techniques
    # de notre benchmark initial.

    baseline_csv = (
        BASE_DIR
        / "reports"
        / "xgboost"
        / "comparison.csv"
    )

    baseline_report = pd.read_csv(
        baseline_csv
    )

    baseline_row = baseline_report[
        baseline_report["Model"] == "balanced"
    ].iloc[0]

    baseline_results["Training_Time"] = float(
        baseline_row["Training_Time"]
    )

    baseline_results["Model_Size_MB"] = (
        (BASELINE_DIR / "balanced.joblib")
        .stat().st_size / (1024 ** 2)
    )

    optimized_results["Training_Time"] = (
        training_time
    )

    optimized_results["Model_Size_MB"] = (
        model_size
    )

    comparison = pd.DataFrame([
        baseline_results,
        optimized_results
    ])

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

    plot_data = comparison.set_index(
        "Model"
    )[metrics].T

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 7)
    )

    ax.set_title(
        "XGBoost - Before and After Optimization"
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
    # GRAPHIQUE 2 : PRECISION-RECALL CURVES
    # --------------------------------------

    plt.figure(figsize=(9, 7))

    # Calculer les courbes en faisant varier
    # tous les seuils de classification possibles.

    for name, probabilities, results in [

        (
            "Baseline",
            baseline_probabilities,
            baseline_results
        ),

        (
            "Optimized",
            optimized_probabilities,
            optimized_results
        )
    ]:

        precision, recall, _ = (
            precision_recall_curve(
                y_val,
                probabilities
            )
        )

        ap = results["Average_Precision"]

        plt.plot(
            recall,
            precision,
            label=f"{name} - AP: {ap:.4f}"
        )

    # La proportion de fraudes représente
    # la référence théorique d'un classement
    # aléatoire non informatif.

    prevalence = float(y_val.mean())

    plt.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label="Fraud prevalence"
    )

    plt.title(
        "XGBoost - Precision Recall Curves"
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
        1, 2,
        figsize=(13, 5)
    )

    matrices = [
        ("Baseline", baseline_matrix),
        ("Optimized", optimized_matrix)
    ]

    for ax, (name, matrix) in zip(
        axes,
        matrices
    ):

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax,
            cbar=False
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
    # CALCUL DES GAINS
    # --------------------------------------

    improvement = (
        optimized_results["Average_Precision"]
        - baseline_results["Average_Precision"]
    )

    print("\n===== OPTIMIZATION SUMMARY =====")

    print(
        f"Baseline AP: "
        f"{baseline_results['Average_Precision']:.6f}"
    )

    print(
        f"Optimized AP: "
        f"{optimized_results['Average_Precision']:.6f}"
    )

    print(
        f"AP improvement: "
        f"{improvement * 100:+.4f} percentage points"
    )

    if improvement > 0:

        print(
            "Optimization improved validation AP!"
        )

    else:

        print(
            "The baseline remains competitive."
        )

    # Le modèle initial n'est jamais écrasé,
    # même si le modèle optimisé est meilleur.

    return comparison


# ==========================================
# 14. IMPORTANCE DES VARIABLES
# ==========================================

def save_feature_importance(model, columns):

    importance = pd.DataFrame({
        "Feature": columns,

        "Importance": model.feature_importances_
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
        "Optimized XGBoost - Feature Importance"
    )

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "feature_importance.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 15. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)

    print("FRAUDDETECT AI - XGBoost Optimization")

    print("=" * 60)

    # Etape 1 : charger les données.

    X_train, y_train, X_val, y_val = (
        load_data()
    )

    # Etape 2 : rechercher les meilleurs
    # hyperparamètres.

    (
        search_results,
        best_parameters,
        best_n_estimators
    ) = run_randomized_search(
        X_train,
        y_train
    )

    print("\n===== BEST PARAMETERS =====")

    for parameter, value in best_parameters.items():

        print(f"{parameter}: {value}")

    print(
        f"Best number of trees: "
        f"{best_n_estimators}"
    )

    # Etape 3 : générer les graphiques
    # de recherche.

    plot_search_results(
        search_results
    )

    # Etape 4 : réentraîner le meilleur modèle
    # sur la totalité du Training Set.

    optimized_model, training_time = (
        train_final_model(
            X_train,
            y_train,
            best_parameters,
            best_n_estimators
        )
    )

    # Etape 5 : sauvegarder le modèle.

    model_size = save_model(
        optimized_model,
        best_parameters,
        best_n_estimators,
        search_results,
        training_time
    )

    # Etape 6 : comparer le modèle optimisé
    # au modèle initial.

    compare_with_baseline(
        optimized_model,
        X_val,
        y_val,
        training_time,
        model_size
    )

    # Etape 7 : analyser les variables.

    save_feature_importance(
        optimized_model,
        X_train.columns
    )

    print("\n" + "=" * 60)

    print("XGBoost optimization completed successfully!")

    print(f"Reports saved in: {REPORTS_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()
