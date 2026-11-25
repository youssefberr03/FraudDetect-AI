
from pathlib import Path
from time import perf_counter
from copy import deepcopy

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Notre réseau de neurones.
from sklearn.neural_network import MLPClassifier

# Outils de préparation des données.
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Gestion du déséquilibre des classes.
from imblearn.over_sampling import RandomOverSampler

# Validation croisée.
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

# Métriques d'évaluation.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    log_loss
)


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

BASELINE_DIR = (
    BASE_DIR / "models" / "neural_network"
)

MODELS_DIR = (
    BASE_DIR / "models" / "optimized"
    / "neural_network"
)

REPORTS_DIR = (
    BASE_DIR / "reports" / "optimization"
    / "neural_network"
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Garantir la reproductibilité.
RANDOM_STATE = 42

# Nombre de folds.
N_FOLDS = 3

# Nombre maximal d'epochs.
MAX_EPOCHS = 25

# Nombre d'epochs sans amélioration
# avant d'arrêter l'entraînement.
PATIENCE = 5

# Part du fold d'entraînement réservée
# à l'Early Stopping.
EARLY_STOPPING_SIZE = 0.15

# Ratio fraude / légitime après oversampling.
SAMPLING_RATIO = 0.1

# Taille des mini-batches.
BATCH_SIZE = 256

# Seuil de classification actuel.
THRESHOLD = 0.5


# ==========================================
# 2. CONFIGURATIONS A TESTER
# ==========================================

# Chaque dictionnaire représente une
# expérience indépendante.

CONFIGURATIONS = [

    {
        "Trial": 1,
        "Architecture": (32, 16),
        "Learning_Rate": 0.001,
        "Alpha": 0.001,
        "Sampling": "original"
    },

    {
        "Trial": 2,
        "Architecture": (32, 16),
        "Learning_Rate": 0.0005,
        "Alpha": 0.01,
        "Sampling": "oversampling"
    },

    {
        "Trial": 3,
        "Architecture": (64, 32),
        "Learning_Rate": 0.001,
        "Alpha": 0.001,
        "Sampling": "oversampling"
    },

    {
        "Trial": 4,
        "Architecture": (64, 32),
        "Learning_Rate": 0.0005,
        "Alpha": 0.01,
        "Sampling": "original"
    },

    {
        "Trial": 5,
        "Architecture": (64, 32, 16),
        "Learning_Rate": 0.0005,
        "Alpha": 0.001,
        "Sampling": "oversampling"
    },

    {
        "Trial": 6,
        "Architecture": (64, 32, 16),
        "Learning_Rate": 0.001,
        "Alpha": 0.0001,
        "Sampling": "original"
    }
]


# ==========================================
# 3. CHARGEMENT DES DONNEES
# ==========================================

def load_data():

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    )["Class"]

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    # Vérifier les dimensions.
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
# 4. CREATION DU RESEAU DE NEURONES
# ==========================================

def create_model(configuration):

    # Construire un réseau non entraîné.
    #
    # partial_fit() sera utilisé pour réaliser
    # un passage sur les données à chaque appel.

    model = MLPClassifier(

        hidden_layer_sizes=(
            configuration["Architecture"]
        ),

        activation="relu",

        solver="adam",

        # Régularisation L2.
        alpha=configuration["Alpha"],

        # Vitesse d'apprentissage.
        learning_rate_init=(
            configuration["Learning_Rate"]
        ),

        batch_size=BATCH_SIZE,

        max_iter=1,

        shuffle=True,

        random_state=RANDOM_STATE,

        # Nous programmons notre propre
        # mécanisme d'Early Stopping.
        early_stopping=False,

        # Eviter l'arrêt interne du solveur.
        n_iter_no_change=MAX_EPOCHS + 1
    )

    return model


# ==========================================
# 5. REEQUILIBRAGE DES DONNEES
# ==========================================

def apply_sampling(X_train, y_train, strategy):

    # Aucun rééquilibrage.
    if strategy == "original":

        return X_train, y_train

    # Dupliquer certaines transactions
    # frauduleuses pour augmenter leur
    # représentation pendant l'apprentissage.
    if strategy == "oversampling":

        sampler = RandomOverSampler(

            sampling_strategy=SAMPLING_RATIO,

            random_state=RANDOM_STATE
        )

        X_resampled, y_resampled = (
            sampler.fit_resample(
                X_train,
                y_train
            )
        )

        return X_resampled, y_resampled

    raise ValueError(
        f"Unknown sampling strategy: {strategy}"
    )


# ==========================================
# 6. PREPARATION DES FOLDS
# ==========================================

def create_cv_splits(X_train, y_train):

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

        # Diviser le fold d'entraînement
        # en deux parties :
        #
        # fit_idx  : apprentissage.
        # stop_idx : Early Stopping.
        #
        # score_idx : évaluation indépendante.

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
# 7. ENTRAINEMENT AVEC EARLY STOPPING
# ==========================================

def train_with_early_stopping(

    model,

    X_fit,

    y_fit,

    X_stop,

    y_stop
):

    best_ap = -np.inf

    best_epoch = 0

    best_model = None

    epochs_without_improvement = 0

    history = []

    # Entraîner progressivement le réseau.
    for epoch in range(1, MAX_EPOCHS + 1):

        # ==================================
        # APPRENTISSAGE D'UNE EPOCH
        # ==================================
        #
        # partial_fit conserve les poids
        # précédemment appris et poursuit
        # l'entraînement.

        model.partial_fit(

            X_fit,

            y_fit,

            classes=np.array([0, 1])
        )

        # Calculer les scores sur les
        # données d'Early Stopping.
        stop_probabilities = (
            model.predict_proba(X_stop)[:, 1]
        )

        # Appliquer le seuil de classification.
        stop_predictions = (
            stop_probabilities >= THRESHOLD
        ).astype(int)

        # Calculer l'Average Precision.
        current_ap = average_precision_score(
            y_stop,
            stop_probabilities
        )

        current_f1 = f1_score(

            y_stop,

            stop_predictions,

            zero_division=0
        )

        validation_loss = log_loss(

            y_stop,

            stop_probabilities,

            labels=[0, 1]
        )

        # model.loss_ représente la fonction
        # objectif calculée pendant
        # l'entraînement du réseau.

        history.append({

            "Epoch": epoch,

            "Training_Loss": float(
                model.loss_
            ),

            "Validation_Loss": validation_loss,

            "Average_Precision": current_ap,

            "F1": current_f1
        })

        # ==================================
        # MODEL CHECKPOINTING
        # ==================================

        if current_ap > best_ap:

            best_ap = current_ap

            best_epoch = epoch

            # Sauvegarder une véritable copie
            # du réseau avec ses meilleurs poids.
            best_model = deepcopy(model)

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        print(

            f"    Epoch {epoch:02d}/{MAX_EPOCHS} | "

            f"Loss: {model.loss_:.6f} | "

            f"AP: {current_ap:.6f} | "

            f"Best AP: {best_ap:.6f}"
        )

        # ==================================
        # EARLY STOPPING
        # ==================================

        if epochs_without_improvement >= PATIENCE:

            print(
                f"    Early stopping at epoch {epoch}"
            )

            break

    # Retourner le meilleur réseau et non
    # celui de la dernière epoch.

    return (

        best_model,

        best_epoch,

        best_ap,

        pd.DataFrame(history)
    )


# ==========================================
# 8. EVALUATION D'UNE CONFIGURATION
# ==========================================

def evaluate_configuration(

    configuration,

    X_train,

    y_train,

    splits
):

    fold_scores = []

    fold_epochs = []

    all_histories = []

    start_time = perf_counter()

    for fold_number, split in enumerate(
        splits,
        start=1
    ):

        print(
            f"\n  ===== FOLD "
            f"{fold_number}/{N_FOLDS} ====="
        )

        X_fit = X_train.iloc[split["fit"]]
        y_fit = y_train.iloc[split["fit"]]

        X_stop = X_train.iloc[split["stop"]]
        y_stop = y_train.iloc[split["stop"]]

        X_score = X_train.iloc[split["score"]]
        y_score = y_train.iloc[split["score"]]

        # ==================================
        # STANDARDISATION
        # ==================================
        #
        # Le scaler apprend uniquement
        # sur les données d'apprentissage
        # du fold actuel.

        scaler = StandardScaler()

        X_fit_scaled = scaler.fit_transform(
            X_fit
        )

        X_stop_scaled = scaler.transform(
            X_stop
        )

        X_score_scaled = scaler.transform(
            X_score
        )

        # ==================================
        # OVERSAMPLING
        # ==================================
        #
        # Ne jamais appliquer le rééquilibrage
        # aux données d'évaluation.

        X_fit_scaled, y_fit_resampled = (
            apply_sampling(

                X_fit_scaled,

                y_fit,

                configuration["Sampling"]
            )
        )

        # Créer un nouveau réseau
        # indépendant pour chaque fold.

        model = create_model(
            configuration
        )

        (
            best_model,
            best_epoch,
            best_stop_ap,
            history

        ) = train_with_early_stopping(

            model,

            X_fit_scaled,

            y_fit_resampled,

            X_stop_scaled,

            y_stop
        )

        # Evaluer le meilleur checkpoint
        # sur le fold indépendant.

        probabilities = best_model.predict_proba(
            X_score_scaled
        )[:, 1]

        fold_ap = average_precision_score(

            y_score,

            probabilities
        )

        fold_scores.append(float(fold_ap))

        fold_epochs.append(best_epoch)

        history["Trial"] = configuration["Trial"]

        history["Fold"] = fold_number

        all_histories.append(history)

        print(

            f"\n  Fold AP: {fold_ap:.6f} | "

            f"Best epoch: {best_epoch}"
        )

    # Calculer la moyenne et la stabilité
    # des performances entre les folds.

    result = {

        **configuration,

        "Mean_AP": float(
            np.mean(fold_scores)
        ),

        "Std_AP": float(
            np.std(fold_scores)
        ),

        "Recommended_Epochs": int(
            np.median(fold_epochs)
        ),

        "Search_Time": (
            perf_counter() - start_time
        )
    }

    for index, score in enumerate(
        fold_scores,
        start=1
    ):

        result[f"Fold_{index}_AP"] = score

        result[f"Fold_{index}_Epochs"] = (
            fold_epochs[index - 1]
        )

    histories = pd.concat(

        all_histories,

        ignore_index=True
    )

    return result, histories


# ==========================================
# 9. RECHERCHE DES HYPERPARAMETRES
# ==========================================

def run_search(X_train, y_train):

    print("\n===== CROSS VALIDATION SPLITS =====")

    splits = create_cv_splits(

        X_train,

        y_train
    )

    all_results = []

    all_histories = []

    start_time = perf_counter()

    for configuration in CONFIGURATIONS:

        print("\n" + "=" * 60)

        print(
            f"CONFIGURATION "
            f"{configuration['Trial']}"
            f"/{len(CONFIGURATIONS)}"
        )

        print("=" * 60)

        for key, value in configuration.items():

            print(f"{key}: {value}")

        result, histories = (
            evaluate_configuration(

                configuration,

                X_train,

                y_train,

                splits
            )
        )

        all_results.append(result)

        all_histories.append(histories)

        print(
            f"\nMean AP: {result['Mean_AP']:.6f}"
        )

        print(
            f"Std AP: {result['Std_AP']:.6f}"
        )

        print(
            f"Recommended epochs: "
            f"{result['Recommended_Epochs']}"
        )

        # Sauvegarder après chaque configuration
        # afin de conserver les résultats déjà
        # calculés en cas d'interruption.

        pd.DataFrame(all_results).to_csv(

            REPORTS_DIR / "search_results.csv",

            index=False
        )

        pd.concat(
            all_histories,
            ignore_index=True

        ).to_csv(

            REPORTS_DIR / "learning_history.csv",

            index=False
        )

    search_time = perf_counter() - start_time

    results = pd.DataFrame(all_results)

    results = results.sort_values(

        by=["Mean_AP", "Std_AP"],

        ascending=[False, True]

    ).reset_index(drop=True)

    results.to_csv(

        REPORTS_DIR / "search_results.csv",

        index=False
    )

    histories = pd.concat(

        all_histories,

        ignore_index=True
    )

    # Récupérer la configuration gagnante.
    best_trial = int(
        results.iloc[0]["Trial"]
    )

    best_configuration = next(

        configuration

        for configuration in CONFIGURATIONS

        if configuration["Trial"] == best_trial
    )

    recommended_epochs = int(

        results.iloc[0]["Recommended_Epochs"]
    )

    print("\n===== SEARCH RESULTS =====")

    print(

        results[

            [
                "Trial",
                "Architecture",
                "Sampling",
                "Mean_AP",
                "Std_AP",
                "Recommended_Epochs"
            ]

        ].round(5).to_string(index=False)
    )

    print(
        f"\nTotal search time: "
        f"{search_time:.2f}s"
    )

    return (

        results,

        histories,

        best_configuration,

        recommended_epochs,

        search_time
    )


# ==========================================
# 10. REENTRAINEMENT FINAL
# ==========================================

def train_final_model(

    X_train,

    y_train,

    configuration,

    epochs
):

    print("\n===== FINAL MODEL TRAINING =====")

    start_time = perf_counter()

    # Apprendre la standardisation sur
    # la totalité du Training Set original.

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    # Appliquer la stratégie sélectionnée.

    X_resampled, y_resampled = apply_sampling(

        X_scaled,

        y_train,

        configuration["Sampling"]
    )

    model = create_model(configuration)

    # Le nombre d'epochs a été déterminé
    # à partir des résultats des trois folds.
    #
    # Nous pouvons donc réentraîner sur
    # tout le Training Set sans utiliser
    # la validation externe.

    for epoch in range(1, epochs + 1):

        model.partial_fit(

            X_resampled,

            y_resampled,

            classes=np.array([0, 1])
        )

        print(

            f"Epoch {epoch}/{epochs} | "

            f"Loss: {model.loss_:.6f}"
        )

    # Réunir le scaler et le réseau dans
    # un seul objet pour faciliter
    # notre future API FastAPI.
    #
    # Les deux composants sont déjà ajustés.
    # Il ne faut pas appeler fit() ici.

    pipeline = Pipeline([

        ("scaler", scaler),

        ("classifier", model)
    ])

    training_time = (

        perf_counter() - start_time
    )

    return pipeline, training_time


# ==========================================
# 11. EVALUATION DES MODELES
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

    for key, value in results.items():

        if isinstance(value, float):

            print(f"{key}: {value:.6f}")

        else:

            print(f"{key}: {value}")

    return results, matrix, probabilities


# ==========================================
# 12. SAUVEGARDE DU MODELE
# ==========================================

def save_model(

    model,

    configuration,

    epochs,

    results,

    training_time,

    search_time,

    feature_names
):

    model_path = (

        MODELS_DIR / "optimized_model.joblib"
    )

    joblib.dump(

        model,

        model_path,

        compress=3
    )

    classifier = model.named_steps["classifier"]

    # Compter les poids et les biais
    # réellement appris par le réseau.

    number_of_parameters = int(

        sum(
            weights.size
            for weights in classifier.coefs_
        )

        +

        sum(
            biases.size
            for biases in classifier.intercepts_
        )
    )

    metadata = {

        "configuration": configuration,

        "epochs": epochs,

        "batch_size": BATCH_SIZE,

        "activation": "relu",

        "solver": "adam",

        "sampling_ratio": SAMPLING_RATIO,

        "mean_cv_ap": float(
            results.iloc[0]["Mean_AP"]
        ),

        "std_cv_ap": float(
            results.iloc[0]["Std_AP"]
        ),

        "training_time": training_time,

        "search_time": search_time,

        "trainable_parameters": (
            number_of_parameters
        ),

        "classification_threshold": THRESHOLD,

        "feature_names": list(feature_names),

        "input_format": "processed_features"
    }

    with open(

        MODELS_DIR / "best_parameters.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metadata,

            file,

            indent=4
        )

    size = (

        model_path.stat().st_size

        / (1024 ** 2)
    )

    print(f"\nModel saved: {model_path}")

    print(
        f"Trainable parameters: "
        f"{number_of_parameters}"
    )

    print(f"Model size: {size:.6f} MiB")

    return size


# ==========================================
# 13. GRAPHIQUES DE LA RECHERCHE
# ==========================================

def plot_search_results(results, histories):

    # --------------------------------------
    # GRAPHIQUE 1 : CLASSEMENT
    # --------------------------------------

    plt.figure(figsize=(10, 6))

    plt.barh(

        results["Trial"].astype(str),

        results["Mean_AP"],

        xerr=results["Std_AP"],

        capsize=4,

        color="steelblue"
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Neural Network - Configuration Ranking"
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
    # GRAPHIQUE 2 : PROGRESSION
    # --------------------------------------

    chronological = results.sort_values(
        "Trial"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(

        chronological["Trial"],

        chronological["Mean_AP"],

        marker="o",

        label="Configuration AP"
    )

    plt.plot(

        chronological["Trial"],

        chronological["Mean_AP"].cummax(),

        label="Best AP So Far"
    )

    plt.title(
        "Neural Network - Search Progress"
    )

    plt.xlabel("Trial")
    plt.ylabel("Average Precision")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(

        REPORTS_DIR / "search_progress.png",

        dpi=300
    )

    plt.close()

    # Identifier la configuration gagnante.
    best_trial = int(
        results.iloc[0]["Trial"]
    )

    best_history = histories[

        histories["Trial"] == best_trial
    ]

    # --------------------------------------
    # GRAPHIQUE 3 : COURBES D'APPRENTISSAGE
    # --------------------------------------

    for fold in range(1, N_FOLDS + 1):

        fold_history = best_history[

            best_history["Fold"] == fold
        ]

        plt.figure(figsize=(10, 6))

        plt.plot(

            fold_history["Epoch"],

            fold_history["Training_Loss"],

            label="Training Objective"
        )

        plt.plot(

            fold_history["Epoch"],

            fold_history["Validation_Loss"],

            label="Early Stopping Log Loss"
        )

        plt.title(

            f"Best Configuration - Loss - Fold {fold}"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.2)

        plt.tight_layout()

        plt.savefig(

            REPORTS_DIR
            / f"learning_curves_fold_{fold}.png",

            dpi=300
        )

        plt.close()

    # --------------------------------------
    # GRAPHIQUE 4 : EARLY STOPPING
    # --------------------------------------

    plt.figure(figsize=(11, 7))

    for fold in range(1, N_FOLDS + 1):

        fold_history = best_history[

            best_history["Fold"] == fold
        ]

        plt.plot(

            fold_history["Epoch"],

            fold_history["Average_Precision"],

            marker="o",

            label=f"Fold {fold}"
        )

        # Identifier la meilleure epoch
        # pour chacun des trois folds.

        best_index = (
            fold_history["Average_Precision"]
            .idxmax()
        )

        best_point = fold_history.loc[
            best_index
        ]

        plt.scatter(

            best_point["Epoch"],

            best_point["Average_Precision"],

            marker="*",

            s=220
        )

    plt.title(
        "Neural Network - Early Stopping Checkpoints"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Early Stopping Average Precision")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(

        REPORTS_DIR / "early_stopping.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 14. COMPARAISON AVANT / APRES
# ==========================================

def compare_with_baselines(

    optimized_model,

    X_val,

    y_val,

    training_time,

    model_size
):

    # Charger les deux réseaux déjà entraînés.
    models = {

        "Baseline Original": joblib.load(

            BASELINE_DIR / "original.joblib"
        ),

        "Baseline Oversampling": joblib.load(

            BASELINE_DIR / "oversampling.joblib"
        ),

        "Optimized Neural Network": optimized_model
    }

    baseline_report = pd.read_csv(

        BASE_DIR

        / "reports"

        / "neural_network"

        / "comparison.csv"
    )

    all_results = []

    matrices = {}

    scores = {}

    for name, model in models.items():

        results, matrix, probabilities = (

            evaluate_model(

                model,

                X_val,

                y_val,

                name
            )
        )

        if name == "Optimized Neural Network":

            results["Training_Time"] = (
                training_time
            )

            results["Model_Size_MB"] = (
                model_size
            )

        else:

            strategy = (

                "original"

                if name == "Baseline Original"

                else "oversampling"
            )

            baseline_row = baseline_report[

                baseline_report["Model"]
                == strategy

            ].iloc[0]

            results["Training_Time"] = float(

                baseline_row["Training_Time"]
            )

            results["Model_Size_MB"] = (

                (
                    BASELINE_DIR
                    / f"{strategy}.joblib"
                ).stat().st_size

                / (1024 ** 2)
            )

        all_results.append(results)

        matrices[name] = matrix

        scores[name] = probabilities

    comparison = pd.DataFrame(all_results)

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
        "Neural Network - Before and After Optimization"
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

    plt.axhline(

        y_val.mean(),

        linestyle="--",

        color="gray",

        label="Fraud prevalence"
    )

    plt.title(
        "Neural Network - Precision Recall Curves"
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

            xticklabels=["Legitimate", "Fraud"],

            yticklabels=["Legitimate", "Fraud"],

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
            "Seconds",
            "training_time.png"
        ),

        (
            "Prediction_Time",
            "Seconds",
            "prediction_time.png"
        ),

        (
            "Model_Size_MB",
            "MiB",
            "model_size.png"
        )
    ]

    for metric, ylabel, filename in technical_metrics:

        ax = comparison.set_index(

            "Model"

        )[metric].plot(

            kind="bar",

            figsize=(10, 6)
        )

        ax.set_title(
            f"Neural Network - {metric}"
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

        comparison["Model"]
        == "Baseline Oversampling",

        "Average_Precision"

    ].iloc[0]

    optimized_ap = comparison.loc[

        comparison["Model"]
        == "Optimized Neural Network",

        "Average_Precision"

    ].iloc[0]

    improvement = optimized_ap - baseline_ap

    print("\n===== OPTIMIZATION SUMMARY =====")

    print(f"Baseline AP: {baseline_ap:.6f}")

    print(f"Optimized AP: {optimized_ap:.6f}")

    print(

        f"AP improvement: "

        f"{improvement * 100:+.4f} percentage points"
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

    if improvement > 0:

        print(
            "Optimization improved validation AP!"
        )

    else:

        print(
            "The baseline remains competitive."
        )

    return comparison


# ==========================================
# 15. ANALYSE DE L'ARCHITECTURE
# ==========================================

def save_network_statistics(pipeline):

    model = pipeline.named_steps["classifier"]

    statistics = []

    # Chaque matrice coefs_ contient les poids
    # reliant deux couches successives.

    for index, (weights, biases) in enumerate(

        zip(model.coefs_, model.intercepts_),

        start=1
    ):

        statistics.append({

            "Layer": index,

            "Input_Neurons": weights.shape[0],

            "Output_Neurons": weights.shape[1],

            "Weights": weights.size,

            "Biases": biases.size,

            "Total_Parameters": (
                weights.size + biases.size
            )
        })

    df = pd.DataFrame(statistics)

    df.to_csv(

        REPORTS_DIR / "network_statistics.csv",

        index=False
    )

    print("\n===== NETWORK STATISTICS =====")

    print(df.to_string(index=False))

    print(

        f"\nTotal trainable parameters: "

        f"{df['Total_Parameters'].sum()}"
    )

    plt.figure(figsize=(9, 6))

    plt.bar(

        df["Layer"].astype(str),

        df["Total_Parameters"],

        color="steelblue"
    )

    plt.title(
        "Optimized Neural Network - Layer Parameters"
    )

    plt.xlabel("Layer")
    plt.ylabel("Trainable Parameters")

    plt.tight_layout()

    plt.savefig(

        REPORTS_DIR / "network_parameters.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 16. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)

    print(
        "FRAUDDETECT AI - Neural Network Optimization"
    )

    print("=" * 60)

    # 1. Charger les données.
    X_train, y_train, X_val, y_val = (
        load_data()
    )

    # 2. Rechercher les hyperparamètres.
    (

        results,

        histories,

        best_configuration,

        recommended_epochs,

        search_time

    ) = run_search(

        X_train,

        y_train
    )

    print("\n===== BEST CONFIGURATION =====")

    for key, value in best_configuration.items():

        print(f"{key}: {value}")

    print(
        f"Recommended epochs: {recommended_epochs}"
    )

    # 3. Générer les graphiques de recherche.
    plot_search_results(

        results,

        histories
    )

    # 4. Réentraîner le meilleur réseau.
    optimized_model, training_time = (

        train_final_model(

            X_train,

            y_train,

            best_configuration,

            recommended_epochs
        )
    )

    # 5. Sauvegarder le réseau et son scaler.
    model_size = save_model(

        optimized_model,

        best_configuration,

        recommended_epochs,

        results,

        training_time,

        search_time,

        X_train.columns
    )

    # 6. Comparer les performances.
    compare_with_baselines(

        optimized_model,

        X_val,

        y_val,

        training_time,

        model_size
    )

    # 7. Analyser la structure du réseau.
    save_network_statistics(
        optimized_model
    )

    print("\n" + "=" * 60)

    print(
        "Neural Network optimization completed successfully!"
    )

    print(f"Total search time: {search_time:.2f}s")

    print(f"Reports saved in: {REPORTS_DIR}")

    print("=" * 60)


if __name__ == "__main__":

    main()
