
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    matthews_corrcoef
)


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

THRESHOLD_DIR = (
    BASE_DIR / "reports" / "threshold_tuning"
)

OUTPUT_DIR = (
    BASE_DIR / "reports" / "final_evaluation"
)

# Créer le dossier des résultats finaux.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seuil historique utilisé comme référence.
DEFAULT_THRESHOLD = 0.5


# ==========================================
# 2. CHARGEMENT DU MODELE SELECTIONNE
# ==========================================

def load_selected_model():

    metadata_path = (
        THRESHOLD_DIR / "selected_model.json"
    )

    # Charger les informations générées
    # pendant le Threshold Tuning.

    with open(
        metadata_path,
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    model_path = (
        BASE_DIR / metadata["model_path"]
    )

    if not model_path.is_file():

        raise FileNotFoundError(model_path)

    # Charger uniquement notre propre
    # modèle sauvegardé précédemment.

    model = joblib.load(model_path)

    # Récupérer le seuil exact.
    # Ne pas utiliser une valeur arrondie.

    threshold = float(metadata["threshold"])

    if not 0 <= threshold <= 1:

        raise ValueError("Invalid threshold.")

    print("\n===== SELECTED MODEL =====")

    print(f"Model: {metadata['model_name']}")

    print(f"Threshold: {threshold:.12f}")

    print(
        f"Strategy: "
        f"{metadata['selection_strategy']}"
    )

    print(f"Model path: {model_path}")

    return model, metadata


# ==========================================
# 3. CHARGEMENT DU TEST SET
# ==========================================

def load_test_data(metadata):

    # Les données ont déjà été préparées
    # pendant notre phase de preprocessing.
    #
    # Aucun nouveau scaler ne sera entraîné.

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    )["Class"]

    # Vérifier les dimensions.

    if len(X_test) != len(y_test):

        raise ValueError(
            "Invalid test dimensions."
        )

    expected_features = metadata["feature_names"]

    # Vérifier les noms des variables.

    if set(X_test.columns) != set(expected_features):

        raise ValueError(
            "Test features do not match model features."
        )

    # Respecter exactement l'ordre des variables
    # utilisé pendant l'entraînement.

    X_test = X_test[expected_features]

    # Vérifier les valeurs manquantes
    # et les valeurs infinies.

    if not np.isfinite(
        X_test.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Invalid values in test features."
        )

    if set(y_test.unique()) != {0, 1}:

        raise ValueError(
            "Invalid test labels."
        )

    print("\n===== TEST DATA =====")

    print(f"Transactions: {len(X_test)}")

    print(f"Features: {X_test.shape[1]}")

    print(f"Legitimate: {(y_test == 0).sum()}")

    print(f"Fraud: {(y_test == 1).sum()}")

    print(
        f"Fraud percentage: "
        f"{y_test.mean() * 100:.6f}%"
    )

    return X_test, y_test


# ==========================================
# 4. CALCUL DES SCORES DE FRAUDE
# ==========================================

def predict_probabilities(model, X_test):

    # Vérifier l'ordre des classes du modèle.

    classes = np.asarray(model.classes_)

    fraud_indices = np.flatnonzero(classes == 1)

    if len(fraud_indices) != 1:

        raise ValueError(
            f"Invalid model classes: {classes}"
        )

    fraud_index = fraud_indices[0]

    # Mesurer le temps de prédiction.

    start_time = time.perf_counter()

    probabilities = model.predict_proba(
        X_test
    )[:, fraud_index]

    prediction_time = (
        time.perf_counter() - start_time
    )

    # Vérifier la validité des scores.

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):

        raise ValueError(
            "Invalid prediction scores."
        )

    print("\n===== PREDICTION PERFORMANCE =====")

    print(
        f"Prediction time: "
        f"{prediction_time:.6f} seconds"
    )

    print(
        f"Transactions per second: "
        f"{len(X_test) / prediction_time:.2f}"
    )

    return probabilities, prediction_time


# ==========================================
# 5. CALCUL DES METRIQUES
# ==========================================

def evaluate_model(
    y_true,
    probabilities,
    threshold,
    strategy,
    metadata
):

    # Transformer les scores en classes
    # à partir du seuil choisi.

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Calculer la matrice de confusion.

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Récupérer nos hypothèses de coûts
    # définies pendant le GO 15.

    cost_fp = metadata[
        "cost_assumptions"
    ]["false_positive"]

    cost_fn = metadata[
        "cost_assumptions"
    ]["false_negative"]

    total_cost = (
        fp * cost_fp + fn * cost_fn
    )

    # Calculer les métriques statistiques.

    metrics = {

        "Strategy": strategy,

        "Threshold": float(threshold),

        "Accuracy": accuracy_score(
            y_true, predictions
        ),

        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F2": fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "Average_Precision": average_precision_score(
            y_true,
            probabilities
        ),

        # Proportion de transactions légitimes
        # correctement reconnues.

        "Specificity": tn / (tn + fp),

        # Proportion de transactions légitimes
        # incorrectement signalées.

        "False_Positive_Rate": fp / (fp + tn),

        # Proportion de fraudes non détectées.

        "False_Negative_Rate": fn / (fn + tp),

        # Coefficient de Matthews.

        "MCC": matthews_corrcoef(
            y_true, predictions
        ),

        "True_Positives": int(tp),

        "True_Negatives": int(tn),

        "False_Positives": int(fp),

        "False_Negatives": int(fn),

        "Total_Alerts": int(tp + fp),

        "Total_Cost": float(total_cost)
    }

    return metrics, predictions


# ==========================================
# 6. AFFICHAGE DES RESULTATS
# ==========================================

def print_metrics(metrics):

    print("\n" + "=" * 60)

    print(f"STRATEGY: {metrics['Strategy']}")

    print("=" * 60)

    print(
        f"Threshold: "
        f"{metrics['Threshold']:.12f}"
    )

    metric_names = [

        "Accuracy",

        "Balanced_Accuracy",

        "Precision",

        "Recall",

        "F1",

        "F2",

        "ROC_AUC",

        "Average_Precision",

        "Specificity",

        "False_Positive_Rate",

        "False_Negative_Rate",

        "MCC"
    ]

    for metric in metric_names:

        print(
            f"{metric}: "
            f"{metrics[metric]:.6f}"
        )

    print("\nConfusion matrix results:")

    print(
        f"Detected fraud: "
        f"{metrics['True_Positives']}"
    )

    print(
        f"Missed fraud: "
        f"{metrics['False_Negatives']}"
    )

    print(
        f"False alerts: "
        f"{metrics['False_Positives']}"
    )

    print(
        f"Correct legitimate: "
        f"{metrics['True_Negatives']}"
    )

    print(
        f"Estimated cost: "
        f"{metrics['Total_Cost']:.2f} EUR"
    )


# ==========================================
# 7. COMPARAISON VALIDATION / TEST
# ==========================================

def compare_validation_test(metadata, test_metrics):

    validation = metadata["validation_metrics"]

    comparisons = []

    metric_mapping = {

        "Precision": "precision",

        "Recall": "recall",

        "F1": "f1",

        "F2": "f2",

        "Average_Precision": "average_precision"
    }

    # Comparer les mêmes métriques sur
    # le Validation Set et le Test Set.

    for metric, validation_key in metric_mapping.items():

        validation_score = validation[validation_key]

        test_score = test_metrics[metric]

        comparisons.append({

            "Metric": metric,

            "Validation": validation_score,

            "Test": test_score,

            "Difference": (
                test_score - validation_score
            ),

            "Difference_Points": (
                test_score - validation_score
            ) * 100
        })

    comparison = pd.DataFrame(comparisons)

    comparison.to_csv(

        OUTPUT_DIR / "validation_test_comparison.csv",

        index=False
    )

    print("\n===== VALIDATION VS TEST =====")

    print(
        comparison.round(6)
        .to_string(index=False)
    )

    return comparison


# ==========================================
# 8. GRAPHIQUE DES MATRICES DE CONFUSION
# ==========================================

def plot_confusion_matrices(default, optimized):

    fig, axes = plt.subplots(

        1, 2,

        figsize=(13, 5)
    )

    for ax, metrics in zip(
        axes,
        [default, optimized]
    ):

        matrix = np.array([

            [
                metrics["True_Negatives"],
                metrics["False_Positives"]
            ],

            [
                metrics["False_Negatives"],
                metrics["True_Positives"]
            ]
        ])

        sns.heatmap(

            matrix,

            annot=True,

            fmt="d",

            cmap="Blues",

            cbar=False,

            xticklabels=["Legitimate", "Fraud"],

            yticklabels=["Legitimate", "Fraud"],

            ax=ax
        )

        ax.set_title(

            f"{metrics['Strategy']} - "

            f"Threshold {metrics['Threshold']:.5f}"
        )

        ax.set_xlabel("Predicted Class")

        ax.set_ylabel("Actual Class")

    fig.suptitle(
        "FraudDetect AI - Test Set Confusion Matrices"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "test_confusion_matrices.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 9. COURBE PRECISION / RECALL
# ==========================================

def plot_precision_recall(y_test, probabilities):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_test,
            probabilities
        )
    )

    ap = average_precision_score(
        y_test,
        probabilities
    )

    # Prévalence des fraudes.
    # Niveau de référence d'un classement aléatoire.

    prevalence = y_test.mean()

    plt.figure(figsize=(10, 7))

    plt.plot(

        recall,

        precision,

        linewidth=2,

        label=f"Random Forest - AP: {ap:.4f}"
    )

    plt.axhline(

        prevalence,

        color="gray",

        linestyle="--",

        label="Fraud prevalence"
    )

    plt.title(
        "FraudDetect AI - Test Precision Recall Curve"
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.xlim(0, 1.02)

    plt.ylim(0, 1.05)

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "test_precision_recall_curve.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 10. COURBE ROC
# ==========================================

def plot_roc_curve(y_test, probabilities):

    fpr, tpr, thresholds = roc_curve(

        y_test,

        probabilities
    )

    auc = roc_auc_score(

        y_test,

        probabilities
    )

    plt.figure(figsize=(9, 7))

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"Random Forest - AUC: {auc:.4f}"
    )

    # Diagonale représentant un classement
    # aléatoire.

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        color="gray",

        label="Random classifier"
    )

    plt.title(
        "FraudDetect AI - Test ROC Curve"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "test_roc_curve.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 11. COMPARAISON VALIDATION / TEST
# ==========================================

def plot_validation_test(comparison):

    ax = comparison.set_index("Metric")[

        ["Validation", "Test"]

    ].plot(

        kind="bar",

        figsize=(12, 7)
    )

    ax.set_title(
        "FraudDetect AI - Validation vs Test Performance"
    )

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.08)

    # Afficher les valeurs au-dessus
    # de chaque barre.

    for container in ax.containers:

        ax.bar_label(

            container,

            fmt="%.3f",

            padding=3
        )

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "validation_vs_test.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 12. COMPARAISON DES DEUX SEUILS
# ==========================================

def plot_threshold_comparison(default, optimized):

    results = pd.DataFrame(
        [default, optimized]
    )

    ax = results.set_index("Strategy")[

        ["Precision", "Recall", "F1", "F2"]

    ].T.plot(

        kind="bar",

        figsize=(11, 7)
    )

    ax.set_title(
        "Test Set - Default vs Optimized Threshold"
    )

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.08)

    for container in ax.containers:

        ax.bar_label(

            container,

            fmt="%.3f",

            padding=3
        )

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "test_threshold_comparison.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 13. COMPARAISON DES COUTS
# ==========================================

def plot_cost_comparison(default, optimized):

    strategies = [

        "Default Threshold",

        "Selected Threshold"
    ]

    costs = [

        default["Total_Cost"],

        optimized["Total_Cost"]
    ]

    plt.figure(figsize=(9, 6))

    bars = plt.bar(

        strategies,

        costs,

        color=["steelblue", "seagreen"]
    )

    plt.bar_label(

        bars,

        fmt="%.2f EUR",

        padding=5
    )

    plt.ylim(
        0,
        max(costs) * 1.2 + 1
    )

    plt.title(
        "FraudDetect AI - Hypothetical Error Cost"
    )

    plt.ylabel("Estimated Cost (EUR)")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "test_cost_comparison.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 14. SAUVEGARDE DES RESULTATS
# ==========================================

def save_results(

    metadata,

    y_test,

    probabilities,

    predictions,

    default_metrics,

    final_metrics,

    prediction_time
):

    # Sauvegarder les métriques des deux seuils.

    pd.DataFrame([

        default_metrics,

        final_metrics

    ]).to_csv(

        OUTPUT_DIR / "test_metrics.csv",

        index=False
    )

    # Sauvegarder les prédictions individuelles.
    #
    # Ce fichier permettra d'analyser les erreurs
    # sans relancer les prédictions.

    prediction_results = pd.DataFrame({

        "Actual_Class": np.asarray(y_test),

        "Fraud_Score": probabilities,

        "Predicted_Class": predictions
    })

    prediction_results["Correct"] = (

        prediction_results["Actual_Class"]

        == prediction_results["Predicted_Class"]
    )

    prediction_results.to_csv(

        OUTPUT_DIR / "test_predictions.csv",

        index=False
    )

    # Générer un rapport de classification.

    report = classification_report(

        y_test,

        predictions,

        labels=[0, 1],

        target_names=["Legitimate", "Fraud"],

        digits=6,

        zero_division=0
    )

    with open(

        OUTPUT_DIR / "classification_report.txt",

        "w",

        encoding="utf-8"

    ) as file:

        file.write(report)

    print("\n===== CLASSIFICATION REPORT =====")

    print(report)

    # Construire notre résumé scientifique final.

    summary = {

        "project": "FraudDetect AI",

        "selected_model": metadata["model_name"],

        "model_path": metadata["model_path"],

        "selection_strategy": metadata[
            "selection_strategy"
        ],

        "threshold": metadata["threshold"],

        "test_transactions": int(len(y_test)),

        "test_frauds": int(y_test.sum()),

        "test_legitimate": int(
            (y_test == 0).sum()
        ),

        "test_prevalence": float(y_test.mean()),

        "prediction_time_seconds": float(
            prediction_time
        ),

        "validation_metrics": metadata[
            "validation_metrics"
        ],

        "test_default_threshold_metrics": default_metrics,

        "test_final_metrics": final_metrics,

        "cost_assumptions": metadata[
            "cost_assumptions"
        ],

        "cost_reduction_vs_default": float(

            default_metrics["Total_Cost"]

            - final_metrics["Total_Cost"]
        ),

        "model_retrained": False,

        "threshold_changed_on_test": False,

        "test_set_used": True
    }

    with open(

        OUTPUT_DIR / "final_evaluation_summary.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4
        )

    return summary


# ==========================================
# 15. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 65)

    print("FRAUDDETECT AI - Final Model Evaluation")

    print("=" * 65)

    # 1. Charger le modèle sélectionné.

    model, metadata = load_selected_model()

    # 2. Charger les données de test.

    X_test, y_test = load_test_data(metadata)

    # 3. Calculer une seule fois les scores.

    probabilities, prediction_time = (
        predict_probabilities(model, X_test)
    )

    # 4. Evaluer le seuil de référence.

    default_metrics, default_predictions = (
        evaluate_model(

            y_test,

            probabilities,

            DEFAULT_THRESHOLD,

            "Default Threshold",

            metadata
        )
    )

    # 5. Evaluer le seuil sélectionné
    # précédemment sur le Validation Set.
    #
    # Ce seuil est définitivement fixé.

    final_metrics, final_predictions = (
        evaluate_model(

            y_test,

            probabilities,

            metadata["threshold"],

            "Selected Threshold",

            metadata
        )
    )

    # 6. Afficher les résultats.

    print_metrics(default_metrics)

    print_metrics(final_metrics)

    # 7. Comparer Validation et Test.

    comparison = compare_validation_test(

        metadata,

        final_metrics
    )

    # 8. Générer les graphiques.

    plot_confusion_matrices(

        default_metrics,

        final_metrics
    )

    plot_precision_recall(

        y_test,

        probabilities
    )

    plot_roc_curve(

        y_test,

        probabilities
    )

    plot_validation_test(comparison)

    plot_threshold_comparison(

        default_metrics,

        final_metrics
    )

    plot_cost_comparison(

        default_metrics,

        final_metrics
    )

    # 9. Sauvegarder les rapports finaux.

    summary = save_results(

        metadata,

        y_test,

        probabilities,

        final_predictions,

        default_metrics,

        final_metrics,

        prediction_time
    )

    # 10. Afficher le bilan final du projet.

    print("\n" + "=" * 65)

    print("FINAL EVALUATION SUMMARY")

    print("=" * 65)

    print(f"Model: {summary['selected_model']}")

    print(
        f"Threshold: "
        f"{summary['threshold']:.12f}"
    )

    print(
        f"Test transactions: "
        f"{summary['test_transactions']}"
    )

    print(
        f"Test Average Precision: "
        f"{final_metrics['Average_Precision']:.6f}"
    )

    print(
        f"Test Precision: "
        f"{final_metrics['Precision']:.6f}"
    )

    print(
        f"Test Recall: "
        f"{final_metrics['Recall']:.6f}"
    )

    print(
        f"Test F1: "
        f"{final_metrics['F1']:.6f}"
    )

    print(
        f"Test F2: "
        f"{final_metrics['F2']:.6f}"
    )

    print(
        f"Test ROC AUC: "
        f"{final_metrics['ROC_AUC']:.6f}"
    )

    print(
        f"Detected fraud: "
        f"{final_metrics['True_Positives']}"
    )

    print(
        f"Missed fraud: "
        f"{final_metrics['False_Negatives']}"
    )

    print(
        f"False alerts: "
        f"{final_metrics['False_Positives']}"
    )

    print(
        f"Hypothetical cost reduction: "
        f"{summary['cost_reduction_vs_default']:.2f} EUR"
    )

    print("\nFinal evaluation completed successfully!")

    print(f"Reports saved in: {OUTPUT_DIR}")

    print("No model retraining performed.")

    print("No threshold optimization on test data.")

    print("=" * 65)


if __name__ == "__main__":

    main()
