
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import average_precision_score


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

BENCHMARK_DIR = (
    BASE_DIR / "reports" / "final_benchmark"
)

OUTPUT_DIR = (
    BASE_DIR / "reports" / "threshold_tuning"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seuil utilisé jusqu'à présent.
DEFAULT_THRESHOLD = 0.5

# Contraintes de performance.
MIN_PRECISION = 0.95
MIN_RECALL = 0.85

# Coûts fictifs pour notre expérimentation.
COST_FP = 5
COST_FN = 500

# Objectif principal retenu pour cette expérience.
# Nous privilégions la détection des fraudes.
PRIMARY_STRATEGY = "Best F2"


# ==========================================
# 2. CHARGEMENT DU VALIDATION SET
# ==========================================

def load_validation_data():

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    if len(X_val) != len(y_val):

        raise ValueError(
            "Invalid validation dimensions."
        )

    if X_val.shape[1] != 30:

        raise ValueError(
            "Expected 30 features."
        )

    if set(y_val.unique()) != {0, 1}:

        raise ValueError(
            "Invalid validation classes."
        )

    print("\n===== VALIDATION DATA =====")

    print(f"Transactions: {len(X_val)}")

    print(f"Legitimate: {(y_val == 0).sum()}")

    print(f"Fraud: {(y_val == 1).sum()}")

    return X_val, y_val


# ==========================================
# 3. CHARGEMENT DES CANDIDATS
# ==========================================

def load_candidates():

    # Charger les candidats sélectionnés
    # pendant le benchmark précédent.

    path = BENCHMARK_DIR / "threshold_candidates.json"

    with open(path, encoding="utf-8") as file:

        candidates = json.load(file)

    # Ajouter quelques modèles intéressants
    # qui pourraient ne pas avoir été
    # sélectionnés automatiquement.

    benchmark = pd.read_csv(
        BENCHMARK_DIR / "global_benchmark.csv"
    )

    additional_models = [

        "Random Forest - original",

        "Random Forest - optimized",

        "XGBoost - optimized",

        "XGBoost - oversampling"
    ]

    existing_names = {

        candidate["Model_Name"]

        for candidate in candidates
    }

    for name in additional_models:

        rows = benchmark[
            benchmark["Model_Name"] == name
        ]

        if rows.empty or name in existing_names:

            continue

        row = rows.iloc[0]

        candidates.append({

            "Model_Name": name,

            "Model_Path": row["Model_Path"],

            "Average_Precision": float(
                row["Average_Precision"]
            )
        })

        existing_names.add(name)

    # Vérifier la disponibilité des fichiers
    # avant de commencer les prédictions.

    for candidate in candidates:

        model_path = (
            BASE_DIR / candidate["Model_Path"]
        )

        if not model_path.is_file():

            raise FileNotFoundError(model_path)

    print("\n===== SELECTED CANDIDATES =====")

    for candidate in candidates:

        print(candidate["Model_Name"])

    print(f"Total candidates: {len(candidates)}")

    return candidates


# ==========================================
# 4. CALCUL DE TOUS LES SEUILS UTILES
# ==========================================

def evaluate_thresholds(y_true, probabilities):

    y_true = np.asarray(y_true)

    probabilities = np.asarray(probabilities)

    # Tester tous les scores distincts
    # du modèle comme seuils possibles.
    #
    # Ajouter également le seuil de référence
    # et les deux extrémités.

    thresholds = np.unique(

        np.concatenate([

            probabilities,

            [0.0, DEFAULT_THRESHOLD, 1.0]
        ])
    )

    # Ajouter un seuil sans aucune alerte,
    # uniquement s'il reste dans [0, 1].

    no_alert_threshold = np.nextafter(
        probabilities.max(), np.inf
    )

    if no_alert_threshold <= 1:

        thresholds = np.unique(
            np.append(thresholds, no_alert_threshold)
        )

    # Trier les transactions selon leurs scores.
    #
    # Ce tri évite de recalculer toutes les
    # prédictions pour chaque seuil.

    order = np.argsort(probabilities)

    sorted_scores = probabilities[order]

    sorted_labels = y_true[order]

    # Calculer le nombre cumulé de fraudes
    # parmi les transactions triées.

    cumulative_fraud = np.concatenate([

        [0],

        np.cumsum(sorted_labels)
    ])

    total_fraud = int(y_true.sum())

    total_legitimate = len(y_true) - total_fraud

    # Pour chaque seuil, identifier la position
    # à partir de laquelle les scores sont
    # supérieurs ou égaux au seuil.

    positions = np.searchsorted(

        sorted_scores,

        thresholds,

        side="left"
    )

    # Calcul vectorisé des matrices de confusion.

    fn = cumulative_fraud[positions]

    tp = total_fraud - fn

    tn = positions - fn

    fp = total_legitimate - tn

    # Nombre total d'alertes générées.
    alerts = tp + fp

    # Precision = TP / (TP + FP)
    precision = np.divide(

        tp,

        alerts,

        out=np.zeros(len(thresholds)),

        where=alerts > 0
    )

    # Recall = TP / (TP + FN)
    recall = tp / total_fraud

    # F1-score.
    f1_denominator = 2 * tp + fp + fn

    f1 = np.divide(

        2 * tp,

        f1_denominator,

        out=np.zeros(len(thresholds)),

        where=f1_denominator > 0
    )

    # F2-score :
    # le rappel est davantage privilégié.

    f2_denominator = 5 * tp + 4 * fn + fp

    f2 = np.divide(

        5 * tp,

        f2_denominator,

        out=np.zeros(len(thresholds)),

        where=f2_denominator > 0
    )

    # Pourcentage de transactions légitimes
    # incorrectement signalées.

    false_positive_rate = fp / total_legitimate

    # Coût théorique des erreurs.

    total_cost = (

        fp * COST_FP

        + fn * COST_FN
    )

    # Retourner un tableau contenant
    # les résultats de tous les seuils.

    return pd.DataFrame({

        "Threshold": thresholds,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "F2": f2,

        "False_Positive_Rate": false_positive_rate,

        "True_Positives": tp,

        "True_Negatives": tn,

        "False_Positives": fp,

        "False_Negatives": fn,

        "Alerts": alerts,

        "Total_Cost": total_cost
    })


# ==========================================
# 5. SELECTION DES MEILLEURS SEUILS
# ==========================================

def select_operating_points(results):

    selected = []

    def add_result(strategy, row):

        result = row.to_dict()

        result["Strategy"] = strategy

        selected.append(result)

    # --------------------------------------
    # A. SEUIL PAR DEFAUT
    # --------------------------------------

    default = results.loc[

        (
            results["Threshold"]
            - DEFAULT_THRESHOLD
        ).abs().idxmin()
    ]

    add_result("Default", default)

    # --------------------------------------
    # B. MEILLEUR F1
    # --------------------------------------

    best_f1 = results.sort_values(

        ["F1", "Precision", "Threshold"],

        ascending=[False, False, False]

    ).iloc[0]

    add_result("Best F1", best_f1)

    # --------------------------------------
    # C. MEILLEUR F2
    # --------------------------------------

    best_f2 = results.sort_values(

        ["F2", "Precision", "Threshold"],

        ascending=[False, False, False]

    ).iloc[0]

    add_result("Best F2", best_f2)

    # --------------------------------------
    # D. PRECISION MINIMALE DE 95 %
    # --------------------------------------

    high_precision = results[

        (results["Precision"] >= MIN_PRECISION)

        & (results["Alerts"] > 0)
    ]

    if not high_precision.empty:

        best = high_precision.sort_values(

            ["Recall", "Precision", "Threshold"],

            ascending=[False, False, False]

        ).iloc[0]

        add_result("High Precision", best)

    else:

        print("95% precision constraint not reached.")

    # --------------------------------------
    # E. RECALL MINIMAL DE 85 %
    # --------------------------------------

    high_recall = results[

        results["Recall"] >= MIN_RECALL
    ]

    if not high_recall.empty:

        best = high_recall.sort_values(

            ["Precision", "Recall", "Threshold"],

            ascending=[False, False, False]

        ).iloc[0]

        add_result("High Recall", best)

    else:

        print("85% recall constraint not reached.")

    # --------------------------------------
    # F. COUT MINIMAL
    # --------------------------------------

    best_cost = results.sort_values(

        ["Total_Cost", "F2", "Precision"],

        ascending=[True, False, False]

    ).iloc[0]

    add_result("Minimum Cost", best_cost)

    return pd.DataFrame(selected)


# ==========================================
# 6. ANALYSE D'UN MODELE
# ==========================================

def analyze_model(candidate, X_val, y_val):

    name = candidate["Model_Name"]

    path = BASE_DIR / candidate["Model_Path"]

    print("\n" + "=" * 60)

    print(f"MODEL: {name}")

    print("=" * 60)

    # Charger un modèle déjà entraîné.
    # Ne charger que nos propres fichiers Joblib.

    model = joblib.load(path)

    # Identifier la colonne correspondant
    # à la classe frauduleuse.

    classes = np.asarray(model.classes_)

    fraud_indices = np.flatnonzero(classes == 1)

    if len(fraud_indices) != 1:

        raise ValueError(
            f"Invalid model classes: {classes}"
        )

    fraud_index = fraud_indices[0]

    # Calculer les scores une seule fois.
    # Aucun nouvel entraînement.

    probabilities = model.predict_proba(
        X_val
    )[:, fraud_index]

    if (

        not np.isfinite(probabilities).all()

        or np.any(probabilities < 0)

        or np.any(probabilities > 1)
    ):

        raise ValueError("Invalid model scores.")

    # Vérifier que le modèle retrouve son
    # Average Precision du benchmark précédent.

    ap = average_precision_score(

        y_val,

        probabilities
    )

    expected_ap = candidate["Average_Precision"]

    print(f"Benchmark AP: {expected_ap:.6f}")

    print(f"Current AP:   {ap:.6f}")

    # Un écart important pourrait révéler
    # un mauvais modèle ou prétraitement.

    if abs(ap - expected_ap) > 0.0001:

        raise ValueError(

            f"AP mismatch for {name}. "
            "Check model and preprocessing."
        )

    # Evaluer tous les seuils possibles.

    results = evaluate_thresholds(

        y_val,

        probabilities
    )

    selected = select_operating_points(results)

    # Ajouter les informations du modèle.

    results["Model_Name"] = name

    results["Average_Precision"] = ap

    selected["Model_Name"] = name

    selected["Model_Path"] = candidate["Model_Path"]

    selected["Average_Precision"] = ap

    # Enregistrer les courbes complètes.

    filename = (

        name.lower()

        .replace(" - ", "_")

        .replace(" ", "_")
    )

    results.to_csv(

        OUTPUT_DIR / f"{filename}_thresholds.csv",

        index=False
    )

    print(f"Thresholds tested: {len(results)}")

    print(

        selected[[

            "Strategy",

            "Threshold",

            "Precision",

            "Recall",

            "F1",

            "F2",

            "False_Positives",

            "False_Negatives",

            "Total_Cost"

        ]].round(6).to_string(index=False)
    )

    return results, selected


# ==========================================
# 7. GRAPHIQUES DES SEUILS
# ==========================================

def plot_threshold_curves(results, selected):

    name = results["Model_Name"].iloc[0]

    filename = (

        name.lower()

        .replace(" - ", "_")

        .replace(" ", "_")
    )

    # --------------------------------------
    # GRAPHIQUE 1 : METRIQUES PAR SEUIL
    # --------------------------------------

    plt.figure(figsize=(12, 7))

    for metric in ["Precision", "Recall", "F1", "F2"]:

        plt.plot(

            results["Threshold"],

            results[metric],

            label=metric
        )

    plt.axvline(

        DEFAULT_THRESHOLD,

        color="gray",

        linestyle="--",

        label="Default threshold"
    )

    best_f1 = selected.loc[

        selected["Strategy"] == "Best F1"

    ].iloc[0]

    best_f2 = selected.loc[

        selected["Strategy"] == "Best F2"

    ].iloc[0]

    plt.scatter(

        best_f1["Threshold"],

        best_f1["F1"],

        marker="*",

        s=180,

        label="Best F1"
    )

    plt.scatter(

        best_f2["Threshold"],

        best_f2["F2"],

        marker="*",

        s=180,

        label="Best F2"
    )

    plt.title(f"{name} - Threshold Optimization")

    plt.xlabel("Classification Threshold")

    plt.ylabel("Score")

    plt.xlim(0, 1)

    plt.ylim(0, 1.05)

    plt.grid(alpha=0.2)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / f"{filename}_metrics.png",

        dpi=200
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 2 : COUT DES ERREURS
    # --------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(

        results["Threshold"],

        results["Total_Cost"],

        label="Total Cost"
    )

    best_cost = selected.loc[

        selected["Strategy"] == "Minimum Cost"

    ].iloc[0]

    plt.axvline(

        best_cost["Threshold"],

        color="green",

        linestyle="--",

        label="Minimum Cost Threshold"
    )

    plt.axvline(

        DEFAULT_THRESHOLD,

        color="gray",

        linestyle="--",

        label="Default Threshold"
    )

    plt.title(f"{name} - Error Cost")

    plt.xlabel("Threshold")

    plt.ylabel("Estimated Cost (EUR)")

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / f"{filename}_cost.png",

        dpi=200
    )

    plt.close()


# ==========================================
# 8. COMPARAISON GLOBALE
# ==========================================

def compare_all_models(results):

    # Sauvegarder toutes les stratégies.

    results.to_csv(

        OUTPUT_DIR / "all_operating_points.csv",

        index=False
    )

    # Sélectionner les meilleurs seuils
    # selon le F2-score.

    best_f2 = results[

        results["Strategy"] == "Best F2"

    ].sort_values(

        ["F2", "Precision", "Average_Precision"],

        ascending=False
    )

    best_f2.to_csv(

        OUTPUT_DIR / "best_f2_ranking.csv",

        index=False
    )

    # Sélectionner les meilleurs F1.

    best_f1 = results[

        results["Strategy"] == "Best F1"

    ].sort_values(

        ["F1", "Precision"],

        ascending=False
    )

    best_f1.to_csv(

        OUTPUT_DIR / "best_f1_ranking.csv",

        index=False
    )

    # Sélectionner les coûts minimaux.

    best_cost = results[

        results["Strategy"] == "Minimum Cost"

    ].sort_values(

        ["Total_Cost", "F2"],

        ascending=[True, False]
    )

    best_cost.to_csv(

        OUTPUT_DIR / "minimum_cost_ranking.csv",

        index=False
    )

    columns = [

        "Model_Name",

        "Threshold",

        "Precision",

        "Recall",

        "F1",

        "F2",

        "False_Positives",

        "False_Negatives",

        "Total_Cost"
    ]

    print("\n===== BEST F1 MODELS =====")

    print(
        best_f1[columns]
        .round(6)
        .to_string(index=False)
    )

    print("\n===== BEST F2 MODELS =====")

    print(
        best_f2[columns]
        .round(6)
        .to_string(index=False)
    )

    print("\n===== MINIMUM COST MODELS =====")

    print(
        best_cost[columns]
        .round(6)
        .to_string(index=False)
    )

    # Comparer les scores F2 de tous les modèles.

    plt.figure(figsize=(12, 8))

    sns.barplot(

        data=best_f2,

        x="F2",

        y="Model_Name",

        color="steelblue"
    )

    plt.title("FraudDetect AI - Best F2 Ranking")

    plt.xlabel("F2-score")

    plt.xlim(0, 1.05)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "best_f2_ranking.png",

        dpi=250
    )

    plt.close()

    # Comparer les coûts de chaque modèle.

    plt.figure(figsize=(12, 8))

    sns.barplot(

        data=best_cost,

        x="Total_Cost",

        y="Model_Name",

        color="steelblue"
    )

    plt.title("FraudDetect AI - Minimum Error Cost")

    plt.xlabel("Estimated Cost (EUR)")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "minimum_cost_ranking.png",

        dpi=250
    )

    plt.close()


# ==========================================
# 9. SELECTION DU MODELE FINAL
# ==========================================

def select_final_model(results, feature_names):

    # Pour notre projet de fraude, nous
    # privilégions le F2-score.
    #
    # Ce choix est effectué sur la validation
    # et non sur le Test Set.

    eligible = results[

        results["Strategy"] == PRIMARY_STRATEGY
    ]

    best = eligible.sort_values(

        ["F2", "Precision", "Average_Precision"],

        ascending=False

    ).iloc[0]

    # Comparer avec le seuil initial
    # du même modèle.

    baseline = results[

        (results["Model_Name"] == best["Model_Name"])

        & (results["Strategy"] == "Default")

    ].iloc[0]

    comparison = pd.DataFrame([baseline, best])

    comparison.to_csv(

        OUTPUT_DIR / "final_model_comparison.csv",

        index=False
    )

    print("\n===== SELECTED FINAL MODEL =====")

    print(f"Model: {best['Model_Name']}")

    print(f"Strategy: {PRIMARY_STRATEGY}")

    print(f"Threshold: {best['Threshold']:.10f}")

    print(f"Precision: {best['Precision']:.6f}")

    print(f"Recall: {best['Recall']:.6f}")

    print(f"F1: {best['F1']:.6f}")

    print(f"F2: {best['F2']:.6f}")

    print(f"AP: {best['Average_Precision']:.6f}")

    print(f"Detected fraud: {int(best['True_Positives'])}")

    print(f"Missed fraud: {int(best['False_Negatives'])}")

    print(f"False alerts: {int(best['False_Positives'])}")

    print(f"Estimated cost: {best['Total_Cost']:.2f}")

    # Sauvegarder les informations qui seront
    # utilisées lors de l'évaluation finale.

    metadata = {

        "model_name": best["Model_Name"],

        "model_path": best["Model_Path"],

        "selection_strategy": PRIMARY_STRATEGY,

        "threshold": float(best["Threshold"]),

        "feature_names": list(feature_names),

        "input_format": "processed_features",

        "evaluation_dataset": "validation",

        "test_set_used": False,

        "validation_metrics": {

            "precision": float(best["Precision"]),

            "recall": float(best["Recall"]),

            "f1": float(best["F1"]),

            "f2": float(best["F2"]),

            "average_precision": float(
                best["Average_Precision"]
            ),

            "true_positives": int(
                best["True_Positives"]
            ),

            "true_negatives": int(
                best["True_Negatives"]
            ),

            "false_positives": int(
                best["False_Positives"]
            ),

            "false_negatives": int(
                best["False_Negatives"]
            )
        },

        "cost_assumptions": {

            "false_positive": COST_FP,

            "false_negative": COST_FN,

            "currency": "EUR",

            "hypothetical": True
        }
    }

    with open(

        OUTPUT_DIR / "selected_model.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(metadata, file, indent=4)

    # --------------------------------------
    # GRAPHIQUE 1 : AVANT ET APRES
    # --------------------------------------

    ax = comparison.set_index("Strategy")[

        ["Precision", "Recall", "F1", "F2"]

    ].T.plot(

        kind="bar",

        figsize=(11, 6)
    )

    ax.set_title(
        "Final Model - Before and After Threshold Tuning"
    )

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "final_model_comparison.png",

        dpi=250
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 2 : MATRICES DE CONFUSION
    # --------------------------------------

    fig, axes = plt.subplots(

        1, 2,

        figsize=(12, 5)
    )

    for ax, (_, row) in zip(

        axes,

        comparison.iterrows()
    ):

        matrix = np.array([

            [
                row["True_Negatives"],
                row["False_Positives"]
            ],

            [
                row["False_Negatives"],
                row["True_Positives"]
            ]

        ], dtype=int)

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

            f"{row['Strategy']} - "

            f"Threshold {row['Threshold']:.5g}"
        )

        ax.set_xlabel("Predicted Class")

        ax.set_ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "final_confusion_comparison.png",

        dpi=250
    )

    plt.close()

    return metadata


# ==========================================
# 10. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 65)

    print("FRAUDDETECT AI - Threshold Tuning")

    print("=" * 65)

    # 1. Charger les données de validation.

    X_val, y_val = load_validation_data()

    # 2. Charger les candidats.

    candidates = load_candidates()

    all_results = []

    # 3. Optimiser les seuils de chaque modèle.

    for candidate in candidates:

        results, selected = analyze_model(

            candidate,

            X_val,

            y_val
        )

        all_results.append(selected)

        plot_threshold_curves(

            results,

            selected
        )

    # 4. Fusionner tous les résultats.

    comparison = pd.concat(

        all_results,

        ignore_index=True
    )

    # 5. Comparer les modèles.

    compare_all_models(comparison)

    # 6. Sélectionner le modèle et son seuil.

    select_final_model(

        comparison,

        X_val.columns
    )

    print("\n" + "=" * 65)

    print("Threshold tuning completed successfully!")

    print(f"Reports saved in: {OUTPUT_DIR}")

    print("No model training performed.")

    print("Test Set remains untouched.")

    print("=" * 65)


if __name__ == "__main__":

    main()
