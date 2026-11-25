
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

# Récupérer le dossier racine du projet.
BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

# Créer un dossier dédié au benchmark final.
OUTPUT_DIR = REPORTS_DIR / "final_benchmark"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Les six familles d'algorithmes étudiées.
ALGORITHMS = {

    "logistic_regression": "Logistic Regression",

    "decision_tree": "Decision Tree",

    "knn": "KNN",

    "random_forest": "Random Forest",

    "xgboost": "XGBoost",

    "neural_network": "Neural Network"
}

# Les trois algorithmes que nous avons optimisés.
OPTIMIZED_ALGORITHMS = [

    "xgboost",

    "random_forest",

    "neural_network"
]

# Métriques utilisées pour les comparaisons.
METRICS = [

    "Precision",

    "Recall",

    "F1",

    "ROC_AUC",

    "Average_Precision"
]


# ==========================================
# 2. NORMALISATION DES COLONNES
# ==========================================

def normalize_columns(df):

    # Certains rapports peuvent utiliser
    # des noms légèrement différents.
    #
    # Cette fonction uniformise les noms
    # avant de fusionner les DataFrames.

    df = df.copy()

    aliases = {

        "model": "Model",

        "strategy": "Model",

        "sampling_strategy": "Model",

        "precision": "Precision",

        "recall": "Recall",

        "f1": "F1",

        "f1_score": "F1",

        "roc_auc": "ROC_AUC",

        "average_precision": "Average_Precision",

        "ap": "Average_Precision",

        "accuracy": "Accuracy",

        "training_time": "Training_Time",

        "prediction_time": "Prediction_Time",

        "model_size_mb": "Model_Size_MB",

        "model_size_mib": "Model_Size_MB"
    }

    rename_mapping = {}

    for column in df.columns:

        key = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if key in aliases:

            rename_mapping[column] = aliases[key]

    df = df.rename(columns=rename_mapping)

    # Si deux colonnes ont été renommées
    # avec le même nom, garder la première.

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    return df


# ==========================================
# 3. CHARGEMENT DES RAPPORTS
# ==========================================

def read_report(path):

    # Vérifier que le fichier existe.
    if not path.exists():

        raise FileNotFoundError(
            f"Report not found: {path}"
        )

    df = pd.read_csv(path)

    df = normalize_columns(df)

    required_columns = [

        "Model",

        "Precision",

        "Recall",

        "F1",

        "Average_Precision"
    ]

    # Vérifier la présence des colonnes
    # indispensables au benchmark.

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(

            f"Missing columns in {path}: "
            f"{missing_columns}"
        )

    # Certains rapports peuvent ne pas
    # contenir toutes les métriques techniques.
    #
    # Une donnée absente reste NaN.
    # Elle n'est jamais remplacée par zéro.

    optional_columns = [

        "Accuracy",

        "ROC_AUC",

        "Training_Time",

        "Prediction_Time",

        "Model_Size_MB"
    ]

    for column in optional_columns:

        if column not in df.columns:

            df[column] = np.nan

    # Convertir les métriques en valeurs
    # numériques pour pouvoir les comparer.

    numeric_columns = [

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

    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce"
        )

    # Vérifier que les scores indispensables
    # sont bien compris entre 0 et 1.

    for column in required_columns[1:]:

        if (

            df[column].isna().any()

            or not df[column].between(0, 1).all()

        ):

            raise ValueError(

                f"Invalid metric {column} "
                f"in {path}"
            )

    print(f"Loaded: {path.name}")

    return df


# ==========================================
# 4. CHARGEMENT DES 28 MODELES INITIAUX
# ==========================================

def load_baseline_results():

    all_results = []

    print("\n===== BASELINE MODELS =====")

    for folder, algorithm in ALGORITHMS.items():

        path = (

            REPORTS_DIR

            / folder

            / "comparison.csv"
        )

        df = read_report(path)

        # Ajouter le nom de l'algorithme.
        df["Algorithm"] = algorithm

        df["Algorithm_Folder"] = folder

        # Identifier la stratégie d'entraînement.
        df["Strategy"] = (

            df["Model"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Marquer ces modèles comme initiaux.
        df["Version"] = "Baseline"

        # Construire un identifiant lisible.
        df["Model_Name"] = (

            algorithm

            + " - "

            + df["Strategy"]
        )

        # Construire le chemin du modèle
        # réellement sauvegardé.

        df["Model_Path"] = df["Strategy"].apply(

            lambda strategy: str(

                Path("models")

                / folder

                / f"{strategy}.joblib"
            )
        )

        all_results.append(df)

        print(

            f"{algorithm}: "

            f"{len(df)} models"
        )

    # Fusionner verticalement les six rapports.
    results = pd.concat(

        all_results,

        ignore_index=True
    )

    print(

        f"\nTotal baseline models: "

        f"{len(results)}"
    )

    return results


# ==========================================
# 5. CHARGEMENT DES 3 MODELES OPTIMISES
# ==========================================

def load_optimized_results():

    all_results = []

    print("\n===== OPTIMIZED MODELS =====")

    for folder in OPTIMIZED_ALGORITHMS:

        algorithm = ALGORITHMS[folder]

        path = (

            REPORTS_DIR

            / "optimization"

            / folder

            / "before_after_comparison.csv"
        )

        df = read_report(path)

        # Les rapports d'optimisation contiennent
        # également les modèles initiaux.
        #
        # Nous sélectionnons uniquement la
        # nouvelle version optimisée pour éviter
        # de compter deux fois les baselines.

        names = (

            df["Model"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        optimized = df[

            names.str.contains(
                r"\boptimized\b",
                regex=True
            )

        ].copy()

        if len(optimized) != 1:

            raise ValueError(

                f"Expected exactly one optimized "
                f"model in {path}, "
                f"found {len(optimized)}"
            )

        optimized["Algorithm"] = algorithm

        optimized["Algorithm_Folder"] = folder

        optimized["Strategy"] = "optimized"

        optimized["Version"] = "Optimized"

        optimized["Model_Name"] = (

            algorithm + " - optimized"
        )

        optimized["Model_Path"] = str(

            Path("models")

            / "optimized"

            / folder

            / "optimized_model.joblib"
        )

        all_results.append(optimized)

        print(

            f"{algorithm}: optimized model loaded"
        )

    results = pd.concat(

        all_results,

        ignore_index=True
    )

    print(

        f"\nTotal optimized models: "

        f"{len(results)}"
    )

    return results


# ==========================================
# 6. CONSTRUCTION DU BENCHMARK GLOBAL
# ==========================================

def build_benchmark(baselines, optimized):

    # Réunir tous nos modèles.
    results = pd.concat(

        [baselines, optimized],

        ignore_index=True
    )

    # Vérifier l'absence de doublons.
    if results["Model_Name"].duplicated().any():

        raise ValueError(
            "Duplicate model names detected."
        )

    # Vérifier l'inventaire attendu.
    if len(baselines) != 28:

        raise ValueError(

            f"Expected 28 baseline models, "
            f"found {len(baselines)}"
        )

    if len(optimized) != 3:

        raise ValueError(

            f"Expected 3 optimized models, "
            f"found {len(optimized)}"
        )

    # Vérifier que les fichiers des modèles
    # existent réellement sur la machine.

    results["Model_Available"] = (

        results["Model_Path"].apply(

            lambda path: (
                BASE_DIR / path
            ).is_file()
        )
    )

    # Classer les modèles selon notre
    # métrique principale : Average Precision.
    #
    # En cas d'égalité, utiliser le F1-score.

    results = results.sort_values(

        by=[

            "Average_Precision",

            "F1",

            "Model_Name"
        ],

        ascending=[False, False, True]

    ).reset_index(drop=True)

    # Ajouter le classement général.
    results.insert(

        0,

        "Rank",

        range(1, len(results) + 1)
    )

    # Calculer des classements supplémentaires.

    results["F1_Rank"] = (

        results["F1"]

        .rank(
            ascending=False,
            method="min"
        )
    )

    results["Recall_Rank"] = (

        results["Recall"]

        .rank(
            ascending=False,
            method="min"
        )
    )

    results["Precision_Rank"] = (

        results["Precision"]

        .rank(
            ascending=False,
            method="min"
        )
    )

    results["Speed_Rank"] = (

        results["Prediction_Time"]

        .rank(
            ascending=True,
            method="min"
        )
    )

    # Exporter le benchmark complet.
    results.to_csv(

        OUTPUT_DIR / "global_benchmark.csv",

        index=False
    )

    print("\n===== GLOBAL BENCHMARK =====")

    print(f"Total models: {len(results)}")

    print(

        results[

            [

                "Rank",

                "Model_Name",

                "Precision",

                "Recall",

                "F1",

                "Average_Precision",

                "Prediction_Time"
            ]

        ].round(5).to_string(index=False)
    )

    return results


# ==========================================
# 7. COMPARAISON AVANT ET APRES OPTIMISATION
# ==========================================

def compare_optimization(results):

    comparisons = []

    for folder in OPTIMIZED_ALGORITHMS:

        algorithm = ALGORITHMS[folder]

        # Récupérer les modèles de cette famille.
        family = results[

            results["Algorithm"] == algorithm
        ]

        # Identifier la meilleure baseline
        # selon Average Precision.

        baselines = family[

            family["Version"] == "Baseline"
        ]

        best_baseline = baselines.loc[

            baselines["Average_Precision"].idxmax()
        ]

        optimized = family[

            family["Version"] == "Optimized"

        ].iloc[0]

        # Mesurer l'évolution des performances.

        ap_gain = (

            optimized["Average_Precision"]

            - best_baseline["Average_Precision"]
        )

        f1_gain = (

            optimized["F1"]

            - best_baseline["F1"]
        )

        recall_gain = (

            optimized["Recall"]

            - best_baseline["Recall"]
        )

        comparisons.append({

            "Algorithm": algorithm,

            "Best_Baseline": best_baseline["Model_Name"],

            "Baseline_AP": best_baseline["Average_Precision"],

            "Optimized_AP": optimized["Average_Precision"],

            # Conversion en points de pourcentage.
            "AP_Gain_Points": ap_gain * 100,

            "F1_Gain_Points": f1_gain * 100,

            "Recall_Gain_Points": recall_gain * 100,

            "Optimization_Improved_AP": bool(
                ap_gain > 0
            )
        })

    comparison = pd.DataFrame(comparisons)

    comparison.to_csv(

        OUTPUT_DIR / "optimization_summary.csv",

        index=False
    )

    print("\n===== OPTIMIZATION SUMMARY =====")

    print(

        comparison.round(5)
        .to_string(index=False)
    )

    return comparison


# ==========================================
# 8. MEILLEUR MODELE DE CHAQUE ALGORITHME
# ==========================================

def select_best_per_algorithm(results):

    # Pour chaque famille, récupérer l'index
    # du modèle possédant la meilleure AP.

    indices = (

        results.groupby("Algorithm")

        ["Average_Precision"]

        .idxmax()
    )

    best_models = results.loc[

        indices

    ].sort_values(

        "Average_Precision",

        ascending=False
    )

    best_models.to_csv(

        OUTPUT_DIR / "best_per_algorithm.csv",

        index=False
    )

    print("\n===== BEST MODEL PER ALGORITHM =====")

    print(

        best_models[

            [

                "Model_Name",

                "Precision",

                "Recall",

                "F1",

                "Average_Precision"
            ]

        ].round(5).to_string(index=False)
    )

    return best_models


# ==========================================
# 9. ANALYSE DE PARETO
# ==========================================

def calculate_pareto_front(results):

    # Un bon modèle doit idéalement posséder :
    #
    # - Une Average Precision élevée.
    # - Un temps de prédiction faible.
    #
    # Un modèle est dominé lorsqu'un autre
    # est au moins aussi performant et rapide,
    # et strictement meilleur sur un critère.

    valid = results.dropna(

        subset=[

            "Average_Precision",

            "Prediction_Time"
        ]

    ).copy()

    valid = valid[

        valid["Prediction_Time"] > 0
    ]

    pareto_models = []

    for index, current in valid.iterrows():

        dominated = False

        for other_index, other in valid.iterrows():

            if index == other_index:

                continue

            better_or_equal_ap = (

                other["Average_Precision"]

                >= current["Average_Precision"]
            )

            faster_or_equal = (

                other["Prediction_Time"]

                <= current["Prediction_Time"]
            )

            strictly_better = (

                other["Average_Precision"]

                > current["Average_Precision"]

                or

                other["Prediction_Time"]

                < current["Prediction_Time"]
            )

            if (

                better_or_equal_ap

                and faster_or_equal

                and strictly_better

            ):

                dominated = True

                break

        if not dominated:

            pareto_models.append(index)

    pareto = valid.loc[

        pareto_models

    ].sort_values("Prediction_Time")

    pareto.to_csv(

        OUTPUT_DIR / "pareto_models.csv",

        index=False
    )

    print("\n===== PARETO FRONT =====")

    print(

        pareto[

            [

                "Model_Name",

                "Average_Precision",

                "Prediction_Time"
            ]

        ].round(6).to_string(index=False)
    )

    return pareto


# ==========================================
# 10. SELECTION POUR LE THRESHOLD TUNING
# ==========================================

def select_candidates(results):

    # Ne sélectionner que les modèles
    # dont le fichier existe réellement.

    available = results[

        results["Model_Available"]
    ].copy()

    if available.empty:

        raise ValueError(
            "No saved model files found."
        )

    selected = {}

    def add_candidate(row, reason):

        name = row["Model_Name"]

        # Si un modèle est sélectionné pour
        # plusieurs raisons, éviter les doublons.

        if name in selected:

            selected[name]["Reasons"].append(
                reason
            )

        else:

            selected[name] = {

                "Model_Name": name,

                "Algorithm": row["Algorithm"],

                "Strategy": row["Strategy"],

                "Model_Path": row["Model_Path"],

                "Average_Precision": float(

                    row["Average_Precision"]
                ),

                "F1": float(row["F1"]),

                "Recall": float(row["Recall"]),

                "Precision": float(row["Precision"]),

                "Reasons": [reason]
            }

    # Les cinq meilleurs modèles selon AP.
    for _, row in available.head(5).iterrows():

        add_candidate(

            row,

            "Top 5 Average Precision"
        )

    # Ajouter le meilleur modèle selon F1.
    best_f1 = available.loc[

        available["F1"].idxmax()
    ]

    add_candidate(

        best_f1,

        "Best F1-score"
    )

    # Ajouter le meilleur modèle selon Recall.
    best_recall = available.loc[

        available["Recall"].idxmax()
    ]

    add_candidate(

        best_recall,

        "Best Recall"
    )

    candidates = list(selected.values())

    # Trier les candidats selon AP.
    candidates.sort(

        key=lambda model: model["Average_Precision"],

        reverse=True
    )

    # Sauvegarder les informations nécessaires
    # à notre prochain programme Python.

    with open(

        OUTPUT_DIR / "threshold_candidates.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            candidates,

            file,

            indent=4,

            ensure_ascii=False
        )

    pd.DataFrame(candidates).to_csv(

        OUTPUT_DIR / "threshold_candidates.csv",

        index=False
    )

    print("\n===== THRESHOLD TUNING CANDIDATES =====")

    for candidate in candidates:

        print(

            f"\n{candidate['Model_Name']}"

            f"\nAP: {candidate['Average_Precision']:.6f}"

            f"\nReasons: {candidate['Reasons']}"
        )

    return candidates


# ==========================================
# 11. GRAPHIQUE DU CLASSEMENT GLOBAL
# ==========================================

def plot_global_ranking(results):

    plt.figure(figsize=(13, 16))

    sns.barplot(

        data=results,

        x="Average_Precision",

        y="Model_Name",

        hue="Algorithm",

        dodge=False
    )

    plt.title(

        "FraudDetect AI - Final Model Ranking",

        fontsize=16
    )

    plt.xlabel("Average Precision")
    plt.ylabel("Model")

    plt.xlim(0, 1.05)

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "global_ranking.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 12. HEATMAP DES PERFORMANCES
# ==========================================

def plot_metrics_heatmap(results):

    # Construire une matrice où chaque
    # ligne représente un modèle et
    # chaque colonne une métrique.

    heatmap_data = results.set_index(

        "Model_Name"

    )[METRICS]

    plt.figure(figsize=(12, 17))

    sns.heatmap(

        heatmap_data,

        annot=True,

        fmt=".3f",

        cmap="RdYlGn",

        vmin=0,

        vmax=1,

        linewidths=0.5
    )

    plt.title(

        "FraudDetect AI - Final Metrics Heatmap"
    )

    plt.xlabel("Evaluation Metric")
    plt.ylabel("Model")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "metrics_heatmap.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 13. IMPACT DES OPTIMISATIONS
# ==========================================

def plot_optimization_impact(comparison):

    # Comparer les meilleurs modèles
    # initiaux avec les modèles optimisés.

    ax = comparison.set_index(

        "Algorithm"

    )[

        ["Baseline_AP", "Optimized_AP"]

    ].plot(

        kind="bar",

        figsize=(11, 6)
    )

    ax.set_title(

        "Average Precision - Before and After Optimization"
    )

    ax.set_ylabel("Average Precision")

    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "optimization_comparison.png",

        dpi=300
    )

    plt.close()

    # Visualiser directement les gains ou pertes.

    colors = [

        "seagreen" if gain > 0

        else "indianred"

        for gain in comparison["AP_Gain_Points"]
    ]

    plt.figure(figsize=(10, 6))

    plt.barh(

        comparison["Algorithm"],

        comparison["AP_Gain_Points"],

        color=colors
    )

    plt.axvline(

        0,

        color="black",

        linewidth=1
    )

    plt.title(

        "Impact of Hyperparameter Optimization"
    )

    plt.xlabel(

        "Average Precision Gain (percentage points)"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "optimization_gains.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 14. MEILLEURS MODELES PAR ALGORITHME
# ==========================================

def plot_best_models(best_models):

    ax = best_models.set_index(

        "Model_Name"

    )[

        [

            "Precision",

            "Recall",

            "F1",

            "Average_Precision"
        ]

    ].plot(

        kind="bar",

        figsize=(15, 7)
    )

    ax.set_title(

        "Best Model of Each Algorithm"
    )

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.05)

    plt.xticks(

        rotation=20,

        ha="right"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "best_algorithms.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 15. PERFORMANCE ET RAPIDITE
# ==========================================

def plot_performance_speed(results, pareto):

    valid = results.dropna(

        subset=["Prediction_Time"]

    ).copy()

    valid = valid[

        valid["Prediction_Time"] > 0
    ]

    if valid.empty:

        print(
            "No prediction times available."
        )

        return

    plt.figure(figsize=(13, 8))

    sns.scatterplot(

        data=valid,

        x="Prediction_Time",

        y="Average_Precision",

        hue="Algorithm",

        style="Version",

        size="F1",

        sizes=(60, 250),

        alpha=0.85
    )

    # Relier les modèles non dominés.
    if not pareto.empty:

        plt.plot(

            pareto["Prediction_Time"],

            pareto["Average_Precision"],

            linestyle="--",

            color="black",

            label="Pareto Front"
        )

    # Une échelle logarithmique permet
    # de comparer des durées très différentes.

    plt.xscale("log")

    plt.title(

        "Model Performance vs Prediction Time"
    )

    plt.xlabel(

        "Prediction Time (seconds, log scale)"
    )

    plt.ylabel("Average Precision")

    plt.legend(

        bbox_to_anchor=(1.02, 1),

        loc="upper left"
    )

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "performance_vs_speed.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 16. COMPARAISON PRECISION / RECALL
# ==========================================

def plot_precision_recall_tradeoff(results):

    plt.figure(figsize=(12, 8))

    sns.scatterplot(

        data=results,

        x="Recall",

        y="Precision",

        hue="Algorithm",

        style="Version",

        size="F1",

        sizes=(60, 250)
    )

    # Identifier les cinq meilleurs modèles.
    for _, row in results.head(5).iterrows():

        plt.annotate(

            str(row["Rank"]),

            (

                row["Recall"],

                row["Precision"]
            ),

            xytext=(6, 5),

            textcoords="offset points",

            fontsize=10
        )

    plt.title(

        "Precision vs Recall - Model Trade-off"
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.xlim(0, 1.05)

    plt.ylim(0, 1.05)

    plt.grid(alpha=0.2)

    plt.legend(

        bbox_to_anchor=(1.02, 1),

        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "precision_recall_tradeoff.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 17. COMPARAISON DES TAILLES DES MODELES
# ==========================================

def plot_model_sizes(results):

    valid = results.dropna(

        subset=["Model_Size_MB"]

    ).copy()

    valid = valid[

        valid["Model_Size_MB"] > 0
    ]

    if valid.empty:

        print("No model sizes available.")

        return

    plt.figure(figsize=(12, 8))

    sns.scatterplot(

        data=valid,

        x="Model_Size_MB",

        y="Average_Precision",

        hue="Algorithm",

        style="Version",

        s=100
    )

    plt.xscale("log")

    plt.title(

        "Model Performance vs Model Size"
    )

    plt.xlabel("Model Size (log scale)")

    plt.ylabel("Average Precision")

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "performance_vs_size.png",

        dpi=300
    )

    plt.close()


# ==========================================
# 18. SAUVEGARDE DU RESUME FINAL
# ==========================================

def save_final_summary(results, candidates):

    best_ap = results.iloc[0]

    best_f1 = results.loc[

        results["F1"].idxmax()
    ]

    best_recall = results.loc[

        results["Recall"].idxmax()
    ]

    best_precision = results.loc[

        results["Precision"].idxmax()
    ]

    # Préparer une structure JSON que nous
    # pourrons exploiter plus tard dans
    # notre backend et notre frontend React.

    def model_summary(row):

        return {

            "name": row["Model_Name"],

            "algorithm": row["Algorithm"],

            "strategy": row["Strategy"],

            "average_precision": float(

                row["Average_Precision"]
            ),

            "precision": float(row["Precision"]),

            "recall": float(row["Recall"]),

            "f1": float(row["F1"]),

            "model_path": row["Model_Path"],

            "available": bool(

                row["Model_Available"]
            )
        }

    summary = {

        "total_models": int(len(results)),

        "total_algorithms": int(

            results["Algorithm"].nunique()
        ),

        "baseline_models": int(

            (results["Version"] == "Baseline").sum()
        ),

        "optimized_models": int(

            (results["Version"] == "Optimized").sum()
        ),

        "available_model_files": int(

            results["Model_Available"].sum()
        ),

        "primary_metric": "Average Precision",

        "evaluation_dataset": "validation",

        "test_set_used": False,

        "best_ap_model": model_summary(best_ap),

        "best_f1_model": model_summary(best_f1),

        "best_recall_model": model_summary(best_recall),

        "best_precision_model": model_summary(

            best_precision
        ),

        "threshold_candidates": candidates
    }

    with open(

        OUTPUT_DIR / "benchmark_summary.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4,

            ensure_ascii=False
        )

    print("\n===== FINAL BENCHMARK SUMMARY =====")

    print(

        f"Total models: "

        f"{summary['total_models']}"
    )

    print(

        f"Best AP model: "

        f"{best_ap['Model_Name']}"
    )

    print(

        f"Best AP: "

        f"{best_ap['Average_Precision']:.6f}"
    )

    print(

        f"Best F1 model: "

        f"{best_f1['Model_Name']}"
    )

    print(

        f"Best Recall model: "

        f"{best_recall['Model_Name']}"
    )

    print(

        f"Best Precision model: "

        f"{best_precision['Model_Name']}"
    )

    print(

        f"Threshold candidates: "

        f"{len(candidates)}"
    )

    missing = results[

        ~results["Model_Available"]
    ]

    if not missing.empty:

        print("\nWARNING: Missing model files:")

        print(

            missing[

                ["Model_Name", "Model_Path"]

            ].to_string(index=False)
        )

    return summary


# ==========================================
# 19. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 65)

    print(

        "FRAUDDETECT AI - Final Model Benchmark"
    )

    print("=" * 65)

    # 1. Charger les 28 modèles initiaux.
    baselines = load_baseline_results()

    # 2. Charger les 3 modèles optimisés.
    optimized = load_optimized_results()

    # 3. Construire le classement global.
    results = build_benchmark(

        baselines,

        optimized
    )

    # 4. Comparer avant et après optimisation.
    comparison = compare_optimization(

        results
    )

    # 5. Identifier le meilleur modèle
    # de chaque famille d'algorithmes.
    best_models = select_best_per_algorithm(

        results
    )

    # 6. Identifier les modèles non dominés.
    pareto = calculate_pareto_front(

        results
    )

    # 7. Sélectionner les candidats.
    candidates = select_candidates(

        results
    )

    # 8. Générer les graphiques.
    plot_global_ranking(results)

    plot_metrics_heatmap(results)

    plot_optimization_impact(comparison)

    plot_best_models(best_models)

    plot_performance_speed(results, pareto)

    plot_precision_recall_tradeoff(results)

    plot_model_sizes(results)

    # 9. Sauvegarder le résumé final.
    save_final_summary(

        results,

        candidates
    )

    print("\n" + "=" * 65)

    print(

        "Final benchmark completed successfully!"
    )

    print(f"Reports saved in: {OUTPUT_DIR}")

    print("No model training performed.")

    print("Test Set remains untouched.")

    print("=" * 65)


if __name__ == "__main__":

    main()
