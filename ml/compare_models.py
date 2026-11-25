
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

# Récupérer le dossier racine FraudDetect.
BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR = BASE_DIR / "models"

# Créer un dossier destiné au benchmark général.
BENCHMARK_DIR = REPORTS_DIR / "benchmark"

BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


# Associer chaque dossier au nom de son algorithme.
#
# Clé : nom du dossier.
# Valeur : nom affiché dans les résultats.

ALGORITHMS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network"
}


# Nombre attendu de modèles pour chaque algorithme.

EXPECTED_COUNTS = {
    "logistic_regression": 5,
    "decision_tree": 5,
    "knn": 4,
    "random_forest": 5,
    "xgboost": 5,
    "neural_network": 4
}


# Métriques communes à tous les algorithmes.

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "Average_Precision"
]


# ==========================================
# 2. CHARGEMENT DES RESULTATS
# ==========================================

def load_all_results():

    # Cette liste contiendra les six DataFrames.
    all_results = []

    print("\n===== LOADING MODEL RESULTS =====")

    # Parcourir les six familles d'algorithmes.
    for folder, algorithm_name in ALGORITHMS.items():

        # Construire le chemin du fichier CSV.
        csv_path = (
            REPORTS_DIR / folder / "comparison.csv"
        )

        # Vérifier que le fichier existe.
        if not csv_path.exists():

            raise FileNotFoundError(
                f"Missing comparison file: {csv_path}"
            )

        # Charger les résultats de cet algorithme.
        df = pd.read_csv(csv_path)

        # Vérifier les colonnes indispensables.
        required_columns = [
            "Model",
            *METRICS,
            "Training_Time",
            "Prediction_Time",
            "False_Positives",
            "False_Negatives",
            "True_Positives",
            "True_Negatives"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{algorithm_name}: "
                f"Missing columns {missing_columns}"
            )

        # Vérifier que nous avons bien récupéré
        # le nombre attendu de modèles.
        expected = EXPECTED_COUNTS[folder]

        if len(df) != expected:

            raise ValueError(
                f"{algorithm_name}: "
                f"Expected {expected} models, "
                f"found {len(df)}."
            )

        if df["Model"].duplicated().any():

            raise ValueError(
                f"Duplicate strategies: {algorithm_name}"
            )

        # Ajouter une colonne indiquant
        # à quelle famille appartient le modèle.
        df["Algorithm"] = algorithm_name

        df["Algorithm_Folder"] = folder

        # La colonne Model contient actuellement
        # la stratégie : original, smote, etc.
        #
        # Nous la renommons Strategy.

        df = df.rename(
            columns={"Model": "Strategy"}
        )

        # Créer un identifiant unique.
        #
        # Exemple :
        # XGBoost - balanced

        df["Model_Name"] = (
            df["Algorithm"]
            + " - "
            + df["Strategy"]
        )

        # Les premiers scripts ne calculaient pas
        # tous la taille des fichiers .joblib.
        #
        # Nous allons récupérer directement la
        # taille de chaque modèle sur le disque.

        model_sizes = []

        for strategy in df["Strategy"]:

            model_path = (
                MODELS_DIR
                / folder
                / f"{strategy}.joblib"
            )

            if model_path.exists():

                size_mb = (
                    model_path.stat().st_size
                    / (1024 ** 2)
                )

            else:

                print(
                    f"Warning: Missing model "
                    f"{model_path}"
                )

                # NaN signifie que la valeur
                # est indisponible.
                size_mb = float("nan")

            model_sizes.append(size_mb)

        df["Model_Size_MB"] = model_sizes

        # Ajouter le DataFrame à notre liste.
        all_results.append(df)

        print(
            f"{algorithm_name}: "
            f"{len(df)} models loaded."
        )

    # Fusionner verticalement les six DataFrames.
    #
    # ignore_index=True permet de reconstruire
    # un index unique pour le tableau final.

    global_results = pd.concat(
        all_results,
        ignore_index=True
    )

    print(
        f"\nTotal models loaded: "
        f"{len(global_results)}"
    )

    return global_results


# ==========================================
# 3. VERIFICATION DES RESULTATS
# ==========================================

def validate_results(df):

    print("\n===== VALIDATING BENCHMARK =====")

    # Vérifier le nombre total de modèles.
    expected_total = sum(
        EXPECTED_COUNTS.values()
    )

    if len(df) != expected_total:

        raise ValueError(
            f"Expected {expected_total} models, "
            f"found {len(df)}."
        )

    # Vérifier que chaque modèle a un nom unique.
    if df["Model_Name"].duplicated().any():

        raise ValueError(
            "Duplicate models detected."
        )

    # Vérifier que les métriques sont comprises
    # entre 0 et 1.

    for metric in METRICS:

        if df[metric].isna().any():

            raise ValueError(
                f"Missing values in {metric}"
            )

        if not df[metric].between(0, 1).all():

            raise ValueError(
                f"Invalid values in {metric}"
            )

    # Vérifier que tous les modèles ont été
    # évalués sur le même nombre d'observations.

    df["Validation_Size"] = (
        df["True_Positives"]
        + df["True_Negatives"]
        + df["False_Positives"]
        + df["False_Negatives"]
    )

    if df["Validation_Size"].nunique() != 1:

        raise ValueError(
            "Models have different validation sizes."
        )

    # Vérifier que le nombre de fraudes réelles
    # est identique pour chaque évaluation.

    df["Actual_Frauds"] = (
        df["True_Positives"]
        + df["False_Negatives"]
    )

    if df["Actual_Frauds"].nunique() != 1:

        raise ValueError(
            "Validation fraud counts are different."
        )

    print(
        f"Validation transactions: "
        f"{df['Validation_Size'].iloc[0]}"
    )

    print(
        f"Actual validation frauds: "
        f"{df['Actual_Frauds'].iloc[0]}"
    )

    print("Benchmark validation successful!")

    return df


# ==========================================
# 4. CLASSEMENT GENERAL
# ==========================================

def rank_models(df):

    # Classer les modèles selon Average Precision.
    #
    # Le F1-score départage les modèles ayant
    # éventuellement la même Average Precision.

    ranking = df.sort_values(
        by=["Average_Precision", "F1"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Attribuer un numéro de classement.
    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1)
    )

    print("\n===== GLOBAL MODEL RANKING =====")

    display_columns = [
        "Rank",
        "Algorithm",
        "Strategy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Average_Precision"
    ]

    print(
        ranking[display_columns]
        .round(4)
        .to_string(index=False)
    )

    # Sauvegarder tous les résultats.
    ranking.to_csv(
        BENCHMARK_DIR / "global_comparison.csv",
        index=False
    )

    return ranking


# ==========================================
# 5. MEILLEUR MODELE PAR ALGORITHME
# ==========================================

def get_best_by_algorithm(ranking):

    # Le classement est déjà trié selon AP.
    #
    # groupby().head(1) permet de conserver
    # la première ligne de chaque famille.

    best_models = (
        ranking
        .groupby("Algorithm", sort=False)
        .head(1)
        .copy()
    )

    print("\n===== BEST MODEL BY ALGORITHM =====")

    columns = [
        "Algorithm",
        "Strategy",
        "Precision",
        "Recall",
        "F1",
        "Average_Precision"
    ]

    print(
        best_models[columns]
        .round(4)
        .to_string(index=False)
    )

    best_models.to_csv(
        BENCHMARK_DIR / "best_by_algorithm.csv",
        index=False
    )

    return best_models


# ==========================================
# 6. MEILLEUR MODELE PAR METRIQUE
# ==========================================

def get_best_by_metric(ranking):

    # Chaque métrique peut avoir
    # un modèle gagnant différent.

    winners = []

    for metric in METRICS:

        # idxmax() retourne l'index
        # de la valeur maximale.
        index = ranking[metric].idxmax()

        best = ranking.loc[index]

        winners.append({
            "Metric": metric,
            "Algorithm": best["Algorithm"],
            "Strategy": best["Strategy"],
            "Score": best[metric]
        })

    winners_df = pd.DataFrame(winners)

    print("\n===== BEST MODEL BY METRIC =====")

    print(
        winners_df
        .round(4)
        .to_string(index=False)
    )

    winners_df.to_csv(
        BENCHMARK_DIR / "best_by_metric.csv",
        index=False
    )

    return winners_df


# ==========================================
# 7. GRAPHIQUE DU CLASSEMENT GENERAL
# ==========================================

def plot_global_ranking(ranking):

    plt.figure(figsize=(12, 15))

    # Créer un graphique horizontal
    # contenant les 28 modèles.

    sns.barplot(
        data=ranking,
        x="Average_Precision",
        y="Model_Name",
        hue="Algorithm",
        dodge=False
    )

    plt.title(
        "FraudDetect AI - Global Model Ranking"
    )

    plt.xlabel("Average Precision")
    plt.ylabel("Model")
    plt.xlim(0, 1)

    plt.legend(
        title="Algorithm",
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "global_ranking.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. COMPARAISON DES SIX ALGORITHMES
# ==========================================

def plot_algorithm_comparison(best_models):

    # Sélectionner les quatre métriques
    # les plus importantes pour notre projet.

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "Average_Precision"
    ]

    plot_data = best_models.set_index(
        "Algorithm"
    )[metrics]

    # Créer un graphique à barres groupées.
    ax = plot_data.plot(
        kind="bar",
        figsize=(14, 7)
    )

    ax.set_title(
        "Best Model of Each Algorithm"
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "algorithm_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 9. COMPARAISON DES MODELES ORIGINAUX
# ==========================================

def plot_original_comparison(ranking):

    # Ne conserver que les modèles entraînés
    # sans rééquilibrage des classes.

    original_models = ranking[
        ranking["Strategy"] == "original"
    ].copy()

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "Average_Precision"
    ]

    ax = original_models.set_index(
        "Algorithm"
    )[metrics].plot(
        kind="bar",
        figsize=(14, 7)
    )

    ax.set_title(
        "Original Dataset - Algorithm Comparison"
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "original_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 10. PRECISION / RECALL TRADEOFF
# ==========================================

def plot_precision_recall_tradeoff(ranking):

    plt.figure(figsize=(12, 8))

    # Chaque point représente un modèle.
    #
    # Axe X : Recall.
    # Axe Y : Precision.
    #
    # La taille du point dépend du F1-score.
    # La couleur représente l'algorithme.

    sns.scatterplot(
        data=ranking,
        x="Recall",
        y="Precision",
        hue="Algorithm",
        style="Strategy",
        size="F1",
        sizes=(50, 300),
        alpha=0.8
    )

    plt.title(
        "Fraud Detection - Precision Recall Tradeoff"
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
        BENCHMARK_DIR
        / "precision_recall_tradeoff.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 11. PERFORMANCE VS VITESSE
# ==========================================

def plot_performance_vs_speed(ranking):

    # Une échelle logarithmique nécessite
    # des temps strictement positifs.

    plot_data = ranking[
        ranking["Prediction_Time"] > 0
    ]

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=plot_data,
        x="Prediction_Time",
        y="Average_Precision",
        hue="Algorithm",
        style="Strategy",
        size="F1",
        sizes=(50, 300),
        alpha=0.8
    )

    # KNN est beaucoup plus lent que certains
    # modèles. Une échelle logarithmique rend
    # les écarts plus faciles à visualiser.

    plt.xscale("log")

    plt.title(
        "Model Performance vs Prediction Time"
    )

    plt.xlabel(
        "Prediction Time (seconds, log scale)"
    )

    plt.ylabel("Average Precision")
    plt.ylim(0, 1.05)

    plt.grid(alpha=0.2)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "performance_vs_speed.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 12. PERFORMANCE VS TAILLE
# ==========================================

def plot_performance_vs_size(ranking):

    # Ignorer les modèles dont la taille
    # est inconnue ou non positive.

    plot_data = ranking[
        ranking["Model_Size_MB"] > 0
    ]

    if plot_data.empty:

        print("No model sizes available.")
        return

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=plot_data,
        x="Model_Size_MB",
        y="Average_Precision",
        hue="Algorithm",
        style="Strategy",
        size="F1",
        sizes=(50, 300),
        alpha=0.8
    )

    plt.xscale("log")

    plt.title(
        "Model Performance vs Model Size"
    )

    plt.xlabel("Model Size (MiB, log scale)")
    plt.ylabel("Average Precision")
    plt.ylim(0, 1.05)

    plt.grid(alpha=0.2)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "performance_vs_size.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 13. HEATMAP DES METRIQUES
# ==========================================

def plot_metrics_heatmap(ranking):

    # Construire un tableau dont chaque ligne
    # représente un modèle et chaque colonne
    # représente une métrique.

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Average_Precision"
    ]

    heatmap_data = ranking.set_index(
        "Model_Name"
    )[metrics]

    plt.figure(figsize=(11, 15))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.3
    )

    plt.title(
        "FraudDetect AI - Global Metrics Heatmap"
    )

    plt.xlabel("Evaluation Metric")
    plt.ylabel("Model")

    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "metrics_heatmap.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 14. ANALYSE DES STRATEGIES
# ==========================================

def plot_sampling_comparison(ranking):

    # pivot() transforme les valeurs d'une
    # colonne en nouvelles colonnes.
    #
    # Lignes : algorithmes.
    # Colonnes : stratégies.
    # Valeurs : Average Precision.

    sampling_data = ranking.pivot(
        index="Algorithm",
        columns="Strategy",
        values="Average_Precision"
    )

    sampling_data = sampling_data.reindex(
        columns=[
            "original",
            "balanced",
            "undersampling",
            "oversampling",
            "smote"
        ]
    )

    # Sauvegarder le tableau de comparaison.
    sampling_data.to_csv(
        BENCHMARK_DIR / "sampling_comparison.csv"
    )

    plt.figure(figsize=(11, 6))

    sns.heatmap(
        sampling_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5
    )

    plt.title(
        "Impact of Sampling Strategies"
    )

    plt.xlabel("Training Strategy")
    plt.ylabel("Algorithm")

    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "sampling_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 15. ANALYSE DES ERREURS
# ==========================================

def plot_error_comparison(best_models):

    # Comparer les faux positifs des meilleurs
    # modèles de chaque famille.
    #
    # Un faux positif correspond à une
    # transaction légitime signalée comme fraude.

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=best_models,
        x="Algorithm",
        y="False_Positives",
        ax=ax,
        color="steelblue"
    )

    ax.set_title(
        "Best Models - False Positives"
    )

    ax.set_ylabel("Number of False Positives")
    ax.set_xlabel("Algorithm")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "false_positives.png",
        dpi=300
    )

    plt.close()

    # Comparer maintenant les faux négatifs.
    #
    # Un faux négatif correspond à une
    # véritable fraude non détectée.

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=best_models,
        x="Algorithm",
        y="False_Negatives",
        ax=ax,
        color="indianred"
    )

    ax.set_title(
        "Best Models - False Negatives"
    )

    ax.set_ylabel("Number of False Negatives")
    ax.set_xlabel("Algorithm")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        BENCHMARK_DIR / "false_negatives.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 16. SELECTION DES CANDIDATS
# ==========================================

def select_candidates(best_models):

    # Sélectionner les trois meilleures
    # familles selon Average Precision.
    #
    # Nous choisissons une variante par famille
    # pour diversifier les candidats.

    candidates = best_models.head(3).copy()

    columns = [
        "Rank",
        "Algorithm",
        "Strategy",
        "Precision",
        "Recall",
        "F1",
        "Average_Precision",
        "Training_Time",
        "Prediction_Time",
        "Model_Size_MB"
    ]

    print("\n===== OPTIMIZATION CANDIDATES =====")

    print(
        candidates[columns]
        .round(5)
        .to_string(index=False)
    )

    # Sauvegarder les candidats qui seront
    # utilisés lors du GO 14.

    candidates.to_csv(
        BENCHMARK_DIR / "optimization_candidates.csv",
        index=False
    )

    return candidates


# ==========================================
# 17. AFFICHAGE DU MEILLEUR MODELE
# ==========================================

def display_best_model(ranking):

    best = ranking.iloc[0]

    print("\n" + "=" * 60)

    print("BEST OVERALL MODEL - VALIDATION")

    print("=" * 60)

    print(f"Algorithm: {best['Algorithm']}")

    print(f"Strategy: {best['Strategy']}")

    print(
        f"Average Precision: "
        f"{best['Average_Precision']:.6f}"
    )

    print(f"Precision: {best['Precision']:.6f}")

    print(f"Recall: {best['Recall']:.6f}")

    print(f"F1-score: {best['F1']:.6f}")

    print(f"ROC AUC: {best['ROC_AUC']:.6f}")

    print(
        f"False Positives: "
        f"{int(best['False_Positives'])}"
    )

    print(
        f"Missed Frauds: "
        f"{int(best['False_Negatives'])}"
    )

    print(
        f"Training Time: "
        f"{best['Training_Time']:.4f}s"
    )

    print(
        f"Prediction Time: "
        f"{best['Prediction_Time']:.6f}s"
    )

    print(
        f"Model Size: "
        f"{best['Model_Size_MB']:.6f} MiB"
    )

    print("\nThis model is not yet the final model.")

    print("Hyperparameter optimization is required.")


# ==========================================
# 18. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)

    print("FRAUDDETECT AI - Global Model Benchmark")

    print("=" * 60)

    # Etape 1 : charger tous les résultats.
    results = load_all_results()

    # Etape 2 : vérifier leur cohérence.
    results = validate_results(results)

    # Etape 3 : classer les 28 modèles.
    ranking = rank_models(results)

    # Etape 4 : identifier les meilleurs
    # modèles de chaque algorithme.
    best_models = get_best_by_algorithm(
        ranking
    )

    # Etape 5 : trouver les gagnants
    # pour chaque métrique.
    get_best_by_metric(ranking)

    # Etape 6 : générer les graphiques.
    print("\n===== GENERATING BENCHMARK CHARTS =====")

    plot_global_ranking(ranking)

    plot_algorithm_comparison(best_models)

    plot_original_comparison(ranking)

    plot_precision_recall_tradeoff(ranking)

    plot_performance_vs_speed(ranking)

    plot_performance_vs_size(ranking)

    plot_metrics_heatmap(ranking)

    plot_sampling_comparison(ranking)

    plot_error_comparison(best_models)

    # Etape 7 : sélectionner les candidats.
    select_candidates(best_models)

    # Etape 8 : afficher le meilleur modèle.
    display_best_model(ranking)

    print("\n" + "=" * 60)

    print("Global benchmark completed successfully!")

    print(f"Reports saved in: {BENCHMARK_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()
