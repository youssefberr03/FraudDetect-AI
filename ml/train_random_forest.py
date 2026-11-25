
from pathlib import Path
from time import perf_counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Importer notre quatrième algorithme.
from sklearn.ensemble import RandomForestClassifier

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

MODELS_DIR = BASE_DIR / "models" / "random_forest"

REPORTS_DIR = BASE_DIR / "reports" / "random_forest"

# Créer les dossiers s'ils n'existent pas.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Garantir la reproductibilité.
RANDOM_STATE = 42

# Nombre d'arbres dans chaque forêt.
N_ESTIMATORS = 100

# Seuil de classification.
THRESHOLD = 0.5


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

def load_dataset(suffix=""):

    # Charger les variables d'entrée selon
    # la stratégie d'entraînement sélectionnée.
    X_train = pd.read_csv(
        PROCESSED_DIR / f"X_train{suffix}.csv"
    )

    # Charger les réponses attendues.
    y_train = pd.read_csv(
        PROCESSED_DIR / f"y_train{suffix}.csv"
    )["Class"]

    return X_train, y_train


def load_validation_data():

    # Charger le même jeu de validation
    # que pour les algorithmes précédents.
    #
    # Ne jamais rééquilibrer la validation.

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    return X_val, y_val


# ==========================================
# 3. ENTRAINEMENT DU RANDOM FOREST
# ==========================================

def train_model(X_train, y_train, class_weight=None):

    # Créer une forêt aléatoire.
    #
    # n_estimators=100 :
    # Construire 100 arbres de décision.
    #
    # criterion="gini" :
    # Utiliser l'indice de Gini pour déterminer
    # les séparations dans les arbres.
    #
    # max_depth=12 :
    # Limiter la profondeur de chaque arbre.
    #
    # min_samples_split=10 :
    # Un noeud doit contenir au moins 10
    # observations pour pouvoir être divisé.
    #
    # min_samples_leaf=5 :
    # Chaque feuille doit contenir au moins
    # cinq observations.
    #
    # max_features="sqrt" :
    # Examiner un sous-ensemble aléatoire
    # des variables à chaque séparation.
    #
    # bootstrap=True :
    # Chaque arbre reçoit un échantillon
    # aléatoire sélectionné avec remise.
    #
    # n_jobs=-1 :
    # Utiliser les ressources CPU disponibles.

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        criterion="gini",
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting Random Forest training...")

    # Démarrer le chronomètre.
    start_time = perf_counter()

    # ======================================
    # VERITABLE APPRENTISSAGE DU MODELE
    # ======================================
    #
    # Construire les 100 arbres de décision.
    #
    # Chaque arbre apprend ses propres
    # conditions, seuils et branches.

    model.fit(X_train, y_train)

    # Calculer la durée de l'entraînement.
    training_time = perf_counter() - start_time

    print(f"\nTraining time: {training_time:.4f} seconds")

    # estimators_ contient tous les arbres
    # réellement construits par Random Forest.
    print(f"Number of trees: {len(model.estimators_)}")

    # Calculer la profondeur moyenne des arbres.
    average_depth = sum(
        tree.get_depth()
        for tree in model.estimators_
    ) / len(model.estimators_)

    # Calculer le nombre moyen de feuilles.
    average_leaves = sum(
        tree.get_n_leaves()
        for tree in model.estimators_
    ) / len(model.estimators_)

    print(f"Average tree depth: {average_depth:.2f}")
    print(f"Average number of leaves: {average_leaves:.2f}")

    return (
        model,
        training_time,
        average_depth,
        average_leaves
    )


# ==========================================
# 4. EVALUATION DU MODELE
# ==========================================

def evaluate_model(model, X_val, y_val, name):

    start_time = perf_counter()

    # predict_proba() combine les scores
    # obtenus par les différents arbres.
    #
    # [:, 1] permet de récupérer uniquement
    # les scores associés à la classe fraude.

    probabilities = model.predict_proba(X_val)[:, 1]

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

    # Extraire les quatre types de prédictions.
    tn, fp, fn, tp = matrix.ravel()

    # Calculer les mêmes métriques que celles
    # utilisées pour les modèles précédents.
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

        # Ces deux métriques utilisent
        # les scores continus du modèle.
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

        "Number_of_Trees": len(model.estimators_)
    }

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

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )

    plt.title(f"Random Forest - {name}")

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

    model_path = MODELS_DIR / f"{name}.joblib"

    # Sauvegarder la forêt complète.
    #
    # Le fichier contient tous les arbres
    # entraînés et leurs paramètres.

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
# 7. IMPORTANCE DES VARIABLES
# ==========================================

def save_feature_importance(model, columns, name):

    # Random Forest calcule l'importance des
    # variables à partir des réductions
    # d'impureté dans ses différents arbres.

    importance = pd.DataFrame({
        "Feature": columns,
        "Importance": model.feature_importances_
    })

    # Trier les variables par importance.
    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n===== TOP 10 IMPORTANT FEATURES =====")

    print(
        importance.head(10).to_string(index=False)
    )

    # Sauvegarder toutes les importances.
    importance.to_csv(
        REPORTS_DIR / f"importance_{name}.csv",
        index=False
    )

    # Sélectionner les 15 variables principales.
    top_features = importance.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
        color="steelblue"
    )

    # Afficher la variable la plus importante en haut.
    plt.gca().invert_yaxis()

    plt.title(f"Random Forest Feature Importance - {name}")

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"importance_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. ANALYSE DES ARBRES DE LA FORET
# ==========================================

def save_tree_statistics(model, name):

    # Créer une liste contenant les informations
    # de chaque arbre de notre Random Forest.

    statistics = []

    # enumerate() permet de récupérer à la fois
    # le numéro de l'arbre et l'objet correspondant.

    for index, tree in enumerate(
        model.estimators_,
        start=1
    ):

        statistics.append({
            "Tree": index,
            "Depth": tree.get_depth(),
            "Leaves": tree.get_n_leaves(),
            "Nodes": tree.tree_.node_count
        })

    # Transformer les statistiques en DataFrame.
    tree_stats = pd.DataFrame(statistics)

    tree_stats.to_csv(
        REPORTS_DIR / f"tree_statistics_{name}.csv",
        index=False
    )

    print("\n===== TREE STATISTICS =====")

    print(tree_stats.describe())

    # Représenter la distribution du nombre
    # de feuilles dans les différents arbres.
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=tree_stats,
        x="Leaves",
        bins=15
    )

    plt.title(f"Tree Leaf Distribution - {name}")

    plt.xlabel("Number of Leaves")
    plt.ylabel("Number of Trees")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"tree_distribution_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 9. COMPARAISON DES MODELES
# ==========================================

def compare_models(all_results):

    # Construire un tableau contenant
    # les résultats des cinq expériences.
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

    # Sauvegarder tous les résultats.
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

    ax.set_title("Random Forest - Sampling Comparison")

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
    # GRAPHIQUE 2 : TEMPS D'ENTRAINEMENT
    # --------------------------------------

    ax = comparison.set_index("Model")[
        "Training_Time"
    ].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("Random Forest - Training Time")
    ax.set_xlabel("Training Strategy")
    ax.set_ylabel("Time (seconds)")

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "training_time.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 3 : TEMPS DE PREDICTION
    # --------------------------------------

    ax = comparison.set_index("Model")[
        "Prediction_Time"
    ].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("Random Forest - Prediction Time")
    ax.set_xlabel("Training Strategy")
    ax.set_ylabel("Time (seconds)")

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "prediction_time.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 4 : TAILLE DES MODELES
    # --------------------------------------

    ax = comparison.set_index("Model")[
        "Model_Size_MB"
    ].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("Random Forest - Model Size")
    ax.set_xlabel("Training Strategy")
    ax.set_ylabel("Size (MB)")

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

    # La première ligne correspond au modèle
    # ayant la meilleure Average Precision.

    best_model = comparison.iloc[0]

    print("\n===== BEST RANDOM FOREST =====")

    print(f"Strategy: {best_model['Model']}")

    print(
        f"Average Precision: "
        f"{best_model['Average_Precision']:.6f}"
    )

    print(f"Precision: {best_model['Precision']:.6f}")
    print(f"Recall: {best_model['Recall']:.6f}")
    print(f"F1-score: {best_model['F1']:.6f}")

    # Charger le meilleur modèle sauvegardé.
    selected_model = joblib.load(
        MODELS_DIR / f"{best_model['Model']}.joblib"
    )

    # Enregistrer une copie sous un nom commun.
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
    print("FRAUDDETECT AI - Random Forest Training")
    print("=" * 60)

    # Charger le jeu de validation une seule fois.
    X_val, y_val = load_validation_data()

    # Définir les cinq expériences.
    #
    # Commencer par Undersampling permet
    # de vérifier rapidement le fonctionnement.

    experiments = [
        {
            "name": "undersampling",
            "suffix": "_undersampled",
            "class_weight": None
        },
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

    # Liste destinée à stocker les résultats.
    all_results = []

    # Entraîner les cinq forêts indépendamment.
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

        # Etape 2 : entraîner Random Forest.
        (
            model,
            training_time,
            average_depth,
            average_leaves
        ) = train_model(
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

        # Ajouter les caractéristiques du modèle.
        results["Training_Time"] = training_time
        results["Average_Depth"] = average_depth
        results["Average_Leaves"] = average_leaves

        # Etape 4 : sauvegarder le modèle.
        model_size = save_model(model, name)

        results["Model_Size_MB"] = model_size

        # Etape 5 : sauvegarder la matrice.
        save_confusion_matrix(matrix, name)

        # Etape 6 : analyser les variables.
        save_feature_importance(
            model,
            X_train.columns,
            name
        )

        # Etape 7 : analyser les arbres.
        save_tree_statistics(model, name)

        # Etape 8 : conserver les résultats.
        all_results.append(results)

        # Libérer les objets avant de passer
        # à la prochaine expérience.
        del model, X_train, y_train

    # Comparer les cinq modèles entraînés.
    compare_models(all_results)

    print("\n" + "=" * 60)
    print("Random Forest training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
