
from pathlib import Path
from time import perf_counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Importer l'algorithme Decision Tree.
from sklearn.tree import DecisionTreeClassifier

# Importer la fonction permettant de dessiner un arbre.
from sklearn.tree import plot_tree

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

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODELS_DIR = BASE_DIR / "models" / "decision_tree"

REPORTS_DIR = BASE_DIR / "reports" / "decision_tree"

# Créer les dossiers de sauvegarde.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Seuil de classification des transactions.
THRESHOLD = 0.5


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

def load_dataset(suffix=""):

    # Charger les variables d'entrée selon
    # la stratégie de rééquilibrage choisie.
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
    # que pour la régression logistique.
    #
    # Cela permet une comparaison équitable.

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    return X_val, y_val


# ==========================================
# 3. ENTRAINEMENT DU DECISION TREE
# ==========================================

def train_model(X_train, y_train, class_weight=None):

    # Créer un arbre de décision non entraîné.
    #
    # criterion="gini" :
    # Utiliser l'indice de Gini pour choisir
    # les meilleures séparations.
    #
    # max_depth=10 :
    # Limiter la profondeur de l'arbre pour
    # réduire le risque de surapprentissage.
    #
    # min_samples_split=20 :
    # Un noeud doit contenir au moins
    # 20 observations pour être divisé.
    #
    # min_samples_leaf=10 :
    # Chaque feuille doit contenir au moins
    # 10 observations.
    #
    # class_weight :
    # Permet de donner plus d'importance
    # aux transactions frauduleuses.

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=class_weight,
        random_state=RANDOM_STATE
    )

    # Démarrer le chronomètre.
    start_time = perf_counter()

    # ======================================
    # APPRENTISSAGE DU MODELE
    # ======================================
    #
    # L'algorithme recherche automatiquement
    # les variables et les seuils permettant
    # de séparer les deux classes.

    model.fit(X_train, y_train)

    # Calculer la durée de l'entraînement.
    training_time = perf_counter() - start_time

    print(f"Training time: {training_time:.4f} seconds")

    # Informations spécifiques aux arbres.
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print(f"Number of nodes: {model.tree_.node_count}")

    return model, training_time


# ==========================================
# 4. EVALUATION DU MODELE
# ==========================================

def evaluate_model(model, X_val, y_val, name):

    start_time = perf_counter()

    # Récupérer les scores associés à la fraude.
    #
    # Pour un arbre, ces scores reposent sur
    # la proportion éventuellement pondérée
    # des classes dans la feuille atteinte.

    probabilities = model.predict_proba(X_val)[:, 1]

    # Transformer les scores en prédictions 0 ou 1.
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

    # Calculer les mêmes métriques que celles
    # utilisées pour la régression logistique.

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

        "Prediction_Time": prediction_time,

        # Ajouter les caractéristiques de l'arbre.
        "Tree_Depth": model.get_depth(),
        "Number_of_Leaves": model.get_n_leaves(),
        "Number_of_Nodes": model.tree_.node_count
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

    plt.title(f"Decision Tree - {name}")

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

    # Enregistrer l'arbre entraîné sur le disque.
    joblib.dump(model, model_path)

    print(f"\nModel saved: {model_path}")


# ==========================================
# 7. IMPORTANCE DES VARIABLES
# ==========================================

def save_feature_importance(model, columns, name):

    # feature_importances_ contient une mesure
    # d'importance calculée à partir des
    # réductions d'impureté dans l'arbre.

    importance = pd.DataFrame({
        "Feature": columns,
        "Importance": model.feature_importances_
    })

    # Trier les variables de la plus importante
    # à la moins importante.
    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n===== TOP 10 IMPORTANT FEATURES =====")

    print(
        importance.head(10).to_string(index=False)
    )

    # Sauvegarder les importances des 30 variables.
    importance.to_csv(
        REPORTS_DIR / f"importance_{name}.csv",
        index=False
    )

    # Sélectionner les 15 premières variables
    # pour obtenir un graphique lisible.
    top_features = importance.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
        color="steelblue"
    )

    # Afficher les variables les plus importantes
    # en haut du graphique.
    plt.gca().invert_yaxis()

    plt.title(f"Feature Importance - {name}")

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"importance_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. VISUALISATION DE L'ARBRE
# ==========================================

def save_tree_visualization(model, columns, name):

    # Créer une grande figure pour afficher
    # les différentes branches de l'arbre.
    plt.figure(figsize=(24, 12))

    # Dessiner les premiers niveaux de l'arbre.
    #
    # feature_names : noms des variables.
    #
    # class_names : noms des classes.
    #
    # filled=True : colorer les noeuds
    # selon la classe majoritaire.
    #
    # max_depth=3 : afficher seulement les
    # premiers niveaux pour rester lisible.
    #
    # Attention : cela ne modifie pas
    # la profondeur réelle du modèle.

    plot_tree(
        model,
        feature_names=columns.tolist(),
        class_names=["Legitimate", "Fraud"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=9
    )

    plt.title(f"Decision Tree Visualization - {name}")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"tree_{name}.png",
        dpi=200
    )

    plt.close()

    print("Tree visualization saved successfully!")


# ==========================================
# 9. COMPARAISON DES MODELES
# ==========================================

def compare_models(all_results):

    # Transformer les résultats en DataFrame.
    comparison = pd.DataFrame(all_results)

    # Classer les modèles selon Average Precision,
    # comme pour notre régression logistique.
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
        "Training_Time",
        "Tree_Depth",
        "Number_of_Leaves"
    ]

    print(
        comparison[columns]
        .round(4)
        .to_string(index=False)
    )

    # Enregistrer les résultats complets.
    comparison.to_csv(
        REPORTS_DIR / "comparison.csv",
        index=False
    )

    # Comparer les quatre principales métriques.
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
        "Decision Tree - Sampling Comparison"
    )

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

    # Identifier la meilleure stratégie.
    best_model = comparison.iloc[0]

    print("\n===== BEST DECISION TREE =====")

    print(f"Strategy: {best_model['Model']}")

    print(
        f"Average Precision: "
        f"{best_model['Average_Precision']:.6f}"
    )

    print(f"F1-score: {best_model['F1']:.6f}")

    # Récupérer le meilleur modèle sauvegardé.
    selected_model = joblib.load(
        MODELS_DIR / f"{best_model['Model']}.joblib"
    )

    # Enregistrer une copie sous un nom commun.
    joblib.dump(
        selected_model,
        MODELS_DIR / "best_model.joblib"
    )


# ==========================================
# 10. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)
    print("FRAUDDETECT AI - Decision Tree Training")
    print("=" * 60)

    # Charger le jeu de validation.
    X_val, y_val = load_validation_data()

    # Définir les cinq expériences.

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

    # Liste des résultats de chaque expérience.
    all_results = []

    # Entraîner les cinq variantes de l'arbre.
    for experiment in experiments:

        name = experiment["name"]

        print("\n" + "=" * 60)
        print(f"TRAINING: {name.upper()}")
        print("=" * 60)

        # Charger les données d'entraînement.
        X_train, y_train = load_dataset(
            experiment["suffix"]
        )

        print(f"Training shape: {X_train.shape}")

        print("\nClass distribution:")
        print(y_train.value_counts())

        # Entraîner notre arbre de décision.
        model, training_time = train_model(
            X_train,
            y_train,
            experiment["class_weight"]
        )

        # Evaluer le modèle sur la validation.
        results, matrix = evaluate_model(
            model,
            X_val,
            y_val,
            name
        )

        results["Training_Time"] = training_time

        # Sauvegarder le modèle entraîné.
        save_model(model, name)

        # Générer la matrice de confusion.
        save_confusion_matrix(matrix, name)

        # Analyser l'importance des variables.
        save_feature_importance(
            model,
            X_train.columns,
            name
        )

        # Dessiner l'arbre de décision.
        save_tree_visualization(
            model,
            X_train.columns,
            name
        )

        # Conserver les résultats.
        all_results.append(results)

    # Comparer les cinq variantes entraînées.
    compare_models(all_results)

    print("\n" + "=" * 60)
    print("Decision Tree training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
