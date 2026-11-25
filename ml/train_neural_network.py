
from pathlib import Path
from time import perf_counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Importer notre algorithme de réseau de neurones.
from sklearn.neural_network import MLPClassifier

# Importer les outils de standardisation.
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Importer les métriques d'évaluation.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    log_loss
)


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODELS_DIR = BASE_DIR / "models" / "neural_network"

REPORTS_DIR = BASE_DIR / "reports" / "neural_network"

# Créer automatiquement les dossiers.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Garantir la reproductibilité.
RANDOM_STATE = 42

# Architecture des couches cachées.
HIDDEN_LAYERS = (64, 32, 16)

# Nombre de passages sur le dataset.
EPOCHS = 30

# Nombre de transactions par lot.
BATCH_SIZE = 256

# Taux d'apprentissage initial.
LEARNING_RATE = 0.001

# Seuil de classification.
THRESHOLD = 0.5


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

def load_dataset(suffix=""):

    # Charger la version du dataset correspondant
    # à la stratégie d'entraînement choisie.

    X_train = pd.read_csv(
        PROCESSED_DIR / f"X_train{suffix}.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / f"y_train{suffix}.csv"
    )["Class"]

    return X_train, y_train


def load_validation_data():

    # Charger le jeu de validation original.
    #
    # Nous utilisons exactement les mêmes
    # transactions que pour les autres modèles.

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    )["Class"]

    return X_val, y_val


# ==========================================
# 3. STANDARDISATION SUPPLEMENTAIRE
# ==========================================

def create_preprocessor(X_original):

    # Time et Amount ont déjà été standardisées
    # dans notre fichier preprocessing.py.
    #
    # Nous allons maintenant standardiser
    # les 28 composantes PCA restantes.

    pca_columns = [
        f"V{i}" for i in range(1, 29)
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "pca_scaler",
                StandardScaler(),
                pca_columns
            )
        ],
        remainder="passthrough"
    )

    # Ajuster le scaler uniquement sur
    # les données d'entraînement originales.
    #
    # Cela évite toute fuite de données
    # provenant de la validation ou du test.

    preprocessor.fit(X_original)

    print("Neural Network preprocessor fitted!")

    return preprocessor


# ==========================================
# 4. CREATION DU RESEAU DE NEURONES
# ==========================================

def create_model():

    # hidden_layer_sizes :
    # Trois couches cachées de 64, 32 et 16 neurones.
    #
    # activation="relu" :
    # Fonction d'activation des couches cachées.
    #
    # solver="adam" :
    # Algorithme d'optimisation des poids.
    #
    # alpha :
    # Force de régularisation L2.
    #
    # learning_rate_init :
    # Taux d'apprentissage initial.
    #
    # max_iter=1 :
    # Nous contrôlons nous-mêmes les epochs
    # grâce à partial_fit().
    #
    # early_stopping=False :
    # Nous ne voulons pas de séparation interne
    # supplémentaire du dataset.

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=1,
        shuffle=True,
        early_stopping=False,
        random_state=RANDOM_STATE
    )

    return model


# ==========================================
# 5. ENTRAINEMENT DU RESEAU
# ==========================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    preprocessor
):

    # Créer un réseau non entraîné.
    network = create_model()

    # Mesurer le temps de préparation
    # des données d'entraînement.
    start_time = perf_counter()

    # Transformer les données avec le
    # préprocesseur déjà ajusté.
    X_train_scaled = preprocessor.transform(
        X_train
    ).astype(np.float32)

    training_time = perf_counter() - start_time

    # Transformer également la validation.
    # Aucun fit() n'est effectué sur celle-ci.
    X_val_scaled = preprocessor.transform(
        X_val
    ).astype(np.float32)

    # Liste destinée à conserver
    # les performances après chaque epoch.
    history = []

    # Mesurer aussi la durée totale,
    # incluant le suivi de la validation.
    total_start_time = perf_counter()

    print("\nStarting Neural Network training...")

    # Effectuer 30 passages sur les données.
    for epoch in range(1, EPOCHS + 1):

        epoch_start_time = perf_counter()

        # ==================================
        # VERITABLE APPRENTISSAGE DU RESEAU
        # ==================================
        #
        # partial_fit() effectue une epoch.
        #
        # Le réseau réalise :
        # 1. Forward propagation.
        # 2. Calcul de la fonction de perte.
        # 3. Backpropagation.
        # 4. Mise à jour des poids avec Adam.
        #
        # Les paramètres appris sont conservés
        # d'une epoch à l'autre.

        network.partial_fit(
            X_train_scaled,
            y_train,
            classes=np.array([0, 1])
        )

        # Ajouter uniquement le temps consacré
        # à l'entraînement du réseau.
        epoch_time = (
            perf_counter() - epoch_start_time
        )

        training_time += epoch_time

        # Calculer les scores sur l'entraînement
        # avec les poids obtenus à la fin de l'epoch.
        train_probabilities = network.predict_proba(
            X_train_scaled
        )[:, 1]

        # Calculer les scores sur la validation.
        val_probabilities = network.predict_proba(
            X_val_scaled
        )[:, 1]

        # Calculer la Log Loss des deux ensembles.
        #
        # Une perte faible signifie généralement
        # que les scores prédits correspondent
        # mieux aux véritables classes.

        train_loss = log_loss(
            y_train,
            train_probabilities,
            labels=[0, 1]
        )

        validation_loss = log_loss(
            y_val,
            val_probabilities,
            labels=[0, 1]
        )

        # Convertir les scores en classes.
        val_predictions = (
            val_probabilities >= THRESHOLD
        ).astype(int)

        # Calculer l'Average Precision.
        validation_ap = average_precision_score(
            y_val,
            val_probabilities
        )

        # Calculer le F1-score.
        validation_f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        # Enregistrer les résultats de l'epoch.
        history.append({
            "Epoch": epoch,
            "Training_Loss": train_loss,
            "Validation_Loss": validation_loss,
            "Validation_AP": validation_ap,
            "Validation_F1": validation_f1,
            "Epoch_Time": epoch_time
        })

        # Afficher la progression du réseau.
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {validation_loss:.6f} | "
            f"Val AP: {validation_ap:.4f} | "
            f"Val F1: {validation_f1:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = perf_counter() - total_start_time

    # ======================================
    # REGROUPER LE PREPROCESSING ET LE MLP
    # ======================================
    #
    # Les deux objets sont déjà ajustés.
    # Nous ne devons pas rappeler fit()
    # sur ce pipeline.

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", network)
    ])

    print("\n===== NETWORK INFORMATION =====")

    print(f"Training time: {training_time:.4f}s")

    print(
        f"Total time with monitoring: "
        f"{total_time:.4f}s"
    )

    print(f"Network layers: {network.n_layers_}")

    print(f"Output activation: {network.out_activation_}")

    # coefs_ contient les matrices de poids.
    # intercepts_ contient les vecteurs de biais.

    number_of_weights = sum(
        weights.size
        for weights in network.coefs_
    )

    number_of_biases = sum(
        biases.size
        for biases in network.intercepts_
    )

    number_of_parameters = (
        number_of_weights + number_of_biases
    )

    print(f"Weights: {number_of_weights}")
    print(f"Biases: {number_of_biases}")
    print(f"Total parameters: {number_of_parameters}")

    return (
        model,
        training_time,
        history,
        number_of_parameters
    )


# ==========================================
# 6. EVALUATION DU MODELE
# ==========================================

def evaluate_model(model, X_val, y_val, name):

    start_time = perf_counter()

    # Récupérer les scores de la classe fraude.
    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # Appliquer le seuil de classification.
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
    # que pour les autres algorithmes.

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

        "Epochs": EPOCHS
    }

    print("\n===== EVALUATION RESULTS =====")

    for metric, value in results.items():

        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")

    return results, matrix


# ==========================================
# 7. MATRICE DE CONFUSION
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

    plt.title(f"Neural Network - {name}")

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"confusion_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. SAUVEGARDE DU MODELE
# ==========================================

def save_model(model, name):

    model_path = MODELS_DIR / f"{name}.joblib"

    # Sauvegarder le pipeline complet :
    #
    # - Le préprocesseur supplémentaire.
    # - L'architecture du réseau.
    # - Les poids et les biais appris.

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    # Calculer la taille du fichier.
    model_size = (
        model_path.stat().st_size / (1024 ** 2)
    )

    print(f"\nModel saved: {model_path}")
    print(f"Model size: {model_size:.4f} MB")

    return model_size


# ==========================================
# 9. COURBES D'APPRENTISSAGE
# ==========================================

def save_learning_curves(history, name):

    # Transformer l'historique des epochs
    # en un tableau Pandas.
    learning_history = pd.DataFrame(history)

    # Sauvegarder toutes les valeurs.
    learning_history.to_csv(
        REPORTS_DIR / f"learning_history_{name}.csv",
        index=False
    )

    # --------------------------------------
    # GRAPHIQUE 1 : FONCTION DE PERTE
    # --------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        learning_history["Epoch"],
        learning_history["Training_Loss"],
        label="Training Loss"
    )

    plt.plot(
        learning_history["Epoch"],
        learning_history["Validation_Loss"],
        label="Validation Loss"
    )

    plt.title(
        f"Neural Network Learning Curves - {name}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"learning_curves_{name}.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------
    # GRAPHIQUE 2 : EVOLUTION DES SCORES
    # --------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        learning_history["Epoch"],
        learning_history["Validation_AP"],
        label="Average Precision"
    )

    plt.plot(
        learning_history["Epoch"],
        learning_history["Validation_F1"],
        label="F1-score"
    )

    plt.title(
        f"Neural Network Validation Scores - {name}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"validation_scores_{name}.png",
        dpi=300
    )

    plt.close()

    # Identifier l'epoch ayant obtenu la meilleure AP.
    #
    # Cette information est uniquement descriptive.
    # Nous conservons le modèle de la dernière epoch
    # pour notre première expérimentation.

    best_epoch = learning_history.loc[
        learning_history["Validation_AP"].idxmax()
    ]

    print("\n===== BEST OBSERVED EPOCH =====")

    print(f"Epoch: {int(best_epoch['Epoch'])}")

    print(
        f"Average Precision: "
        f"{best_epoch['Validation_AP']:.6f}"
    )

    print("Learning curves saved successfully!")


# ==========================================
# 10. ANALYSE DES PARAMETRES DU RESEAU
# ==========================================

def save_network_statistics(model, name):

    # Récupérer le réseau de neurones
    # contenu dans le pipeline.
    network = model.named_steps["classifier"]

    statistics = []

    # Parcourir les matrices de poids.
    for index, (weights, biases) in enumerate(
        zip(network.coefs_, network.intercepts_),
        start=1
    ):

        statistics.append({
            "Layer": index,
            "Input_Neurons": weights.shape[0],
            "Output_Neurons": weights.shape[1],
            "Weights": weights.size,
            "Biases": biases.size,
            "Parameters": weights.size + biases.size
        })

    network_stats = pd.DataFrame(statistics)

    print("\n===== NETWORK STATISTICS =====")

    print(network_stats.to_string(index=False))

    network_stats.to_csv(
        REPORTS_DIR / f"network_statistics_{name}.csv",
        index=False
    )

    # Regrouper tous les poids du réseau
    # dans un seul tableau NumPy.
    all_weights = np.concatenate([
        weights.ravel()
        for weights in network.coefs_
    ])

    # Afficher la distribution des poids.
    plt.figure(figsize=(10, 6))

    plt.hist(
        all_weights,
        bins=50,
        color="steelblue",
        edgecolor="white"
    )

    plt.title(
        f"Neural Network Weight Distribution - {name}"
    )

    plt.xlabel("Weight Value")
    plt.ylabel("Number of Weights")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / f"weight_distribution_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 11. COMPARAISON DES MODELES
# ==========================================

def compare_models(all_results):

    comparison = pd.DataFrame(all_results)

    # Classer selon notre métrique principale.
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

    comparison.to_csv(
        REPORTS_DIR / "comparison.csv",
        index=False
    )

    # --------------------------------------
    # GRAPHIQUE DES PERFORMANCES
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

    ax.set_title(
        "Neural Network - Sampling Comparison"
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

    # --------------------------------------
    # GRAPHIQUES TECHNIQUES
    # --------------------------------------

    technical_metrics = [
        (
            "Training_Time",
            "Neural Network - Training Time",
            "Time (seconds)",
            "training_time.png"
        ),
        (
            "Prediction_Time",
            "Neural Network - Prediction Time",
            "Time (seconds)",
            "prediction_time.png"
        ),
        (
            "Model_Size_MB",
            "Neural Network - Model Size",
            "Size (MB)",
            "model_size.png"
        )
    ]

    for metric, title, ylabel, filename in technical_metrics:

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

    best_model = comparison.iloc[0]

    print("\n===== BEST NEURAL NETWORK =====")

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
# 12. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)
    print("FRAUDDETECT AI - Neural Network Training")
    print("=" * 60)

    # Charger le jeu de validation.
    X_val, y_val = load_validation_data()

    # Ajuster la standardisation supplémentaire
    # uniquement sur l'entraînement original.
    X_original, _ = load_dataset()

    preprocessor = create_preprocessor(
        X_original
    )

    del X_original

    # Définir nos quatre expériences.
    experiments = [
        {
            "name": "undersampling",
            "suffix": "_undersampled"
        },
        {
            "name": "original",
            "suffix": ""
        },
        {
            "name": "oversampling",
            "suffix": "_oversampled"
        },
        {
            "name": "smote",
            "suffix": "_smote"
        }
    ]

    all_results = []

    # Entraîner les quatre réseaux indépendamment.
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

        # Etape 2 : entraîner le réseau.
        (
            model,
            training_time,
            history,
            number_of_parameters
        ) = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            preprocessor
        )

        # Etape 3 : évaluer le modèle final.
        results, matrix = evaluate_model(
            model,
            X_val,
            y_val,
            name
        )

        results["Training_Time"] = training_time

        results["Number_of_Parameters"] = (
            number_of_parameters
        )

        # Etape 4 : sauvegarder le modèle.
        model_size = save_model(model, name)

        results["Model_Size_MB"] = model_size

        # Etape 5 : générer la matrice.
        save_confusion_matrix(matrix, name)

        # Etape 6 : générer les courbes.
        save_learning_curves(history, name)

        # Etape 7 : analyser le réseau.
        save_network_statistics(model, name)

        # Etape 8 : conserver les résultats.
        all_results.append(results)

        # Libérer les objets devenus inutiles.
        del model, X_train, y_train

    # Comparer les quatre modèles.
    compare_models(all_results)

    print("\n" + "=" * 60)
    print("Neural Network training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
