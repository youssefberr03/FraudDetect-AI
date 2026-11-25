
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

# Récupérer le chemin de la racine du projet.
BASE_DIR = Path(__file__).resolve().parents[1]

# Dossier contenant les données prétraitées.
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Dossier contenant le modèle sélectionné.
THRESHOLD_DIR = (
    BASE_DIR / "reports" / "threshold_tuning"
)

# Dossier contenant les résultats finaux.
EVALUATION_DIR = (
    BASE_DIR / "reports" / "final_evaluation"
)

# Dossier de destination du modèle exporté.
EXPORT_DIR = BASE_DIR / "exports" / "frauddetect_v1"

# Créer automatiquement le dossier.
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Version de notre modèle.
MODEL_VERSION = "1.0.0"


# ==========================================
# 2. CHARGEMENT DES FICHIERS JSON
# ==========================================

def load_json(path):

    # Vérifier que le fichier existe.
    if not path.is_file():

        raise FileNotFoundError(path)

    # Lire son contenu et le convertir
    # en dictionnaire Python.
    with open(path, encoding="utf-8") as file:

        return json.load(file)


# ==========================================
# 3. CHARGEMENT DU MODELE FINAL
# ==========================================

def load_final_model():

    # Charger les métadonnées obtenues
    # pendant le Threshold Tuning.
    selection = load_json(
        THRESHOLD_DIR / "selected_model.json"
    )

    # Charger les résultats de l'évaluation finale.
    evaluation = load_json(
        EVALUATION_DIR
        / "final_evaluation_summary.json"
    )

    # Vérifier que les deux rapports concernent
    # le même modèle et le même seuil.
    if (
        selection["model_name"]
        != evaluation["selected_model"]
    ):

        raise ValueError("Model name mismatch.")

    if (
        selection["model_path"]
        != evaluation["model_path"]
    ):

        raise ValueError("Model path mismatch.")

    if (
        selection["threshold"]
        != evaluation["threshold"]
    ):

        raise ValueError("Threshold mismatch.")

    # Récupérer le chemin du modèle entraîné.
    model_path = (
        BASE_DIR / selection["model_path"]
    )

    if not model_path.is_file():

        raise FileNotFoundError(model_path)

    # Charger notre propre modèle sauvegardé.
    model = joblib.load(model_path)

    print("\n===== SELECTED MODEL =====")

    print(f"Model: {selection['model_name']}")

    print(f"Version: {MODEL_VERSION}")

    print(
        f"Threshold: "
        f"{selection['threshold']:.12f}"
    )

    print(f"Source: {model_path}")

    print(
        f"Number of features: "
        f"{model.n_features_in_}"
    )

    return model, selection, evaluation


# ==========================================
# 4. EXPORTATION DU MODELE
# ==========================================

def export_model(model):

    # Définir le nom définitif du fichier.
    output_path = EXPORT_DIR / "model.joblib"

    # Sérialiser le modèle avec compression.
    #
    # Le modèle conserve toutes les connaissances
    # obtenues durant son apprentissage.
    joblib.dump(
        model,
        output_path,
        compress=3
    )

    # Calculer la taille du fichier en Ko.
    size_kb = output_path.stat().st_size / 1024

    print("\n===== MODEL EXPORT =====")

    print(f"Model saved: {output_path}")

    print(f"Model size: {size_kb:.2f} KB")

    return output_path


# ==========================================
# 5. CALCUL DE L'EMPREINTE SHA-256
# ==========================================

def calculate_checksum(path):

    # SHA-256 permet de générer une empreinte
    # numérique représentant le contenu du fichier.

    sha256 = hashlib.sha256()

    # Ouvrir le modèle en mode binaire.
    with open(path, "rb") as file:

        # Lire progressivement le fichier
        # par blocs de 8192 octets.
        for block in iter(
            lambda: file.read(8192),
            b""
        ):

            sha256.update(block)

    # Retourner l'empreinte sous forme hexadécimale.
    return sha256.hexdigest()


# ==========================================
# 6. EXPORTATION DES METADONNEES
# ==========================================

def export_metadata(
    model,
    selection,
    evaluation,
    model_path
):

    metadata = {

        "project_name": "FraudDetect AI",

        "model_version": MODEL_VERSION,

        "model_name": selection["model_name"],

        "algorithm": "Random Forest",

        "model_file": model_path.name,

        "task": "Binary Classification",

        "training_strategy": "balanced",

        "selection_strategy": (
            selection["selection_strategy"]
        ),

        "threshold": selection["threshold"],

        "number_of_features": int(
            model.n_features_in_
        ),

        "feature_names": selection[
            "feature_names"
        ],

        # Le modèle attend des données déjà
        # prétraitées selon notre protocole.
        "input_format": "processed_features",

        "classes": {

            "0": "Legitimate",

            "1": "Fraud"
        },

        "model_parameters": model.get_params(
            deep=False
        ),

        "validation_metrics": selection[
            "validation_metrics"
        ],

        "test_metrics": evaluation[
            "test_final_metrics"
        ],

        "test_transactions": evaluation[
            "test_transactions"
        ],

        "test_frauds": evaluation[
            "test_frauds"
        ],

        "cost_assumptions": evaluation[
            "cost_assumptions"
        ],

        "environment": {

            "python": platform.python_version(),

            "scikit_learn": sklearn.__version__,

            "numpy": np.__version__,

            "pandas": pd.__version__,

            "joblib": joblib.__version__
        },

        "model_size_bytes": (
            model_path.stat().st_size
        ),

        "sha256": calculate_checksum(
            model_path
        ),

        "export_date": datetime.now(
            timezone.utc
        ).isoformat(),

        "model_retrained": False,

        "threshold_changed": False,

        "production_ready": False
    }

    # Sauvegarder les métadonnées au format JSON.
    output_path = EXPORT_DIR / "metadata.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n===== METADATA EXPORT =====")

    print(f"Metadata saved: {output_path}")

    return metadata


# ==========================================
# 7. VERIFICATION DU MODELE EXPORTE
# ==========================================

def verify_export(original_model, metadata):

    # Charger le modèle que nous venons d'exporter.
    exported_model = joblib.load(
        EXPORT_DIR / metadata["model_file"]
    )

    # Vérifier l'intégrité du fichier.
    current_checksum = calculate_checksum(
        EXPORT_DIR / metadata["model_file"]
    )

    if current_checksum != metadata["sha256"]:

        raise ValueError("Checksum verification failed.")

    # Utiliser des exemples du Training Set
    # uniquement pour vérifier la sérialisation.
    #
    # Nous ne réalisons aucune nouvelle évaluation
    # scientifique sur le Test Set.
    samples = pd.read_csv(
        PROCESSED_DIR / "X_train.csv",
        nrows=1000
    )

    # Respecter exactement l'ordre des variables.
    samples = samples[metadata["feature_names"]]

    # Identifier la colonne de la classe fraude.
    fraud_index = np.flatnonzero(
        original_model.classes_ == 1
    )[0]

    exported_fraud_index = np.flatnonzero(
        exported_model.classes_ == 1
    )[0]

    # Calculer les scores du modèle original.
    original_scores = original_model.predict_proba(
        samples
    )[:, fraud_index]

    # Calculer les scores du modèle exporté.
    exported_scores = exported_model.predict_proba(
        samples
    )[:, exported_fraud_index]

    # Vérifier que les deux modèles donnent
    # les mêmes scores à la précision numérique près.
    scores_identical = np.allclose(
        original_scores,
        exported_scores,
        rtol=0,
        atol=1e-12
    )

    threshold = metadata["threshold"]

    # Appliquer le seuil personnalisé.
    original_predictions = (
        original_scores >= threshold
    ).astype(int)

    exported_predictions = (
        exported_scores >= threshold
    ).astype(int)

    # Vérifier que les classifications sont identiques.
    predictions_identical = np.array_equal(
        original_predictions,
        exported_predictions
    )

    max_difference = float(
        np.max(
            np.abs(
                original_scores - exported_scores
            )
        )
    )

    verification = {

        "samples_tested": len(samples),

        "checksum_valid": True,

        "scores_identical": bool(
            scores_identical
        ),

        "predictions_identical": bool(
            predictions_identical
        ),

        "maximum_score_difference": max_difference,

        "verification_successful": bool(
            scores_identical
            and predictions_identical
        )
    }

    print("\n===== EXPORT VERIFICATION =====")

    for key, value in verification.items():

        print(f"{key}: {value}")

    # Interrompre le programme si l'exportation
    # a modifié le comportement du modèle.
    if not verification["verification_successful"]:

        raise RuntimeError(
            "Export verification failed."
        )

    # Enregistrer le rapport de vérification.
    with open(
        EXPORT_DIR / "verification.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            verification,
            file,
            indent=4
        )

    return verification


# ==========================================
# 8. EXPORTATION DU SCHEMA DES VARIABLES
# ==========================================

def export_feature_schema(metadata):

    # Construire une description des données
    # attendues par notre modèle.

    features = []

    for index, name in enumerate(
        metadata["feature_names"]
    ):

        if name == "Time":

            description = (
                "Elapsed time from the first "
                "transaction, after preprocessing."
            )

        elif name == "Amount":

            description = (
                "Transaction amount, "
                "after preprocessing."
            )

        elif name.startswith("V"):

            description = (
                "Anonymized PCA component. "
                "Original meaning unavailable."
            )

        else:

            description = "Processed feature."

        features.append({

            "position": index,

            "name": name,

            "type": "float",

            "description": description
        })

    schema = {

        "number_of_features": len(features),

        "input_format": "processed_features",

        "features": features
    }

    with open(
        EXPORT_DIR / "feature_schema.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            schema,
            file,
            indent=4
        )

    print("\n===== FEATURE SCHEMA =====")

    print(f"Features exported: {len(features)}")


# ==========================================
# 9. EXPORTATION DES VERSIONS DES LIBRAIRIES
# ==========================================

def export_dependencies(metadata):

    environment = metadata["environment"]

    dependencies = [

        f"scikit-learn=={environment['scikit_learn']}",

        f"numpy=={environment['numpy']}",

        f"pandas=={environment['pandas']}",

        f"joblib=={environment['joblib']}"
    ]

    output_path = EXPORT_DIR / "requirements.txt"

    output_path.write_text(
        "\n".join(dependencies) + "\n",
        encoding="utf-8"
    )

    print("\n===== MODEL DEPENDENCIES =====")

    for dependency in dependencies:

        print(dependency)


# ==========================================
# 10. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 65)

    print("FRAUDDETECT AI - Final Model Export")

    print("=" * 65)

    # 1. Charger le modèle et ses résultats.
    model, selection, evaluation = (
        load_final_model()
    )

    # 2. Exporter le modèle entraîné.
    model_path = export_model(model)

    # 3. Exporter sa configuration.
    metadata = export_metadata(
        model,
        selection,
        evaluation,
        model_path
    )

    # 4. Vérifier que l'exportation est correcte.
    verify_export(model, metadata)

    # 5. Décrire les variables attendues.
    export_feature_schema(metadata)

    # 6. Sauvegarder les versions des dépendances.
    export_dependencies(metadata)

    # 7. Afficher le bilan final.
    print("\n" + "=" * 65)

    print("FINAL EXPORT SUMMARY")

    print("=" * 65)

    print(f"Model: {metadata['model_name']}")

    print(f"Version: {MODEL_VERSION}")

    print(
        f"Threshold: "
        f"{metadata['threshold']:.12f}"
    )

    print(
        f"Test AP: "
        f"{metadata['test_metrics']['Average_Precision']:.6f}"
    )

    print(
        f"Test Recall: "
        f"{metadata['test_metrics']['Recall']:.6f}"
    )

    print(
        f"Test F2: "
        f"{metadata['test_metrics']['F2']:.6f}"
    )

    print(f"Export directory: {EXPORT_DIR}")

    print("\nModel exported successfully!")

    print("Export verification passed.")

    print("No model retraining performed.")

    print("=" * 65)


if __name__ == "__main__":

    main()
