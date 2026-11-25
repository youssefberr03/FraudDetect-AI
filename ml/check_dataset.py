from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"


def main():
    print("=" * 50)
    print("FRAUDDETECT AI - Dataset verification")
    print("=" * 50)

    if not DATASET_PATH.exists():
        print("ERROR: Dataset not found.")
        print(f"Expected path: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset loaded successfully!")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nFirst 5 transactions:")
    print(df.head())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().sum())


if __name__ == "__main__":
    main()