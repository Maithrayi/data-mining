from pathlib import Path
import pandas as pd


DATA_DIR = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
NOTICES_DIR = DATA_DIR / "notices"
LABELS_FILE = DATA_DIR / "labelled_pairs.csv"


print("=" * 70)
print("QUESTION 2 - CORPUS INSPECTION")
print("=" * 70)


# ---------------------------------------------------------
# 1. Labelled pairs
# ---------------------------------------------------------
labels = pd.read_csv(LABELS_FILE)

print("\n[1] LABELLED PAIRS")
print("-" * 70)
print(f"Rows: {len(labels):,}")
print(f"Columns: {list(labels.columns)}")

print("\nLabel counts:")
print(labels["label"].value_counts())

print("\nLabel proportions:")
print(
    labels["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .astype(str)
    + "%"
)


# ---------------------------------------------------------
# 2. Notice files
# ---------------------------------------------------------
csv_files = sorted(NOTICES_DIR.glob("*.csv"))

print("\n[2] NOTICE FILES")
print("-" * 70)
print(f"CSV files: {len(csv_files)}")

total_rows = 0

for file in csv_files:
    df_part = pd.read_csv(file)
    rows = len(df_part)
    total_rows += rows

    print(f"{file.name:<20} {rows:>8,} rows")


# ---------------------------------------------------------
# 3. Total corpus size
# ---------------------------------------------------------
print("\n[3] CORPUS SIZE")
print("-" * 70)
print(f"Total notices: {total_rows:,}")


# ---------------------------------------------------------
# 4. Inspect first notice file
# ---------------------------------------------------------
if csv_files:
    first_file = csv_files[0]
    df = pd.read_csv(first_file)

    print("\n[4] NOTICE SCHEMA")
    print("-" * 70)

    print("Columns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 3 notices:")
    print(df.head(3).to_string(index=False))


    # -----------------------------------------------------
    # 5. Basic data quality
    # -----------------------------------------------------
    print("\n[5] BASIC DATA QUALITY")
    print("-" * 70)

    print("\nMissing values in first file:")
    print(df.isna().sum())

    print("\nUnique portal IDs in first file:")
    print(df["portal_id"].nunique())

    print("\nUnique notice IDs in first file:")
    print(df["notice_id"].nunique())


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)