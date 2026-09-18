from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_DIR = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
NOTICES_DIR = DATA_DIR / "notices"
LABELS_FILE = DATA_DIR / "labelled_pairs.csv"


print("Loading notices...")

parts = []

for file in sorted(NOTICES_DIR.glob("*.csv")):
    parts.append(pd.read_csv(file))

notices = pd.concat(parts, ignore_index=True)
notice_lookup = notices.set_index("notice_id")

labels = pd.read_csv(LABELS_FILE)


def get_text(notice_id):
    row = notice_lookup.loc[notice_id]
    return str(row["title"]) + "\n" + str(row["body"])


texts = {}

for notice_id in pd.concat(
    [labels["notice_id_a"], labels["notice_id_b"]]
).unique():
    texts[notice_id] = get_text(notice_id)


def evaluate_representation(name, vectorizer):
    print(f"\nEvaluating: {name}")

    pair_texts = []

    for _, row in labels.iterrows():
        pair_texts.append(texts[row["notice_id_a"]])
        pair_texts.append(texts[row["notice_id_b"]])

    matrix = vectorizer.fit_transform(pair_texts)

    similarities = []

    for i in range(0, len(pair_texts), 2):
        a = matrix[i]
        b = matrix[i + 1]

        numerator = a.multiply(b).sum()
        denominator = np.sqrt(a.multiply(a).sum()) * np.sqrt(
            b.multiply(b).sum()
        )

        if denominator == 0:
            similarity = 0.0
        else:
            similarity = float(numerator / denominator)

        similarities.append(similarity)

    results = labels.copy()
    results["similarity"] = similarities
    results["is_same"] = results["label"].eq("same")

    same_scores = results.loc[
        results["label"] == "same", "similarity"
    ]

    different_scores = results.loc[
        results["label"] == "different", "similarity"
    ]

    print(f"Pairs evaluated: {len(results)}")
    print(f"SAME pairs: {len(same_scores)}")
    print(f"DIFFERENT pairs: {len(different_scores)}")

    print("\nScore distribution")
    print("-" * 70)

    print("SAME")
    print(
        f"  min    = {same_scores.min():.4f}\n"
        f"  median = {same_scores.median():.4f}\n"
        f"  mean   = {same_scores.mean():.4f}\n"
        f"  max    = {same_scores.max():.4f}"
    )

    print("\nDIFFERENT")
    print(
        f"  min    = {different_scores.min():.4f}\n"
        f"  median = {different_scores.median():.4f}\n"
        f"  mean   = {different_scores.mean():.4f}\n"
        f"  max    = {different_scores.max():.4f}"
    )

    print("\nThreshold sweep")
    print("-" * 70)

    best_threshold = None
    best_error = float("inf")

    for threshold in np.arange(0.10, 0.96, 0.05):

        predicted_same = results["similarity"] >= threshold

        false_merge = (
            (predicted_same)
            & (results["label"] == "different")
        ).sum()

        false_split = (
            (~predicted_same)
            & (results["label"] == "same")
        ).sum()

        total_error = false_merge + false_split

        if total_error < best_error:
            best_error = total_error
            best_threshold = threshold

        print(
            f"threshold={threshold:.2f}  "
            f"false_merge={false_merge:3d}  "
            f"false_split={false_split:3d}  "
            f"total={total_error:3d}"
        )

    print(
        f"\nMinimum unweighted error threshold: "
        f"{best_threshold:.2f}"
    )

    return results


word_results = evaluate_representation(
    "Word TF-IDF (1-2)",
    TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )
)

char_results = evaluate_representation(
    "Character TF-IDF (3-5)",
    TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=1
    )
)


print("\n" + "=" * 70)
print("SECTION A - CORPUS-LEVEL COMPARISON")
print("=" * 70)
print("All 900 manually labelled pairs were evaluated.")
print("Only labelled_pairs.csv was used for evaluation.")
print("=" * 70)