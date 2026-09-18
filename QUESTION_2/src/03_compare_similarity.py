from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
NOTICES_DIR = DATA_DIR / "notices"
LABELS_FILE = DATA_DIR / "labelled_pairs.csv"


# ---------------------------------------------------------
# Load corpus
# ---------------------------------------------------------
print("Loading notices...")

parts = []

for file in sorted(NOTICES_DIR.glob("*.csv")):
    parts.append(pd.read_csv(file))

notices = pd.concat(parts, ignore_index=True)

notice_lookup = notices.set_index("notice_id")


# ---------------------------------------------------------
# Load labelled pairs
# ---------------------------------------------------------
labels = pd.read_csv(LABELS_FILE)

same_pair = labels[labels["label"] == "same"].iloc[0]
different_pair = labels[labels["label"] == "different"].iloc[0]


def get_text(notice_id):
    row = notice_lookup.loc[notice_id]

    # Title + body
    return str(row["title"]) + "\n" + str(row["body"])


same_a = get_text(same_pair["notice_id_a"])
same_b = get_text(same_pair["notice_id_b"])

diff_a = get_text(different_pair["notice_id_a"])
diff_b = get_text(different_pair["notice_id_b"])


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def pair_similarity(vectorizer, text_a, text_b):
    matrix = vectorizer.fit_transform([text_a, text_b])
    return cosine_similarity(matrix[0], matrix[1])[0, 0]


# ---------------------------------------------------------
# Choice 1: WORD TF-IDF
# ---------------------------------------------------------
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

word_same = pair_similarity(word_vectorizer, same_a, same_b)
word_diff = pair_similarity(word_vectorizer, diff_a, diff_b)


# ---------------------------------------------------------
# Choice 2: CHARACTER TF-IDF
# ---------------------------------------------------------
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(3, 5),
    min_df=1
)

char_same = pair_similarity(char_vectorizer, same_a, same_b)
char_diff = pair_similarity(char_vectorizer, diff_a, diff_b)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION A - COMPETING SIMILARITY REPRESENTATIONS")
print("=" * 70)

print("\nSelected SAME pair:")
print(
    same_pair["notice_id_a"],
    "<->",
    same_pair["notice_id_b"]
)

print("Selected DIFFERENT pair:")
print(
    different_pair["notice_id_a"],
    "<->",
    different_pair["notice_id_b"]
)

print("\nSimilarity scores")
print("-" * 70)

print(f"{'Representation':<25} {'SAME':>12} {'DIFFERENT':>12}")
print("-" * 70)

print(f"{'Word TF-IDF (1-2)':<25} {word_same:>12.4f} {word_diff:>12.4f}")
print(f"{'Character TF-IDF (3-5)':<25} {char_same:>12.4f} {char_diff:>12.4f}")

print("\nSeparation (SAME - DIFFERENT)")
print("-" * 70)

print(f"Word TF-IDF:       {word_same - word_diff:.4f}")
print(f"Character TF-IDF:  {char_same - char_diff:.4f}")

print("\n" + "=" * 70)