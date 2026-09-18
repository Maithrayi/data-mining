from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
NOTICES_DIR = DATA_DIR / "notices"
LABELS_FILE = DATA_DIR / "labelled_pairs.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

parts = []

for file in sorted(NOTICES_DIR.glob("*.csv")):
    parts.append(pd.read_csv(file))

notices = pd.concat(parts, ignore_index=True)
notice_lookup = notices.set_index("notice_id")
labels = pd.read_csv(LABELS_FILE)


def normalize(text):
    return " ".join(str(text).lower().split())


def shingles(text, k=3):
    words = normalize(text).split()

    if len(words) < k:
        return {" ".join(words)}

    return {
        " ".join(words[i:i+k])
        for i in range(len(words) - k + 1)
    }


def get_text(notice_id):
    row = notice_lookup.loc[notice_id]
    return str(row["title"]) + " " + str(row["body"])


def hash_value(shingle, seed):
    data = f"{seed}:{shingle}".encode("utf-8")
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def minhash_signature(shingle_set, num_hashes=128):
    return np.array([
        min(hash_value(s, seed) for s in shingle_set)
        for seed in range(num_hashes)
    ], dtype=np.uint64)


def exact_jaccard(a, b):
    return len(a & b) / len(a | b)


# ---------------------------------------------------------
# Create signatures only for labelled notices
# ---------------------------------------------------------

print("=" * 70)
print("SECTION C - LSH CANDIDATE RETRIEVAL")
print("=" * 70)

notice_ids = pd.unique(
    pd.concat([labels["notice_id_a"], labels["notice_id_b"]])
)

print(f"\nUnique labelled notices: {len(notice_ids)}")

signatures = {}
shingle_sets = {}

for i, notice_id in enumerate(notice_ids):

    s = shingles(get_text(notice_id))

    shingle_sets[notice_id] = s
    signatures[notice_id] = minhash_signature(s)

print("128-hash signatures created.")


# ---------------------------------------------------------
# LSH configurations
# ---------------------------------------------------------

configs = [
    (8, 16),
    (16, 8),
    (32, 4),
]


for bands, rows in configs:

    assert bands * rows == 128

    buckets = {}

    for notice_id, signature in signatures.items():

        for band in range(bands):

            start = band * rows
            end = start + rows

            chunk = signature[start:end]

            key = (
                band,
                hashlib.blake2b(
                    chunk.tobytes(),
                    digest_size=8
                ).hexdigest()
            )

            buckets.setdefault(key, set()).add(notice_id)


    candidate_counts = []
    survived = []
    true_pairs = 0

    for _, pair in labels.iterrows():

        a = pair["notice_id_a"]
        b = pair["notice_id_b"]

        candidate_set = set()

        for band in range(bands):

            start = band * rows
            end = start + rows

            chunk = signatures[a][start:end]

            key = (
                band,
                hashlib.blake2b(
                    chunk.tobytes(),
                    digest_size=8
                ).hexdigest()
            )

            candidate_set.update(buckets.get(key, set()))

        candidate_set.discard(a)

        candidate_counts.append(len(candidate_set))

        if b in candidate_set:
            survived.append(1)
        else:
            survived.append(0)

    results = labels.copy()
    results["survived"] = survived

    same_survival = results.loc[
        results["label"] == "same",
        "survived"
    ].mean()

    different_survival = results.loc[
        results["label"] == "different",
        "survived"
    ].mean()

    print("\n" + "-" * 70)
    print(f"LSH configuration: {bands} bands × {rows} rows")
    print(f"Average candidates/query: {np.mean(candidate_counts):.1f}")
    print(f"Median candidates/query: {np.median(candidate_counts):.1f}")
    print(f"SAME pair survival: {same_survival * 100:.1f}%")
    print(
        f"DIFFERENT pair survival: "
        f"{different_survival * 100:.1f}%"
    )


# ---------------------------------------------------------
# Plot operating curve
# ---------------------------------------------------------

similarities = []
survival = []

for _, pair in labels.iterrows():

    a = pair["notice_id_a"]
    b = pair["notice_id_b"]

    sim = exact_jaccard(
        shingle_sets[a],
        shingle_sets[b]
    )

    similarities.append(sim)
    survival.append(
        1 if b in set() else 0
    )


# Use selected operating configuration: 32 × 4

bands = 32
rows = 4

buckets = {}

for notice_id, signature in signatures.items():

    for band in range(bands):

        start = band * rows
        end = start + rows

        chunk = signature[start:end]

        key = (
            band,
            hashlib.blake2b(
                chunk.tobytes(),
                digest_size=8
            ).hexdigest()
        )

        buckets.setdefault(key, set()).add(notice_id)


survived = []

for _, pair in labels.iterrows():

    a = pair["notice_id_a"]
    b = pair["notice_id_b"]

    candidate_set = set()

    for band in range(bands):

        start = band * rows
        end = start + rows

        chunk = signatures[a][start:end]

        key = (
            band,
            hashlib.blake2b(
                chunk.tobytes(),
                digest_size=8
            ).hexdigest()
        )

        candidate_set.update(buckets.get(key, set()))

    candidate_set.discard(a)

    survived.append(1 if b in candidate_set else 0)


df = pd.DataFrame({
    "similarity": similarities,
    "survived": survived
})

bins = np.arange(0, 1.01, 0.05)
df["bin"] = pd.cut(
    df["similarity"],
    bins=bins,
    include_lowest=True
)

curve = df.groupby(
    "bin",
    observed=True
)["survived"].mean()

x = [
    interval.mid
    for interval in curve.index
]

y = curve.values

plt.figure(figsize=(9, 5))
plt.plot(x, y, marker="o")
plt.xlabel("Exact Jaccard similarity")
plt.ylabel("Probability pair survives candidate stage")
plt.title("LSH Candidate Survival vs True Similarity")
plt.grid(True)
plt.tight_layout()

plot_path = DATA_DIR.parent / "DATA_MINING_LAB" / "QUESTION_2" / "lsh_survival_curve.png"
plt.savefig(plot_path, dpi=150)

print("\nPlot saved to:")
print(plot_path)

print("\n" + "=" * 70)
print("SECTION C COMPLETE")
print("=" * 70)