from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


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


# ---------------------------------------------------------
# Text representation
# ---------------------------------------------------------

def normalize(text):
    text = str(text).lower()
    return " ".join(text.split())


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


# ---------------------------------------------------------
# Deterministic MinHash
# ---------------------------------------------------------

def hash_value(shingle, seed):
    data = f"{seed}:{shingle}".encode("utf-8")
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def minhash_signature(shingle_set, num_hashes):
    signature = []

    for seed in range(num_hashes):
        minimum = min(
            hash_value(shingle, seed)
            for shingle in shingle_set
        )
        signature.append(minimum)

    return np.array(signature, dtype=np.uint64)


def jaccard(a, b):
    if not a and not b:
        return 1.0

    return len(a & b) / len(a | b)


def estimate_jaccard(sig_a, sig_b):
    return np.mean(sig_a == sig_b)


# ---------------------------------------------------------
# Evaluate different signature sizes
# ---------------------------------------------------------

print("=" * 70)
print("SECTION B - MINHASH COMPRESSION ACCURACY")
print("=" * 70)

print("\nUsing word 3-shingles.")

signature_sizes = [32, 64, 128]

for num_hashes in signature_sizes:

    errors = []
    abs_errors = []

    for _, pair in labels.iterrows():

        text_a = get_text(pair["notice_id_a"])
        text_b = get_text(pair["notice_id_b"])

        set_a = shingles(text_a)
        set_b = shingles(text_b)

        exact = jaccard(set_a, set_b)

        sig_a = minhash_signature(set_a, num_hashes)
        sig_b = minhash_signature(set_b, num_hashes)

        estimated = estimate_jaccard(sig_a, sig_b)

        error = estimated - exact

        errors.append(error)
        abs_errors.append(abs(error))

    print(f"\nSignature size: {num_hashes} hashes")
    print(f"Storage per signature: {num_hashes * 8} bytes")
    print(f"Mean absolute error: {np.mean(abs_errors):.4f}")
    print(f"Median absolute error: {np.median(abs_errors):.4f}")
    print(f"Maximum absolute error: {np.max(abs_errors):.4f}")
    print(
        f"Within ±0.10: "
        f"{np.mean(np.array(abs_errors) <= 0.10) * 100:.1f}%"
    )

print("\n" + "=" * 70)
print("Recommendation: use 128 hashes unless the measurements show")
print("that a smaller signature gives essentially the same accuracy.")
print("=" * 70)