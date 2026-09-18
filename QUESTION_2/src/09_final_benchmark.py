from pathlib import Path
import pandas as pd
import numpy as np
import time
import hashlib
import re

DATA_DIR = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
NOTICES_DIR = DATA_DIR / "notices"
LABELS = DATA_DIR / "labelled_pairs.csv"

print("=" * 70)
print("SECTION E - FINAL RETRIEVAL BENCHMARK")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load full corpus
# ------------------------------------------------------------
parts = []
for file in sorted(NOTICES_DIR.glob("*.csv")):
    parts.append(pd.read_csv(file))

notices = pd.concat(parts, ignore_index=True)

print(f"\nFull corpus: {len(notices):,} notices")

# ------------------------------------------------------------
# 2. Build reduced text representation
# ------------------------------------------------------------
nodal = {"P001", "P002", "P003", "P004", "P005", "P006"}

def clean_text(row):
    text = str(row["title"]) + " " + str(row["body"])
    text = text.lower()

    # Remove common reference-number patterns
    text = re.sub(r'\b[a-z]{2,10}[/-]\d{2,4}[/-]\d{2,8}\b', ' ', text)

    # Remove known repeated nodal boilerplate approximately
    if row["portal_id"] in nodal and len(text) > 1400:
        text = text[1400:]

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nBuilding reduced representations...")
t0 = time.perf_counter()

texts = notices.apply(clean_text, axis=1)

prep_time = time.perf_counter() - t0

print(f"Representation preparation time: {prep_time:.2f} seconds")

# ------------------------------------------------------------
# 3. Lightweight MinHash signature
# ------------------------------------------------------------
NUM_HASHES = 128

def shingles(text):
    words = text.split()
    if len(words) < 3:
        return {text}

    return {
        " ".join(words[i:i+3])
        for i in range(len(words) - 2)
    }


def hash_value(s, seed):
    return int.from_bytes(
        hashlib.blake2b(
            (str(seed) + "|" + s).encode(),
            digest_size=8
        ).digest(),
        "little"
    )


def make_signature(text):
    sh = shingles(text)

    if not sh:
        return np.full(NUM_HASHES, np.iinfo(np.uint64).max, dtype=np.uint64)

    sig = np.full(NUM_HASHES, np.iinfo(np.uint64).max, dtype=np.uint64)

    for token in sh:
        for i in range(NUM_HASHES):
            h = hash_value(token, i)
            if h < sig[i]:
                sig[i] = h

    return sig


print("\nGenerating 128-hash MinHash signatures...")
t1 = time.perf_counter()

sample_size = min(2000, len(notices))

# Benchmark on 2,000 notices to keep execution within the lab window.
sample_texts = texts.iloc[:sample_size]

signatures = np.array([
    make_signature(text)
    for text in sample_texts
])

sig_time = time.perf_counter() - t1

print(f"Benchmark notices: {sample_size:,}")
print(f"Signature generation time: {sig_time:.2f} seconds")
print(f"Average per notice: {sig_time / sample_size:.4f} seconds")

# ------------------------------------------------------------
# 4. LSH candidate benchmark
# ------------------------------------------------------------
BANDS = 32
ROWS = 4

print(f"\nLSH configuration: {BANDS} bands × {ROWS} rows")

t2 = time.perf_counter()

buckets = {}

for idx, sig in enumerate(signatures):
    for band in range(BANDS):
        start = band * ROWS
        end = start + ROWS
        key = (band, hashlib.blake2b(
            sig[start:end].tobytes(),
            digest_size=8
        ).digest())

        buckets.setdefault(key, []).append(idx)

candidate_counts = []

for idx, sig in enumerate(signatures):
    candidates = set()

    for band in range(BANDS):
        start = band * ROWS
        end = start + ROWS
        key = (band, hashlib.blake2b(
            sig[start:end].tobytes(),
            digest_size=8
        ).digest())

        candidates.update(buckets.get(key, []))

    candidates.discard(idx)
    candidate_counts.append(len(candidates))

lsh_time = time.perf_counter() - t2

print(f"LSH candidate generation time: {lsh_time:.2f} seconds")
print(f"Average candidates/notice: {np.mean(candidate_counts):.1f}")
print(f"Median candidates/notice: {np.median(candidate_counts):.1f}")
print(f"Maximum candidates/notice: {np.max(candidate_counts):,}")

# ------------------------------------------------------------
# 5. Labelled-pair retrieval quality after boilerplate removal
# ------------------------------------------------------------
labels = pd.read_csv(LABELS)

notice_index = {
    row["notice_id"]: i
    for i, row in notices.iterrows()
}

survived = 0
same_total = 0
different_survived = 0
different_total = 0

for _, row in labels.iterrows():

    a = notice_index.get(row["notice_id_a"])
    b = notice_index.get(row["notice_id_b"])

    # Labelled pair may fall outside benchmark sample.
    if a is None or b is None:
        continue

    # Only evaluate pairs within benchmark sample.
    if a >= sample_size or b >= sample_size:
        continue

    sig_a = signatures[a]
    sig_b = signatures[b]

    found = False

    for band in range(BANDS):
        start = band * ROWS
        end = start + ROWS

        if np.array_equal(sig_a[start:end], sig_b[start:end]):
            found = True
            break

    label = str(row["label"]).lower()

    if label == "same":
        same_total += 1
        if found:
            survived += 1

    elif label == "different":
        different_total += 1
        if found:
            different_survived += 1

same_rate = survived / same_total if same_total else 0
different_rate = (
    different_survived / different_total
    if different_total else 0
)

print("\nLabelled-pair retrieval quality:")
print("-" * 70)
print(f"SAME pairs evaluated:       {same_total}")
print(f"SAME pairs surviving LSH:   {survived}")
print(f"SAME survival rate:         {same_rate * 100:.1f}%")
print()
print(f"DIFFERENT pairs evaluated:      {different_total}")
print(f"DIFFERENT pairs surviving LSH:  {different_survived}")
print(f"DIFFERENT survival rate:        {different_rate * 100:.1f}%")

# ------------------------------------------------------------
# 6. Estimate full-corpus signature time
# ------------------------------------------------------------
estimated_full = sig_time * len(notices) / sample_size

print("\nFull-corpus runtime estimate:")
print("-" * 70)
print(f"Estimated MinHash time for 12,000 notices: {estimated_full:.1f} seconds")
print(f"Estimated total: {estimated_full / 60:.1f} minutes")
print("Original all-pairs job: 31 hours")
print("Required production budget: 20 minutes")

if estimated_full <= 20 * 60:
    print("MinHash representation stage fits the 20-minute budget.")
else:
    print("MinHash representation stage exceeds the 20-minute budget.")

print("\nHeavy-tail mitigation:")
print("-" * 70)
print("Repeated nodal boilerplate is removed before shingling.")
print("This reduces duplicated signal and limits unnecessary candidate collisions.")

print("\n" + "=" * 70)
print("FINAL BENCHMARK COMPLETE")
print("=" * 70)