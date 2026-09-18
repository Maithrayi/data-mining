from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import re
import time
import sqlite3

BASE = Path(r"C:\Users\ub02-glab-058\Desktop\DATA_MINING_LAB\QUESTION_2")
DATA = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
NOTICES = DATA / "notices"
LABELS = DATA / "labelled_pairs.csv"
DB = BASE / "setubid_retrieval.db"

NUM_HASHES = 128
BANDS = 32
ROWS = 4

print("=" * 78)
print("ACTUAL 12,000-NOTICE END-TO-END BENCHMARK")
print("=" * 78)

# ============================================================
# LOAD CORPUS
# ============================================================

parts = [pd.read_csv(f) for f in sorted(NOTICES.glob("*.csv"))]
df = pd.concat(parts, ignore_index=True)

print(f"\nCorpus: {len(df):,} notices")
print(f"Portals: {df.portal_id.nunique():,}")

# ============================================================
# TEXT PREPARATION
# ============================================================

nodal = {"P001", "P002", "P003", "P004", "P005", "P006"}

def raw_text(row):
    return (
        str(row["title"]) + " " +
        str(row["body"])
    ).lower()

def clean_text(row):
    text = raw_text(row)

    # Remove repeated nodal preamble approximately.
    if row["portal_id"] in nodal and len(text) > 1400:
        text = text[1400:]

    # Remove common reference-number patterns.
    text = re.sub(
        r'\b[a-z]{2,12}[/-]\d{2,6}[/-]\d{2,10}\b',
        ' ',
        text
    )

    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("\nPreparing raw and mitigated text...")
t0 = time.perf_counter()

raw = [raw_text(r) for _, r in df.iterrows()]
clean = [clean_text(r) for _, r in df.iterrows()]

prep_time = time.perf_counter() - t0

print(f"Preparation time: {prep_time:.2f} sec")

# ============================================================
# 3-WORD SHINGLES
# ============================================================

def shingles(text):
    words = text.split()

    if len(words) < 3:
        return [text]

    return [
        " ".join(words[i:i+3])
        for i in range(len(words)-2)
    ]

# ============================================================
# FAST 64-BIT TOKEN HASH
# ============================================================

def token_hash(token):
    return int.from_bytes(
        hashlib.blake2b(
            token.encode(),
            digest_size=8
        ).digest(),
        "little"
    )

# Deterministic hash coefficients.
rng = np.random.default_rng(20260918)

A = rng.integers(
    1,
    np.iinfo(np.uint64).max,
    size=NUM_HASHES,
    dtype=np.uint64
)

B = rng.integers(
    0,
    np.iinfo(np.uint64).max,
    size=NUM_HASHES,
    dtype=np.uint64
)

MAX_UINT = np.iinfo(np.uint64).max


def build_signatures(texts, label):

    print(f"\nBuilding {label} MinHash signatures...")

    t_start = time.perf_counter()

    # Cache shingle hashes globally.
    cache = {}

    signatures = np.empty(
        (len(texts), NUM_HASHES),
        dtype=np.uint64
    )

    for idx, text in enumerate(texts):

        sh = shingles(text)

        values = []

        for s in sh:
            if s not in cache:
                cache[s] = token_hash(s)
            values.append(cache[s])

        x = np.asarray(values, dtype=np.uint64)

        best = np.full(NUM_HASHES, MAX_UINT, dtype=np.uint64)

        # Process hash functions in chunks to keep memory small.
        for start in range(0, NUM_HASHES, 16):
            aa = A[start:start+16, None]
            bb = B[start:start+16, None]

            vals = (aa * x[None, :] + bb)

            best[start:start+16] = vals.min(axis=1)

        signatures[idx] = best

        if (idx + 1) % 2000 == 0:
            print(f"  {idx+1:,}/{len(texts):,}")

    elapsed = time.perf_counter() - t_start

    print(f"{label} signature time: {elapsed:.2f} sec")
    print(f"Average: {elapsed / len(texts):.4f} sec/notice")
    print(f"Unique shingles cached: {len(cache):,}")

    return signatures, elapsed


# ============================================================
# BUILD ACTUAL FULL-CORPUS SIGNATURES
# ============================================================

raw_sig, raw_time = build_signatures(
    raw,
    "RAW"
)

clean_sig, clean_time = build_signatures(
    clean,
    "MITIGATED"
)

# ============================================================
# LSH CANDIDATES
# ============================================================

def lsh_candidates(signatures):

    buckets = {}

    for i, sig in enumerate(signatures):

        for band in range(BANDS):

            s = band * ROWS
            e = s + ROWS

            key = (
                band,
                hashlib.blake2b(
                    sig[s:e].tobytes(),
                    digest_size=8
                ).digest()
            )

            buckets.setdefault(key, []).append(i)

    candidate_counts = np.zeros(
        len(signatures),
        dtype=np.int32
    )

    for i, sig in enumerate(signatures):

        candidates = set()

        for band in range(BANDS):

            s = band * ROWS
            e = s + ROWS

            key = (
                band,
                hashlib.blake2b(
                    sig[s:e].tobytes(),
                    digest_size=8
                ).digest()
            )

            candidates.update(buckets.get(key, []))

        candidates.discard(i)

        candidate_counts[i] = len(candidates)

    return candidate_counts, buckets


print("\nRunning RAW 32×4 LSH...")
t2 = time.perf_counter()

raw_counts, raw_buckets = lsh_candidates(raw_sig)

raw_lsh_time = time.perf_counter() - t2

print(f"RAW LSH time: {raw_lsh_time:.2f} sec")
print(f"RAW average candidates: {raw_counts.mean():.1f}")
print(f"RAW median candidates: {np.median(raw_counts):.1f}")
print(f"RAW maximum candidates: {raw_counts.max():,}")


print("\nRunning MITIGATED 32×4 LSH...")
t3 = time.perf_counter()

clean_counts, clean_buckets = lsh_candidates(clean_sig)

clean_lsh_time = time.perf_counter() - t3

print(f"MITIGATED LSH time: {clean_lsh_time:.2f} sec")
print(f"MITIGATED average candidates: {clean_counts.mean():.1f}")
print(f"MITIGATED median candidates: {np.median(clean_counts):.1f}")
print(f"MITIGATED maximum candidates: {clean_counts.max():,}")

# ============================================================
# LABELLED-PAIR SURVIVAL - ALL 900
# ============================================================

labels = pd.read_csv(LABELS)

index = {
    nid: i
    for i, nid in enumerate(df.notice_id)
}

def survives(sig_a, sig_b):

    for band in range(BANDS):

        s = band * ROWS
        e = s + ROWS

        if np.array_equal(
            sig_a[s:e],
            sig_b[s:e]
        ):
            return True

    return False


def evaluate(signatures):

    same_total = 0
    same_survive = 0

    diff_total = 0
    diff_survive = 0

    for _, r in labels.iterrows():

        a = index.get(r["notice_id_a"])
        b = index.get(r["notice_id_b"])

        if a is None or b is None:
            continue

        result = survives(
            signatures[a],
            signatures[b]
        )

        if r["label"] == "same":

            same_total += 1

            if result:
                same_survive += 1

        else:

            diff_total += 1

            if result:
                diff_survive += 1

    return (
        same_total,
        same_survive,
        diff_total,
        diff_survive
    )


print("\nEvaluating ALL 900 labelled pairs...")

raw_eval = evaluate(raw_sig)
clean_eval = evaluate(clean_sig)

def print_eval(name, result):

    st, ss, dt, ds = result

    print(f"\n{name}")
    print("-" * 50)
    print(f"SAME evaluated:       {st}")
    print(f"SAME survived:        {ss}")
    print(
        f"SAME survival:        "
        f"{ss/st*100:.1f}%"
    )
    print(f"DIFFERENT evaluated:  {dt}")
    print(f"DIFFERENT survived:   {ds}")
    print(
        f"DIFFERENT survival:   "
        f"{ds/dt*100:.1f}%"
    )


print_eval("RAW", raw_eval)
print_eval("MITIGATED", clean_eval)

# ============================================================
# RUNTIME SUMMARY
# ============================================================

total_raw = raw_time + raw_lsh_time
total_clean = clean_time + clean_lsh_time

print("\n\nRUNTIME SUMMARY")
print("-" * 78)

print(f"RAW total:       {total_raw/60:.2f} minutes")
print(f"MITIGATED total: {total_clean/60:.2f} minutes")
print("Budget:          20.00 minutes")

print("\nNote:")
print("These are measured full-corpus MinHash + LSH runtimes.")
print("They exclude disk/database persistence overhead.")

# ============================================================
# ACTUAL DATABASE POPULATION
# ============================================================

print("\n\nPOPULATING ACTUAL 12,000-NOTICE LSH DATABASE")
print("-" * 78)

conn = sqlite3.connect(DB)

# Remove synthetic benchmark rows inserted earlier.
conn.execute("DELETE FROM lsh_buckets")
conn.commit()

# Populate using mitigated signatures.
rows = []

for i, sig in enumerate(clean_sig):

    notice_id = str(df.iloc[i]["notice_id"])

    for band in range(BANDS):

        s = band * ROWS
        e = s + ROWS

        bucket = hashlib.blake2b(
            sig[s:e].tobytes(),
            digest_size=8
        ).hexdigest()

        rows.append(
            (band, bucket, notice_id)
        )

conn.executemany(
    """
    INSERT INTO lsh_buckets
    (band, bucket, notice_id)
    VALUES (?, ?, ?)
    """,
    rows
)

conn.commit()

db_count = conn.execute(
    "SELECT COUNT(*) FROM lsh_buckets"
).fetchone()[0]

print(f"Actual LSH rows inserted: {db_count:,}")

# Planner check
test = conn.execute("""
    SELECT band, bucket
    FROM lsh_buckets
    LIMIT 1
""").fetchone()

band, bucket = test

plan = conn.execute("""
    EXPLAIN QUERY PLAN
    SELECT notice_id
    FROM lsh_buckets
    WHERE band=? AND bucket=?
""", (band, bucket)).fetchall()

print("\nActual production planner:")
for p in plan:
    print(p)

conn.close()

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 78)
print("ACTUAL FULL-CORPUS BENCHMARK COMPLETE")
print("=" * 78)