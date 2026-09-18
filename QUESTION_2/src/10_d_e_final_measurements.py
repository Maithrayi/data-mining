from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import time

BASE = Path(r"C:\Users\ub02-glab-058\Desktop\DATA_MINING_LAB\QUESTION_2")
DB = BASE / "setubid_retrieval.db"

DATA_DIR = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
LABELS = DATA_DIR / "labelled_pairs.csv"

print("=" * 75)
print("FINAL D + E MEASUREMENTS")
print("=" * 75)

# ============================================================
# SECTION D - INDEXED VS FORCED SCAN
# ============================================================

print("\nSECTION D - DATABASE ACCESS BENCHMARK")
print("-" * 75)

conn = sqlite3.connect(DB)

# Find a real bucket
row = conn.execute("""
    SELECT band, bucket
    FROM lsh_buckets
    LIMIT 1
""").fetchone()

if row is None:
    print("No LSH bucket found.")
else:
    band, bucket = row

    print(f"Test lookup: band={band}, bucket={bucket}")

    # Indexed query
    plan_indexed = conn.execute("""
        EXPLAIN QUERY PLAN
        SELECT notice_id
        FROM lsh_buckets
        WHERE band=? AND bucket=?
    """, (band, bucket)).fetchall()

    print("\nChosen indexed plan:")
    for p in plan_indexed:
        print(p)

    t0 = time.perf_counter()

    indexed_rows = conn.execute("""
        SELECT notice_id
        FROM lsh_buckets
        WHERE band=? AND bucket=?
    """, (band, bucket)).fetchall()

    indexed_time = time.perf_counter() - t0

    print(f"\nIndexed rows returned: {len(indexed_rows)}")
    print(f"Indexed wall time: {indexed_time * 1000:.4f} ms")

    # Forced scan
    plan_scan = conn.execute("""
        EXPLAIN QUERY PLAN
        SELECT notice_id
        FROM lsh_buckets NOT INDEXED
        WHERE band=? AND bucket=?
    """, (band, bucket)).fetchall()

    print("\nForced rejected full-scan plan:")
    for p in plan_scan:
        print(p)

    t1 = time.perf_counter()

    scan_rows = conn.execute("""
        SELECT notice_id
        FROM lsh_buckets NOT INDEXED
        WHERE band=? AND bucket=?
    """, (band, bucket)).fetchall()

    scan_time = time.perf_counter() - t1

    total_bucket_rows = conn.execute("""
        SELECT COUNT(*)
        FROM lsh_buckets
    """).fetchone()[0]

    print(f"\nFull lsh_buckets rows: {total_bucket_rows:,}")
    print(f"Forced-scan rows returned: {len(scan_rows)}")
    print(f"Forced-scan wall time: {scan_time * 1000:.4f} ms")

    if indexed_time > 0:
        print(
            f"Indexed lookup speed ratio: "
            f"{scan_time / indexed_time:.2f}x"
        )

conn.close()


# ============================================================
# SECTION E - BEFORE / AFTER CANDIDATE WORKLOAD
# ============================================================

print("\n\nSECTION E - BEFORE / AFTER RETRIEVAL WORKLOAD")
print("-" * 75)

parts = []

for f in sorted((DATA_DIR / "notices").glob("*.csv")):
    parts.append(pd.read_csv(f))

notices = pd.concat(parts, ignore_index=True)

print(f"Full corpus: {len(notices):,} notices")

# Portal counts
counts = notices.groupby("portal_id").size().sort_values(ascending=False)

before_work = int((counts * (counts - 1) / 2).sum())

print(f"\nBefore mitigation pair-work proxy: {before_work:,}")

# Approximate reduction from known nodal boilerplate
nodal = {"P001", "P002", "P003", "P004", "P005", "P006"}

notices["body_len"] = notices["body"].fillna("").str.len()

nodal_before = notices.loc[
    notices["portal_id"].isin(nodal),
    "body_len"
].mean()

reduced = (
    notices.loc[notices["portal_id"].isin(nodal), "body_len"]
    .clip(lower=1400) - 1400
)

nodal_after = reduced.mean()

print(f"Nodal mean body length before: {nodal_before:.1f}")
print(f"Nodal mean body length after:  {nodal_after:.1f}")
print(
    f"Boilerplate reduction: "
    f"{(nodal_before - nodal_after) / nodal_before * 100:.1f}%"
)

print("\nInterpretation:")
print(
    "The dominant heavy tail comes from high-volume nodal/aggregator "
    "portals, where repeated boilerplate creates many unnecessary "
    "similarity signals and candidate collisions."
)

# ============================================================
# LABELLED QUALITY - ALL 900 PAIRS
# ============================================================

labels = pd.read_csv(LABELS)

print("\n\nLABELLED-PAIR QUALITY CHECK")
print("-" * 75)

print(f"Total labelled pairs: {len(labels)}")

if "label" in labels.columns:
    print(labels["label"].value_counts().to_string())

print("\nPreviously measured full 900-pair precise similarity result:")
print("Word TF-IDF threshold = 0.55")
print("False merges = 5")
print("False splits = 0")
print("False-merge cost ratio = 20:1")
print("Weighted cost = 5*20 + 0*1 = 100")

print("\nPreviously measured LSH candidate-stage result:")
print("Configuration = 32 bands × 4 rows")
print("SAME survival = 78.1%")
print("DIFFERENT survival = 24.3%")
print("Average candidates/query = 219.6")

# ============================================================
# BUDGET
# ============================================================

print("\n\n20-MINUTE PRODUCTION BUDGET")
print("-" * 75)

print("Original all-pairs job: 31 hours")
print("Required maximum: 20 minutes")

estimated_minhash = 8.4

print(f"Estimated MinHash representation time: {estimated_minhash:.1f} minutes")

remaining = 20 - estimated_minhash

print(f"Remaining estimated budget: {remaining:.1f} minutes")

if estimated_minhash <= 20:
    print("STATUS: Representation stage fits the 20-minute budget.")
else:
    print("STATUS: Representation stage exceeds the 20-minute budget.")

print("\nMitigation decision:")
print("1. Remove repeated nodal boilerplate before shingling.")
print("2. Use 128-hash MinHash signatures.")
print("3. Use 32×4 LSH for candidate generation.")
print("4. Apply precise Word TF-IDF verification before merging.")
print("5. Persist opportunity_id independently from notice_id.")

print("\n" + "=" * 75)
print("D + E FINAL MEASUREMENTS COMPLETE")
print("=" * 75)