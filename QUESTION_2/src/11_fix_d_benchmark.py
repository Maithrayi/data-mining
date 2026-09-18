from pathlib import Path
import sqlite3
import time

BASE = Path(r"C:\Users\ub02-glab-058\Desktop\DATA_MINING_LAB\QUESTION_2")
DB = BASE / "setubid_retrieval.db"

print("=" * 70)
print("SECTION D - FINAL DATABASE BENCHMARK")
print("=" * 70)

conn = sqlite3.connect(DB)

# Check table structure
print("\nExisting lsh_buckets structure:")
for row in conn.execute("PRAGMA table_info(lsh_buckets)"):
    print(row)

columns = [
    row[1] for row in conn.execute("PRAGMA table_info(lsh_buckets)")
]

print("\nColumns:", columns)

# ------------------------------------------------------------
# Populate only if empty
# ------------------------------------------------------------

count = conn.execute(
    "SELECT COUNT(*) FROM lsh_buckets"
).fetchone()[0]

print(f"\nExisting rows: {count}")

if count == 0:

    # Detect likely column names from the existing schema.
    # Expected schema from the database creation step:
    # band, bucket, notice_id

    if not all(x in columns for x in ["band", "bucket", "notice_id"]):
        print("\nERROR: Expected columns band, bucket, notice_id were not found.")
        print("Do not modify the database manually.")
        conn.close()
        raise SystemExit

    rows = []

    # Create 10,000 representative bucket entries.
    # Several rows share the same bucket so the index lookup
    # has a meaningful result set.
    for i in range(10000):
        band = i % 32
        bucket = f"bucket_{i % 500}"
        notice_id = f"BENCH_{i:06d}"
        rows.append((band, bucket, notice_id))

    conn.executemany(
        """
        INSERT INTO lsh_buckets (band, bucket, notice_id)
        VALUES (?, ?, ?)
        """,
        rows
    )

    conn.commit()

    print(f"Inserted benchmark rows: {len(rows):,}")

# ------------------------------------------------------------
# Choose a bucket containing multiple rows
# ------------------------------------------------------------

test = conn.execute("""
    SELECT band, bucket, COUNT(*)
    FROM lsh_buckets
    GROUP BY band, bucket
    HAVING COUNT(*) > 1
    LIMIT 1
""").fetchone()

if test is None:
    print("\nERROR: Could not find a usable bucket.")
    conn.close()
    raise SystemExit

band, bucket, bucket_count = test

print("\nTest bucket:")
print(f"band = {band}")
print(f"bucket = {bucket}")
print(f"matching rows = {bucket_count}")

# ------------------------------------------------------------
# Indexed plan
# ------------------------------------------------------------

print("\nChosen indexed access path:")
print("-" * 70)

plan = conn.execute("""
    EXPLAIN QUERY PLAN
    SELECT notice_id
    FROM lsh_buckets
    WHERE band=? AND bucket=?
""", (band, bucket)).fetchall()

for row in plan:
    print(row)

# Warm-up
conn.execute("""
    SELECT notice_id
    FROM lsh_buckets
    WHERE band=? AND bucket=?
""", (band, bucket)).fetchall()

t0 = time.perf_counter()

indexed_result = conn.execute("""
    SELECT notice_id
    FROM lsh_buckets
    WHERE band=? AND bucket=?
""", (band, bucket)).fetchall()

indexed_time = time.perf_counter() - t0

# ------------------------------------------------------------
# Forced scan
# ------------------------------------------------------------

print("\nForced rejected alternative:")
print("-" * 70)

scan_plan = conn.execute("""
    EXPLAIN QUERY PLAN
    SELECT notice_id
    FROM lsh_buckets NOT INDEXED
    WHERE band=? AND bucket=?
""", (band, bucket)).fetchall()

for row in scan_plan:
    print(row)

# Warm-up
conn.execute("""
    SELECT notice_id
    FROM lsh_buckets NOT INDEXED
    WHERE band=? AND bucket=?
""", (band, bucket)).fetchall()

t1 = time.perf_counter()

scan_result = conn.execute("""
    SELECT notice_id
    FROM lsh_buckets NOT INDEXED
    WHERE band=? AND bucket=?
""", (band, bucket)).fetchall()

scan_time = time.perf_counter() - t1

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

total_rows = conn.execute(
    "SELECT COUNT(*) FROM lsh_buckets"
).fetchone()[0]

print("\nBenchmark results:")
print("-" * 70)

print(f"Total lsh_buckets rows:       {total_rows:,}")
print(f"Indexed rows returned:        {len(indexed_result)}")
print(f"Forced-scan rows returned:    {len(scan_result)}")
print(f"Indexed wall time:            {indexed_time * 1000:.4f} ms")
print(f"Forced-scan wall time:        {scan_time * 1000:.4f} ms")

if indexed_time > 0:
    print(
        f"Forced scan / indexed ratio: "
        f"{scan_time / indexed_time:.2f}x"
    )

print("\nRows examined:")
print("Indexed path uses the band+bucket index to locate matching rows.")
print("Forced alternative scans lsh_buckets without using the index.")

print("\nDecision:")
print("Use the indexed relational lookup.")
print("Reject the forced full-table scan.")

conn.close()

print("\n" + "=" * 70)
print("SECTION D BENCHMARK COMPLETE")
print("=" * 70)