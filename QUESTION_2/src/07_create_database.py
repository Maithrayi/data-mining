from pathlib import Path
import sqlite3


PROJECT_DIR = Path(
    r"C:\Users\ub02-glab-058\Desktop\DATA_MINING_LAB\QUESTION_2"
)

DB_FILE = PROJECT_DIR / "setubid_retrieval.db"

conn = sqlite3.connect(DB_FILE)

cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS lsh_buckets;
DROP TABLE IF EXISTS signatures;
DROP TABLE IF EXISTS notices;
DROP TABLE IF EXISTS opportunities;
DROP TABLE IF EXISTS opportunity_members;

CREATE TABLE notices (
    notice_id TEXT PRIMARY KEY,
    portal_id TEXT NOT NULL,
    published_at TEXT,
    title TEXT,
    estimated_value INTEGER,
    closing_date TEXT
);

CREATE TABLE signatures (
    notice_id TEXT PRIMARY KEY,
    signature BLOB NOT NULL,
    FOREIGN KEY(notice_id) REFERENCES notices(notice_id)
);

CREATE TABLE lsh_buckets (
    band INTEGER NOT NULL,
    bucket TEXT NOT NULL,
    notice_id TEXT NOT NULL,
    FOREIGN KEY(notice_id) REFERENCES notices(notice_id)
);

CREATE INDEX idx_lsh_bucket
ON lsh_buckets(band, bucket);

CREATE TABLE opportunities (
    opportunity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE opportunity_members (
    opportunity_id INTEGER NOT NULL,
    notice_id TEXT PRIMARY KEY,
    FOREIGN KEY(opportunity_id) REFERENCES opportunities(opportunity_id),
    FOREIGN KEY(notice_id) REFERENCES notices(notice_id)
);

CREATE INDEX idx_opportunity_members
ON opportunity_members(opportunity_id);
""")

conn.commit()

print("=" * 70)
print("SECTION D - RELATIONAL RETRIEVAL STRUCTURE")
print("=" * 70)

print("\nDatabase:")
print(DB_FILE)

print("\nTables:")
for row in cur.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name
"""):
    print(" ", row[0])

print("\nIndexes:")
for row in cur.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='index'
    ORDER BY name
"""):
    print(" ", row[0])

print("\nCandidate lookup query plan:")

plan = cur.execute("""
EXPLAIN QUERY PLAN
SELECT notice_id
FROM lsh_buckets
WHERE band = 1
AND bucket = 'example_bucket'
""").fetchall()

for row in plan:
    print(row)

print("\nStable card ID design:")
print("opportunity_id is persistent and independent of notice_id.")
print("New copies are added to opportunity_members.")
print("Existing bookmarks continue to reference opportunity_id.")

conn.close()

print("\n" + "=" * 70)
print("SECTION D COMPLETE")
print("=" * 70)