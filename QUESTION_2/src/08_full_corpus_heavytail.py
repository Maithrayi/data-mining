from pathlib import Path
import pandas as pd
import numpy as np
import time


DATA_DIR = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
NOTICES_DIR = DATA_DIR / "notices"
PROFILE_FILE = DATA_DIR / "portal_profiles.md"


print("=" * 70)
print("SECTION E - FULL CORPUS HEAVY-TAIL ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# Load full corpus
# ---------------------------------------------------------

parts = []

for file in sorted(NOTICES_DIR.glob("*.csv")):
    parts.append(pd.read_csv(file))

notices = pd.concat(parts, ignore_index=True)

print(f"\nTotal notices: {len(notices):,}")
print(f"Total portals: {notices['portal_id'].nunique():,}")


# ---------------------------------------------------------
# Portal workload
# ---------------------------------------------------------

portal_counts = (
    notices.groupby("portal_id")
    .size()
    .sort_values(ascending=False)
)

print("\nTop 15 portals by notice count:")
print("-" * 70)

print(portal_counts.head(15).to_string())


# ---------------------------------------------------------
# Heavy-tail statistics
# ---------------------------------------------------------

q50 = portal_counts.quantile(0.50)
q90 = portal_counts.quantile(0.90)
q95 = portal_counts.quantile(0.95)
q99 = portal_counts.quantile(0.99)

print("\nPortal workload distribution:")
print("-" * 70)

print(f"Median portal notices: {q50:.0f}")
print(f"90th percentile:       {q90:.0f}")
print(f"95th percentile:       {q95:.0f}")
print(f"99th percentile:       {q99:.0f}")
print(f"Maximum:               {portal_counts.max():.0f}")


# ---------------------------------------------------------
# Concentration
# ---------------------------------------------------------

for n in [5, 10, 15]:

    top_n = portal_counts.head(n).sum()
    percentage = top_n / len(notices) * 100

    print(
        f"Top {n} portals account for "
        f"{top_n:,} notices "
        f"({percentage:.1f}%)"
    )


# ---------------------------------------------------------
# Candidate-work proxy
#
# Use a simple mechanical model:
# portals with more notices generate more pairwise work.
# ---------------------------------------------------------

portal_work = portal_counts * (portal_counts - 1) / 2

total_work = portal_work.sum()

print("\nPairwise-work proxy:")
print("-" * 70)
print(f"Total within-portal pair work: {total_work:,.0f}")

for n in [5, 10, 15]:

    work = portal_work.head(n).sum()

    print(
        f"Top {n} portals account for "
        f"{work / total_work * 100:.1f}% "
        f"of this work proxy"
    )


# ---------------------------------------------------------
# Profile evidence
# ---------------------------------------------------------

print("\nPortal profile observations:")
print("-" * 70)

if PROFILE_FILE.exists():

    profile_text = PROFILE_FILE.read_text(
        encoding="utf-8"
    )

    keywords = [
        "P001",
        "P002",
        "P003",
        "P004",
        "P005",
        "P006",
        "boilerplate",
        "aggregator",
        "truncat"
    ]

    for keyword in keywords:

        if keyword.lower() in profile_text.lower():

            print(f"Profile evidence contains: {keyword}")


# ---------------------------------------------------------
# Simple mitigation estimate
#
# Deduplicate repeated boilerplate by removing the first
# 1400 characters from known nodal portals.
# ---------------------------------------------------------

nodal = {"P001", "P002", "P003", "P004", "P005", "P006"}

notices["body_len"] = notices["body"].fillna("").str.len()

before = notices.loc[
    notices["portal_id"].isin(nodal),
    "body_len"
].mean()

mitigated_lengths = notices.loc[
    notices["portal_id"].isin(nodal),
    "body_len"
].clip(lower=0) - 1400

mitigated_lengths = mitigated_lengths.clip(lower=0)

after = mitigated_lengths.mean()

print("\nBoilerplate mitigation:")
print("-" * 70)

print(
    f"Mean nodal body length before: "
    f"{before:.1f} chars"
)

print(
    f"Mean nodal body length after:  "
    f"{after:.1f} chars"
)

print(
    f"Reduction: "
    f"{(before-after)/before*100:.1f}%"
)


# ---------------------------------------------------------
# Runtime budget statement
# ---------------------------------------------------------

print("\n20-minute budget:")
print("-" * 70)

print(
    "Original all-pairs job: 31 hours "
    "(given in problem statement)."
)

print(
    "Required budget: <= 20 minutes."
)

print(
    "Sublinear LSH candidate retrieval is used "
    "to avoid full all-pairs comparison."
)

print(
    "Heavy-tail mitigation removes repeated nodal "
    "boilerplate before similarity/candidate generation."
)


print("\n" + "=" * 70)
print("SECTION E COMPLETE")
print("=" * 70)