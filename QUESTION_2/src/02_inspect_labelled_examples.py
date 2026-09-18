from pathlib import Path
import pandas as pd


DATA_DIR = Path(r"C:\Users\ub02-glab-058\Desktop\data_2")
NOTICES_DIR = DATA_DIR / "notices"
LABELS_FILE = DATA_DIR / "labelled_pairs.csv"


print("=" * 80)
print("QUESTION 2 - LABELLED PAIR INSPECTION")
print("=" * 80)


# ---------------------------------------------------------
# Load labels
# ---------------------------------------------------------
labels = pd.read_csv(LABELS_FILE)

same_pair = labels[labels["label"] == "same"].iloc[0]
different_pair = labels[labels["label"] == "different"].iloc[0]

print("\nSelected SAME pair:")
print(same_pair.to_dict())

print("\nSelected DIFFERENT pair:")
print(different_pair.to_dict())


# ---------------------------------------------------------
# Load all notices
# ---------------------------------------------------------
notice_frames = []

for file in sorted(NOTICES_DIR.glob("*.csv")):
    part = pd.read_csv(file)
    notice_frames.append(part)

notices = pd.concat(notice_frames, ignore_index=True)

print("\nTotal notices loaded:", len(notices))


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def show_notice(notice_id):
    row = notices[notices["notice_id"] == notice_id]

    if row.empty:
        print(f"\nERROR: {notice_id} not found")
        return

    row = row.iloc[0]

    print("\n" + "-" * 80)
    print(f"NOTICE: {notice_id}")
    print("-" * 80)
    print("Portal:", row["portal_id"])
    print("Published:", row["published_at"])
    print("Title:", row["title"])
    print("Estimated value:", row["estimated_value"])
    print("Closing date:", row["closing_date"])
    print("Body characters:", len(str(row["body"])))

    body = str(row["body"])

    print("\nBODY PREVIEW:")
    print(body[:2000])


# ---------------------------------------------------------
# Display pairs
# ---------------------------------------------------------
print("\n\n" + "=" * 80)
print("SAME PAIR")
print("=" * 80)

show_notice(same_pair["notice_id_a"])
show_notice(same_pair["notice_id_b"])


print("\n\n" + "=" * 80)
print("DIFFERENT PAIR")
print("=" * 80)

show_notice(different_pair["notice_id_a"])
show_notice(different_pair["notice_id_b"])


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)