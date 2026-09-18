
from pathlib import Path
import csv
from datetime import datetime
import psycopg2


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SALES_DIR = Path("sales")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "annapurna",
    "user": "annapurna",
    "password": "annapurna123",
}


# --------------------------------------------------
# Timestamp parser
# --------------------------------------------------
def parse_timestamp(value):
    if not value:
        return None

    value = str(value).strip()

    # Unix timestamp (seconds since 1970-01-01)
    if value.isdigit():
        try:
            return datetime.fromtimestamp(int(value))
        except (ValueError, OSError):
            pass

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unknown timestamp format: {value}")

# --------------------------------------------------
# Database connection
# --------------------------------------------------

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()


files_processed = 0
rows_seen = 0
rows_inserted = 0


# --------------------------------------------------
# Process all CSV files
# --------------------------------------------------

for file_path in sorted(SALES_DIR.glob("*.csv")):

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            # Detect delimiter from header
            header_line = f.readline().strip()

            if ";" in header_line:
                delimiter = ";"
            else:
                delimiter = ","

            # Start reading from beginning
            f.seek(0)

            reader = csv.DictReader(
                f,
                delimiter=delimiter
            )

            for row in reader:

                # Clean column names
                clean_row = {
                    str(key).strip().lstrip("\ufeff"): value
                    for key, value in row.items()
                    if key is not None
                }

                # --------------------------------------------------
                # Format 1:
                # bill_no,line_no,product_code,qty,unit_price,line_type,ts
                #
                # Format 2:
                # ts,bill_no,line_no,line_type,product_code,unit_price,qty
                # --------------------------------------------------

                if "product_code" in clean_row:

                    bill_no = clean_row["bill_no"]
                    line_no = clean_row["line_no"]
                    product_code = clean_row["product_code"]
                    qty = clean_row["qty"]
                    unit_price = clean_row["unit_price"]
                    line_type = clean_row["line_type"]
                    ts = clean_row.get("ts")

                # --------------------------------------------------
                # Format 3:
                # bill_no;line_no;item_code;quantity;rate;type;txn_time
                # --------------------------------------------------

                elif "item_code" in clean_row:

                    bill_no = clean_row["bill_no"]
                    line_no = clean_row["line_no"]
                    product_code = clean_row["item_code"]
                    qty = clean_row["quantity"]
                    unit_price = clean_row["rate"]
                    line_type = clean_row["type"]
                    ts = clean_row.get("txn_time")

                else:
                    raise ValueError(
                        f"Unknown columns in {file_path.name}: "
                        f"{list(clean_row.keys())}"
                    )

                # Make sure timestamp exists
                if ts is None:
                    raise ValueError(
                        f"Missing timestamp in {file_path.name}. "
                        f"Columns found: {list(clean_row.keys())}"
                    )

                # Normalize timestamp
                timestamp = parse_timestamp(ts)

                rows_seen += 1

                # --------------------------------------------------
                # Idempotent insert
                #
                # bill_no + line_no is the primary key.
                # If the same billing line is received again,
                # PostgreSQL ignores it.
                # --------------------------------------------------

                cur.execute(
                    """
                    INSERT INTO sales
                        (
                            bill_no,
                            line_no,
                            product_code,
                            qty,
                            unit_price,
                            line_type,
                            ts
                        )
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (bill_no, line_no)
                    DO NOTHING
                    """,
                    (
                        bill_no,
                        int(line_no),
                        product_code,
                        int(qty) if qty else None,
                        unit_price if unit_price else None,
                        line_type,
                        timestamp,
                    ),
                )

                if cur.rowcount == 1:
                    rows_inserted += 1

        files_processed += 1

        # Progress message
        if files_processed % 100 == 0:
            print(
                f"Processed {files_processed} files | "
                f"Rows seen: {rows_seen} | "
                f"Rows inserted: {rows_inserted}"
            )

    except Exception:
        # Roll back anything from the failed transaction
        conn.rollback()

        print()
        print(f"ERROR in file: {file_path.name}")
        raise


# --------------------------------------------------
# Commit everything
# --------------------------------------------------

conn.commit()


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print(f"Files processed: {files_processed}")
print(f"Rows seen: {rows_seen}")
print(f"Rows inserted: {rows_inserted}")


# --------------------------------------------------
# Close database connection
# --------------------------------------------------

cur.close()
conn.close()
