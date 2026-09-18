# Annapurna Store Data Platform

A data platform for Annapurna Store that provides consistent monthly revenue reporting, historical pricing, and analytical access across 12 supermarkets.

## Architecture

```text
Daily Sales Files
       ↓
     MinIO
       ↓
  PostgreSQL
       ↓
    DuckDB
       ↓
Analytics & Reconciliation
```

## Project Tasks

### Task A – MinIO Object Storage

* Organized sales files using store/year/month partitions.
* Loaded 4,457 source files into MinIO.
* Verified 68,706,877 bytes.
* Created 144 store-month partitions.
* Verified direct DuckDB access to MinIO.

### Task B – PostgreSQL Loading

* Created the raw `sales` table.
* Loaded 1,120,924 unique bill-line records.
* Implemented idempotent loading.
* Verified repeated runs insert zero additional rows.
* Used checksum validation to confirm data consistency.

### Task C – Master Data & Star Schema

* Loaded stores, product categories, products and price revisions.
* Created `dim_store`, `dim_category`, `dim_product` and `fact_sales`.
* Handled product-code reuse using `product_sk` and validity dates.
* Verified 762,021 sale/return lines matched correctly.

### Task D – Historical Pricing

* Joined sales with historical price revisions.
* Used product validity dates to select the applicable historical price.
* Verified exactly one applicable price for every sale/return line.

### Task E – DuckDB Analytics

* Connected DuckDB to PostgreSQL.
* Configured DuckDB to read MinIO objects directly.
* Demonstrated queries combining MinIO and PostgreSQL data.
* Used `EXPLAIN` to inspect the analytical query plan.

### Task F – Finance Reconciliation

* Reconciled platform revenue against `finance_monthly.csv`.
* Applied the documented business-date and revenue rules.
* Identified the July S07 source-data gap.
* Documented unresolved March and December reconciliation differences.

## Repository Structure

```text
Annapurna-Store-Data-Platform/
├── platform/
│   └── docker-compose.yml
├── sql/
│   ├── masters.sql
│   └── star_schema.sql
├── scripts/
│   └── load_sales.py
├── data/
│   └── finance_monthly.csv
├── docs/
│   ├── billing_notes.md
│   ├── Annapurna_Store_Data_Platform_Tasks_A-F.docx
│   └── Annapurna_Store_Important_Queries_Taskwise.docx
├── README.md
└── .gitignore
```

## Technologies

* MinIO
* PostgreSQL 16
* DuckDB
* Python
* SQL
* Docker
* PowerShell

## Important Note

Raw daily sales files are not included in this repository. The repository contains the platform code, SQL, documentation and reference data required to reproduce and understand the solution.
