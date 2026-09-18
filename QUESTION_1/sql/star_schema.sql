CREATE TABLE dim_store AS
SELECT store_id, store_name, address_line, city, state,
       region, floor_area_sqft, opened_on
FROM stores;

ALTER TABLE dim_store ADD PRIMARY KEY (store_id);

CREATE TABLE dim_category AS
SELECT category_id, category_name, department, gst_rate
FROM product_categories;

ALTER TABLE dim_category ADD PRIMARY KEY (category_id);

CREATE TABLE dim_product AS
SELECT product_sk, product_code, product_name, category_id,
       brand, pack_size, uom, valid_from, valid_to
FROM products;

ALTER TABLE dim_product ADD PRIMARY KEY (product_sk);

CREATE TABLE fact_sales AS
SELECT
    s.bill_no,
    s.line_no,
    p.product_sk,
    s.product_code,
    s.qty,
    s.unit_price,
    s.line_type,
    s.ts,
    s.ts::date AS sale_date,
    EXTRACT(DOW FROM s.ts)::integer AS day_of_week,
    DATE_TRUNC('month', s.ts)::date AS sale_month
FROM sales s
JOIN products p
    ON p.product_code = s.product_code
   AND s.ts::date >= p.valid_from
   AND s.ts::date <= p.valid_to
WHERE s.line_type IN ('SALE','RETURN');

ALTER TABLE fact_sales ADD PRIMARY KEY (bill_no, line_no);

CREATE INDEX ix_fact_sales_product
    ON fact_sales(product_sk);

CREATE INDEX ix_fact_sales_date
    ON fact_sales(sale_date);

CREATE INDEX ix_fact_sales_month
    ON fact_sales(sale_month);

CREATE INDEX ix_dim_product_category
    ON dim_product(category_id);