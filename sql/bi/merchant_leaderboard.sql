-- Top merchants by controllable spend: reach (distinct students) and average ticket size.
SELECT
    m.merchant_name,
    c.category,
    COUNT(*)                      AS n_txns,
    COUNT(DISTINCT f.student_key) AS students,
    ROUND(SUM(f.spend), 2)        AS total_spend,
    ROUND(AVG(f.spend), 2)        AS avg_ticket
FROM fact_transactions AS f
JOIN dim_merchant AS m ON m.merchant_key = f.merchant_key
JOIN dim_category AS c ON c.category_key = f.category_key
WHERE f.direction = 'debit'
  AND c.category_group = 'controllable'
GROUP BY m.merchant_name, c.category
ORDER BY total_spend DESC
LIMIT 15;
