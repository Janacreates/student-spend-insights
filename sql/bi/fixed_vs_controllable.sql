-- Monthly debit spend split into fixed (rent, tuition), controllable and other spend,
-- with each group's share of that month (window function over the joined rows).
SELECT
    d.month_start                                                           AS month,
    c.category_group,
    ROUND(SUM(f.spend), 2)                                                  AS spend,
    ROUND(100.0 * SUM(f.spend) / SUM(SUM(f.spend)) OVER (PARTITION BY d.month_start), 1) AS pct_of_month
FROM fact_transactions AS f
JOIN dim_category AS c ON c.category_key = f.category_key
JOIN dim_date     AS d ON d.date_key     = f.date_key
WHERE f.direction = 'debit'
GROUP BY d.month_start, c.category_group
ORDER BY month, c.category_group;
