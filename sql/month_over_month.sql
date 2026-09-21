-- Month-over-month change in total spend per category (window function: LAG).
WITH monthly AS (
    SELECT month, category, SUM(spend) AS total_spend
    FROM transactions
    WHERE direction = 'debit'
    GROUP BY month, category
)
SELECT
    month,
    category,
    ROUND(total_spend, 2) AS total_spend,
    ROUND(100.0 * (total_spend - LAG(total_spend) OVER (PARTITION BY category ORDER BY month))
          / LAG(total_spend) OVER (PARTITION BY category ORDER BY month), 1) AS pct_change_vs_prev_month
FROM monthly
ORDER BY category, month;
