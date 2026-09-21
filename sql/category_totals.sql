-- Where does the money go? Share of all debit spend by category.
SELECT
    category,
    ROUND(SUM(spend), 2)                                   AS total_spend,
    COUNT(*)                                               AS n_txns,
    ROUND(100.0 * SUM(spend) / SUM(SUM(spend)) OVER (), 2) AS pct_of_spend
FROM transactions
WHERE direction = 'debit'
GROUP BY category
ORDER BY total_spend DESC;
