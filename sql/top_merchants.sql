-- Top merchants by spend, excluding lumpy fixed payments, with share of the remaining spend.
SELECT
    description_norm                                         AS merchant,
    category,
    COUNT(*)                                                 AS n_txns,
    ROUND(SUM(spend), 2)                                     AS total_spend,
    ROUND(100.0 * SUM(spend) / SUM(SUM(spend)) OVER (), 2)   AS pct_of_variable_spend
FROM transactions
WHERE direction = 'debit'
  AND category NOT IN ('rent', 'tuition')
GROUP BY description_norm, category
ORDER BY total_spend DESC
LIMIT 15;
