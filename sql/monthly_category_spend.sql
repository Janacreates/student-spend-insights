-- Spend per student, month and category (debits only). Feeds the forecasting panel.
SELECT
    student_id,
    month,
    category,
    ROUND(SUM(spend), 2) AS spend,
    COUNT(*)             AS n_txns
FROM transactions
WHERE direction = 'debit'
GROUP BY student_id, month, category
ORDER BY student_id, month, category;
