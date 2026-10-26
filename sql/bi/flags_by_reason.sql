-- Anomaly flags by reason and category: how many, how much money, how many students.
SELECT
    fl.reason,
    c.category,
    COUNT(*)                       AS flags,
    COUNT(DISTINCT fl.student_key) AS students,
    ROUND(SUM(fl.spend), 2)        AS flagged_spend
FROM fact_flags AS fl
JOIN dim_category AS c ON c.category_key = fl.category_key
GROUP BY fl.reason, c.category
ORDER BY flags DESC;
