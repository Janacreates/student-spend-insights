-- Average monthly controllable spend per student, by housing type and city.
-- Joins the transaction fact to three dimensions (student, category, date).
WITH per_student_month AS (
    SELECT
        s.housing,
        s.city,
        f.student_key,
        d.month_start,
        SUM(f.spend) AS spend
    FROM fact_transactions AS f
    JOIN dim_student  AS s ON s.student_key  = f.student_key
    JOIN dim_category AS c ON c.category_key = f.category_key
    JOIN dim_date     AS d ON d.date_key     = f.date_key
    WHERE f.direction = 'debit'
      AND c.category_group = 'controllable'
    GROUP BY s.housing, s.city, f.student_key, d.month_start
)
SELECT
    housing,
    city,
    COUNT(DISTINCT student_key) AS students,
    ROUND(AVG(spend), 2)        AS avg_monthly_spend
FROM per_student_month
GROUP BY housing, city
ORDER BY avg_monthly_spend DESC;
