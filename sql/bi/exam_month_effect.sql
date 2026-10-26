-- Dining spend per student-month in exam months (April, December) versus other months, by
-- housing type. The simulator builds this effect in, so this shows the method, not a discovery.
WITH per_student_month AS (
    SELECT
        s.housing,
        d.is_exam_month,
        f.student_key,
        d.month_start,
        SUM(f.spend) AS spend
    FROM fact_transactions AS f
    JOIN dim_student  AS s ON s.student_key  = f.student_key
    JOIN dim_category AS c ON c.category_key = f.category_key
    JOIN dim_date     AS d ON d.date_key     = f.date_key
    WHERE f.direction = 'debit'
      AND c.category = 'dining'
    GROUP BY s.housing, d.is_exam_month, f.student_key, d.month_start
)
SELECT
    housing,
    ROUND(AVG(CASE WHEN is_exam_month = 1 THEN spend END), 2) AS exam_month_avg,
    ROUND(AVG(CASE WHEN is_exam_month = 0 THEN spend END), 2) AS other_month_avg,
    ROUND(100.0 * (AVG(CASE WHEN is_exam_month = 1 THEN spend END)
                   / AVG(CASE WHEN is_exam_month = 0 THEN spend END) - 1), 1) AS lift_pct
FROM per_student_month
GROUP BY housing
ORDER BY lift_pct DESC;
