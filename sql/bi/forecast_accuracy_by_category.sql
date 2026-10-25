-- Forecast error, interval coverage and day-15 budget-alert quality per category, over all
-- test months. (The README headline scores the alert on the held-out half only, so the
-- precision and recall here differ slightly.)
SELECT
    c.category,
    COUNT(*)                                                    AS predictions,
    ROUND(AVG(ff.abs_error), 2)                                 AS mae,
    ROUND(100.0 * AVG(ff.in_interval), 1)                       AS pct_in_80_interval,
    ROUND(100.0 * AVG(ff.is_material_overspend), 1)             AS overspend_rate_pct,
    ROUND(100.0 * SUM(ff.alert_day15 * ff.is_material_overspend)
          / NULLIF(SUM(ff.alert_day15), 0), 1)                  AS alert_precision_pct,
    ROUND(100.0 * SUM(ff.alert_day15 * ff.is_material_overspend)
          / NULLIF(SUM(ff.is_material_overspend), 0), 1)        AS alert_recall_pct
FROM fact_forecast AS ff
JOIN dim_category AS c ON c.category_key = ff.category_key
GROUP BY c.category
ORDER BY mae DESC;
