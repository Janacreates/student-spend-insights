-- Candidate subscriptions: the same merchant charged in >= 4 distinct months at a near-constant amount.
-- Uses the merchant string rather than the predicted category, so it works even when the
-- categoriser abstained.
SELECT
    student_id,
    description_norm                                  AS merchant,
    COUNT(DISTINCT month)                             AS months_charged,
    ROUND(AVG(spend), 2)                              AS avg_charge,
    MIN(month)                                        AS first_month,
    MAX(month)                                        AS last_month
FROM transactions
WHERE direction = 'debit'
GROUP BY student_id, description_norm
HAVING COUNT(DISTINCT month) >= 4
   AND COUNT(*) = COUNT(DISTINCT month)              -- at most one charge per month
   AND (MAX(spend) - MIN(spend)) <= 0.05 * AVG(spend)
   AND AVG(spend) < 200                              -- excludes rent-sized payments
ORDER BY student_id, avg_charge DESC;
