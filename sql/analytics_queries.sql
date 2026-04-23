-- ============================================================
-- Churn Decision Engine — SQL Analytical Queries
-- Dataset: IBM Telco Customer Churn
-- Assumes the CSV is loaded into a table named `customers`
-- Compatible with: SQLite, DuckDB, PostgreSQL, BigQuery
-- ============================================================


-- ── 1. Overall churn rate ──────────────────────────────────────────────────
SELECT
    COUNT(*)                                        AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers;


-- ── 2. Monthly revenue at risk (churned customers only) ───────────────────
SELECT
    ROUND(SUM(MonthlyCharges), 2)   AS total_monthly_revenue_at_risk,
    ROUND(AVG(MonthlyCharges), 2)   AS avg_monthly_charges_churned,
    COUNT(*)                        AS churned_customers
FROM customers
WHERE Churn = 'Yes';


-- ── 3. Churn rate by contract type ────────────────────────────────────────
SELECT
    Contract,
    COUNT(*)                                        AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ── 4. Churn rate by internet service type ────────────────────────────────
SELECT
    InternetService,
    COUNT(*)                                        AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- ── 5. Churn rate by tenure band ──────────────────────────────────────────
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-12 mo'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 mo'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 mo'
        ELSE '49+ mo'
    END                                             AS tenure_band,
    COUNT(*)                                        AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY MIN(tenure);


-- ── 6. Average monthly charges: churned vs retained ───────────────────────
SELECT
    Churn                               AS churn_status,
    COUNT(*)                            AS customers,
    ROUND(AVG(MonthlyCharges), 2)       AS avg_monthly_charges,
    ROUND(AVG(TotalCharges), 2)         AS avg_total_charges,
    ROUND(AVG(tenure), 1)               AS avg_tenure_months
FROM customers
GROUP BY Churn;


-- ── 7. Service adoption rate (what % of customers use each service) ────────
SELECT 'PhoneService'     AS service, ROUND(100.0 * SUM(CASE WHEN PhoneService     = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS adoption_pct FROM customers
UNION ALL
SELECT 'OnlineSecurity',   ROUND(100.0 * SUM(CASE WHEN OnlineSecurity   = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM customers
UNION ALL
SELECT 'OnlineBackup',     ROUND(100.0 * SUM(CASE WHEN OnlineBackup     = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM customers
UNION ALL
SELECT 'DeviceProtection', ROUND(100.0 * SUM(CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM customers
UNION ALL
SELECT 'TechSupport',      ROUND(100.0 * SUM(CASE WHEN TechSupport      = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM customers
UNION ALL
SELECT 'StreamingTV',      ROUND(100.0 * SUM(CASE WHEN StreamingTV      = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM customers
UNION ALL
SELECT 'StreamingMovies',  ROUND(100.0 * SUM(CASE WHEN StreamingMovies  = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM customers
ORDER BY adoption_pct DESC;


-- ── 8. Churn by payment method ────────────────────────────────────────────
SELECT
    PaymentMethod,
    COUNT(*)                                        AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- ── 9. High-value churner profile (revenue impact analysis) ──────────────
SELECT
    customerID,
    tenure,
    Contract,
    InternetService,
    MonthlyCharges,
    TotalCharges,
    ROUND(TotalCharges / NULLIF(tenure, 0), 2)  AS avg_monthly_revenue
FROM customers
WHERE Churn = 'Yes'
  AND MonthlyCharges >= (
      SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY MonthlyCharges)
      FROM customers
  )
ORDER BY MonthlyCharges DESC
LIMIT 20;


-- ── 10. Cohort churn matrix: contract × tenure band ───────────────────────
SELECT
    Contract,
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-12 mo'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 mo'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 mo'
        ELSE '49+ mo'
    END                                         AS tenure_band,
    COUNT(*)                                    AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                           AS churn_rate_pct
FROM customers
GROUP BY Contract, tenure_band
ORDER BY Contract, MIN(tenure);


-- ── 11. Estimated customer lifetime value (CLV) by segment ───────────────
SELECT
    Contract,
    COUNT(*)                                    AS customers,
    ROUND(AVG(MonthlyCharges), 2)               AS avg_monthly_charges,
    ROUND(AVG(tenure), 1)                       AS avg_tenure,
    ROUND(AVG(MonthlyCharges) * 24, 2)          AS estimated_24mo_clv
FROM customers
WHERE Churn = 'No'
GROUP BY Contract
ORDER BY estimated_24mo_clv DESC;


-- ── 12. Senior citizens: churn and revenue comparison ────────────────────
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
    COUNT(*)                                        AS customers,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly_charges,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers
GROUP BY SeniorCitizen;
