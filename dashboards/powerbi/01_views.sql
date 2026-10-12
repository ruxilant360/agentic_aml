-- =====================================================================
--  AML dashboard views for Power BI
-- =====================================================================
--
--  Run this once against the AML database. It creates read-only views;
--  it never writes to, alters or drops `transactions` / `model_outputs`.
--
--  Why views instead of pointing Power BI at the base tables:
--    * `transactions` has millions of rows. Importing it raw is slow and
--      the interesting numbers are all aggregates anyway.
--    * Power BI/DAX cannot cheaply build a cumulative precision-recall
--      curve, nor parse the `shap_values` JSON column. Both are done here,
--      in SQL, once.
--
--  COLUMN NAMING -- read this before running
--  -----------------------------------------
--  This file uses snake_case (`transaction_id`, `oldbalance_org`,
--  `newbalance_dest`), matching `src/training/data_extraction.py` and
--  `src/tools/sql_query_tool.py`. Note that `src/db/schema.py` declares
--  camelCase instead (`transID`, `oldbalanceOrg`, and a typo'd
--  `newbalancDest`), so the two halves of the repo disagree. Confirm which
--  one your database actually has:
--
--      SELECT table_name, column_name, data_type
--      FROM information_schema.columns
--      WHERE table_name IN ('transactions', 'model_outputs')
--      ORDER BY table_name, ordinal_position;
--
--  If it reports camelCase, rename the identifiers below -- quoting any
--  mixed-case name, e.g. t."transID". (The Streamlit dashboard resolves
--  this automatically at runtime; plain SQL cannot.)
--
--  LABEL COLUMNS
--  -------------
--  `is_fraud` / `is_flagged_fraud` are boolean in the ORM but 0/1 in the
--  source CSV, so a database loaded straight from the CSV may hold either.
--  Every test below casts to text and compares, which reads correctly for
--  both without a per-column type lookup.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. Enriched transaction fact
-- ---------------------------------------------------------------------
-- One row per transaction, with the derived fields a dashboard needs:
-- a day/hour from `step`, an order-of-magnitude amount band, balance
-- deltas, and the behavioural patterns as boolean flags.
--
-- The patterns are descriptive account behaviour, not verdicts: they are
-- context an analyst reads next to the model's score.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_transactions AS
SELECT
    t.transaction_id,
    t.step,
    (t.step / 24)                                        AS sim_day,
    MOD(t.step, 24)                                      AS hour_of_day,
    t.type                                               AS channel,
    t.amount,
    t.name_orig,
    t.name_dest,
    t.oldbalance_org,
    t.newbalance_orig,
    t.oldbalance_dest,
    t.newbalance_dest,
    (t.newbalance_orig - t.oldbalance_org)               AS orig_balance_delta,
    (t.newbalance_dest - t.oldbalance_dest)              AS dest_balance_delta,
    CASE WHEN t.oldbalance_org > 0
         THEN t.amount / t.oldbalance_org END            AS amount_to_balance_ratio,

    CASE
        WHEN t.amount <      100 THEN '1. < 100'
        WHEN t.amount <     1000 THEN '2. 100 - 1K'
        WHEN t.amount <    10000 THEN '3. 1K - 10K'
        WHEN t.amount <   100000 THEN '4. 10K - 100K'
        WHEN t.amount <  1000000 THEN '5. 100K - 1M'
        WHEN t.amount < 10000000 THEN '6. 1M - 10M'
        ELSE                          '7. >= 10M'
    END                                                  AS amount_band,

    -- Expected post-transaction originator balance. CASH_IN credits the
    -- originator; every other channel debits it, so the identity flips.
    CASE WHEN t.type IN ('CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
         THEN t.oldbalance_org - t.amount
         ELSE t.oldbalance_org + t.amount
    END                                                  AS expected_newbalance_orig,

    -- Behavioural patterns
    (t.newbalance_orig = 0 AND t.oldbalance_org > 0
        AND t.amount >= t.oldbalance_org)                AS pat_balance_drained,
    (t.amount > 0 AND t.oldbalance_dest = 0
        AND t.newbalance_dest = 0)                       AS pat_dest_balances_zero,
    (ABS(t.newbalance_orig -
         CASE WHEN t.type IN ('CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
              THEN t.oldbalance_org - t.amount
              ELSE t.oldbalance_org + t.amount END) > 0.01)
                                                         AS pat_ledger_mismatch,
    (t.type IN ('CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
        AND t.amount > t.oldbalance_org)                 AS pat_amount_over_balance,
    (t.amount > 0 AND MOD(t.amount::numeric, 1000) = 0)  AS pat_round_amount,
    (t.amount >= 200000)                                 AS pat_large_amount,

    COALESCE(t.is_flagged_fraud::text, '0') IN ('1', 't', 'true')  AS rule_flagged,
    CASE WHEN t.is_fraud IS NULL THEN NULL
         ELSE t.is_fraud::text IN ('1', 't', 'true') END AS label_positive,
    (t.is_fraud IS NOT NULL)                             AS is_labelled
FROM transactions t;


-- ---------------------------------------------------------------------
-- 2. Model outputs, with a score band for slicers
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_model_outputs AS
SELECT
    mo.transaction_id,
    mo.score,
    COALESCE(mo.flag::text, '0') IN ('1', 't', 'true')   AS model_flagged,
    mo.created_at,
    -- Ten equal score bands, numbered so they sort correctly as text. An
    -- ordered band, so colour it with a single-hue ordinal ramp, never a rainbow.
    band.idx                                             AS score_band_order,
    LPAD((band.idx + 1)::text, 2, '0') || '. '
        || (band.idx / 10.0)::numeric(3, 1)::text || ' - '
        || ((band.idx + 1) / 10.0)::numeric(3, 1)::text  AS score_band
FROM model_outputs mo
CROSS JOIN LATERAL (
    SELECT FLOOR(LEAST(GREATEST(COALESCE(mo.score, 0), 0), 0.999) * 10)::int AS idx
) AS band;


-- ---------------------------------------------------------------------
-- 3. SHAP values, unpivoted to long form
-- ---------------------------------------------------------------------
-- `shap_values` is a JSON object per transaction. Power BI cannot usefully
-- slice that, so it is expanded here to one row per (transaction, feature).
-- This is the table the explanation visuals bind to.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_shap_long AS
SELECT
    mo.transaction_id,
    kv.key                                   AS feature,
    kv.value::double precision               AS shap_value,
    ABS(kv.value::double precision)          AS abs_shap_value,
    CASE WHEN kv.value::double precision >= 0 THEN 'raises score'
         ELSE 'lowers score' END             AS shap_direction,
    mo.score,
    COALESCE(mo.flag::text, '0') IN ('1', 't', 'true')  AS model_flagged
FROM model_outputs mo,
     JSONB_EACH_TEXT(mo.shap_values::jsonb) AS kv
WHERE mo.shap_values IS NOT NULL;


-- ---------------------------------------------------------------------
-- 4. Daily metrics (small, pre-aggregated -- import this one)
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_daily_metrics AS
WITH tx AS (
    SELECT (t.step / 24) AS sim_day,
           COUNT(*)                                            AS n_tx,
           SUM(t.amount)                                       AS total_amount,
           AVG(t.amount)                                       AS avg_amount,
           COUNT(t.is_fraud)                                   AS n_labelled,
           COUNT(*) FILTER (WHERE t.is_fraud::text IN ('1', 't', 'true'))        AS n_positive,
           COUNT(*) FILTER (WHERE t.is_flagged_fraud::text IN ('1', 't', 'true')) AS n_rule_flagged
    FROM transactions t
    GROUP BY 1
),
scored AS (
    SELECT (t.step / 24) AS sim_day,
           COUNT(*)                                            AS n_scored,
           AVG(mo.score)                                       AS avg_score,
           PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY mo.score) AS p95_score,
           COUNT(*) FILTER (WHERE mo.flag::text IN ('1', 't', 'true')) AS n_model_flagged
    FROM model_outputs mo
    JOIN transactions t ON t.transaction_id = mo.transaction_id
    GROUP BY 1
)
SELECT tx.sim_day,
       tx.n_tx, tx.total_amount, tx.avg_amount,
       tx.n_labelled, tx.n_positive, tx.n_rule_flagged,
       COALESCE(scored.n_scored, 0)        AS n_scored,
       scored.avg_score,
       scored.p95_score,
       COALESCE(scored.n_model_flagged, 0) AS n_model_flagged
FROM tx LEFT JOIN scored USING (sim_day)
ORDER BY tx.sim_day;


-- ---------------------------------------------------------------------
-- 5. Channel metrics
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_channel_metrics AS
SELECT t.type                                            AS channel,
       COUNT(*)                                          AS n_tx,
       SUM(t.amount)                                     AS total_amount,
       AVG(t.amount)                                     AS avg_amount,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.amount) AS median_amount,
       COUNT(t.is_fraud)                                 AS n_labelled,
       COUNT(*) FILTER (WHERE t.is_fraud::text IN ('1', 't', 'true'))        AS n_positive,
       COUNT(*) FILTER (WHERE t.is_flagged_fraud::text IN ('1', 't', 'true')) AS n_rule_flagged
FROM transactions t
GROUP BY 1
ORDER BY 2 DESC;


-- ---------------------------------------------------------------------
-- 6. Behavioural pattern metrics, long form
-- ---------------------------------------------------------------------
-- One row per pattern, so a single bar chart can compare them.
-- `hit_rate` is the share of each pattern's labelled rows that are
-- positive: wide coverage with a near-zero hit rate means noise, not
-- signal.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_pattern_metrics AS
WITH flat AS (
    SELECT p.pattern, p.hit, v.label_positive, v.is_labelled
    FROM vw_aml_transactions v
    CROSS JOIN LATERAL (VALUES
        ('Originator balance fully drained',     v.pat_balance_drained),
        ('Destination balances both zero',       v.pat_dest_balances_zero),
        ('Ledger identity broken (originator)',  v.pat_ledger_mismatch),
        ('Amount exceeds originator balance',    v.pat_amount_over_balance),
        ('Round-number amount (multiple of 1K)', v.pat_round_amount),
        ('Amount >= 200,000',                    v.pat_large_amount)
    ) AS p(pattern, hit)
),
book AS (SELECT COUNT(*) AS total_rows FROM vw_aml_transactions)
SELECT flat.pattern,
       COUNT(*) FILTER (WHERE flat.hit)                                      AS n_tx,
       COUNT(*) FILTER (WHERE flat.hit AND flat.is_labelled)                 AS n_labelled,
       COUNT(*) FILTER (WHERE flat.hit AND flat.label_positive)              AS n_positive,
       COUNT(*) FILTER (WHERE flat.hit AND flat.label_positive)::double precision
         / NULLIF(COUNT(*) FILTER (WHERE flat.hit AND flat.is_labelled), 0)  AS hit_rate,
       COUNT(*) FILTER (WHERE flat.hit)::double precision
         / NULLIF(MAX(book.total_rows), 0)                                   AS share_of_book
FROM flat CROSS JOIN book
GROUP BY 1
ORDER BY 2 DESC;


-- ---------------------------------------------------------------------
-- 7. Destination fan-in
-- ---------------------------------------------------------------------
-- Accounts receiving from many distinct originators -- the structural
-- shape a mule network leaves behind. A high count is a reason to look.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_dest_fan_in AS
SELECT t.name_dest,
       COUNT(DISTINCT t.name_orig)                                    AS n_origins,
       COUNT(*)                                                       AS n_tx,
       SUM(t.amount)                                                  AS total_amount,
       COUNT(*) FILTER (WHERE t.is_fraud::text IN ('1', 't', 'true')) AS n_positive
FROM transactions t
GROUP BY 1
HAVING COUNT(DISTINCT t.name_orig) >= 2
ORDER BY n_origins DESC, total_amount DESC;


-- ---------------------------------------------------------------------
-- 8. Score buckets -- the base for every curve below
-- ---------------------------------------------------------------------
-- 101 rows: the labelled score histogram at 0.01 resolution. Everything
-- downstream (PR sweep, macro-PR, lift) is computed from this instead of
-- re-scanning millions of rows per threshold.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_score_buckets AS
WITH labelled AS (
    SELECT mo.score,
           t.is_fraud::text IN ('1', 't', 'true') AS y
    FROM model_outputs mo
    JOIN transactions t ON t.transaction_id = mo.transaction_id
    WHERE t.is_fraud IS NOT NULL
),
bucketed AS (
    SELECT FLOOR(LEAST(GREATEST(score, 0), 1) * 100)::int AS b,
           COUNT(*) FILTER (WHERE y)     AS n_pos,
           COUNT(*) FILTER (WHERE NOT y) AS n_neg
    FROM labelled
    GROUP BY 1
),
grid AS (SELECT generate_series(0, 100) AS b)
SELECT grid.b,
       grid.b / 100.0                      AS score_low,
       (grid.b + 1) / 100.0                AS score_high,
       COALESCE(bucketed.n_pos, 0)         AS n_pos,
       COALESCE(bucketed.n_neg, 0)         AS n_neg,
       COALESCE(bucketed.n_pos, 0)
         + COALESCE(bucketed.n_neg, 0)     AS n_total
FROM grid LEFT JOIN bucketed USING (b)
ORDER BY grid.b;


-- ---------------------------------------------------------------------
-- 9. Threshold sweep -- the precision/recall curve, one row per cut
-- ---------------------------------------------------------------------
-- The table the threshold visuals and the precision-priority DAX measures
-- read. Cumulative sums walk the buckets from the top score down, so each
-- row is "what a cut here would admit".
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_threshold_sweep AS
WITH totals AS (
    SELECT SUM(n_pos) AS total_pos, SUM(n_neg) AS total_neg,
           SUM(n_total) AS total_rows
    FROM vw_aml_score_buckets
),
cum AS (
    SELECT b,
           score_low AS threshold,
           SUM(n_pos) OVER (ORDER BY b DESC) AS tp,
           SUM(n_neg) OVER (ORDER BY b DESC) AS fp
    FROM vw_aml_score_buckets
)
SELECT cum.threshold,
       cum.tp,
       cum.fp,
       totals.total_pos - cum.tp                   AS fn,
       totals.total_neg - cum.fp                   AS tn,
       cum.tp + cum.fp                             AS n_alerts,
       cum.tp::double precision
         / NULLIF(cum.tp + cum.fp, 0)              AS "precision",
       cum.tp::double precision
         / NULLIF(totals.total_pos, 0)             AS recall,
       cum.fp::double precision
         / NULLIF(totals.total_neg, 0)             AS false_positive_rate,
       (cum.tp + cum.fp)::double precision
         / NULLIF(totals.total_rows, 0)            AS alert_rate,
       -- F1 is published only so a report can show what the precision-priority
       -- policy is giving up. It is never the basis for the chosen cut.
       CASE WHEN cum.tp > 0 THEN
           2.0 * cum.tp / (2.0 * cum.tp + cum.fp + (totals.total_pos - cum.tp))
       ELSE 0 END                                  AS f1
FROM cum CROSS JOIN totals
ORDER BY cum.threshold;


-- ---------------------------------------------------------------------
-- 10. Headline model metrics (macro-PR)
-- ---------------------------------------------------------------------
-- Macro-averaged PR-AUC: the mean of average precision for each class.
-- This project's metric of record -- not accuracy, not plain ROC-AUC --
-- because the positive class is a fraction of a percent of the book.
--
-- Average precision (step-wise sum of precision x recall gained) is used
-- rather than trapezoidal area, matching sklearn's estimator.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_model_metrics AS
WITH totals AS (
    SELECT SUM(n_pos) AS total_pos, SUM(n_neg) AS total_neg,
           SUM(n_total) AS total_rows
    FROM vw_aml_score_buckets
),
pos_curve AS (
    SELECT tp::double precision / NULLIF(tp + fp, 0)        AS "precision",
           tp::double precision / NULLIF(total_pos, 0)      AS recall
    FROM (
        SELECT b,
               SUM(n_pos) OVER (ORDER BY b DESC) AS tp,
               SUM(n_neg) OVER (ORDER BY b DESC) AS fp
        FROM vw_aml_score_buckets
    ) c CROSS JOIN totals
),
-- The negative class is the same curve with the label and the score
-- direction both flipped: walk the buckets upward and swap pos/neg.
neg_curve AS (
    SELECT tp::double precision / NULLIF(tp + fp, 0)        AS "precision",
           tp::double precision / NULLIF(total_neg, 0)      AS recall
    FROM (
        SELECT b,
               SUM(n_neg) OVER (ORDER BY b ASC) AS tp,
               SUM(n_pos) OVER (ORDER BY b ASC) AS fp
        FROM vw_aml_score_buckets
    ) c CROSS JOIN totals
),
ap AS (
    SELECT
      (SELECT SUM("precision" * recall_gain) FROM (
           SELECT "precision",
                  recall - COALESCE(LAG(recall) OVER (ORDER BY recall), 0) AS recall_gain
           FROM pos_curve) s)                                AS pr_auc_positive,
      (SELECT SUM("precision" * recall_gain) FROM (
           SELECT "precision",
                  recall - COALESCE(LAG(recall) OVER (ORDER BY recall), 0) AS recall_gain
           FROM neg_curve) s)                                AS pr_auc_negative
)
SELECT ap.pr_auc_positive,
       ap.pr_auc_negative,
       (ap.pr_auc_positive + ap.pr_auc_negative) / 2.0       AS macro_pr_auc,
       totals.total_pos::double precision
         / NULLIF(totals.total_rows, 0)                      AS base_rate,
       totals.total_pos,
       totals.total_neg,
       totals.total_rows
FROM ap CROSS JOIN totals;


-- ---------------------------------------------------------------------
-- 11. Decile lift -- the capacity view
-- ---------------------------------------------------------------------
-- "If the team can work N cases a day, what hit rate do the top N scores
-- give us?" Deciles hold roughly equal row counts even though the score
-- histogram is very skewed.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_lift_deciles AS
WITH totals AS (
    SELECT SUM(n_total) AS total_rows, SUM(n_pos) AS total_pos
    FROM vw_aml_score_buckets
),
ranked AS (
    SELECT sb.b, sb.n_pos, sb.n_total,
           SUM(sb.n_total) OVER (ORDER BY sb.b DESC) AS cum_rows,
           t.total_rows, t.total_pos
    FROM vw_aml_score_buckets sb CROSS JOIN totals t
),
assigned AS (
    SELECT GREATEST(LEAST(CEIL(cum_rows::double precision
                               / NULLIF(total_rows, 0) * 10)::int, 10), 1) AS decile,
           n_pos, n_total, total_rows, total_pos
    FROM ranked
),
agg AS (
    SELECT decile,
           SUM(n_total) AS n_tx,
           SUM(n_pos)   AS n_positive,
           MAX(total_rows) AS total_rows,
           MAX(total_pos)  AS total_pos
    FROM assigned GROUP BY 1
)
SELECT decile,
       n_tx,
       n_positive,
       n_positive::double precision / NULLIF(n_tx, 0)                   AS hit_rate,
       (n_positive::double precision / NULLIF(n_tx, 0))
         / NULLIF(total_pos::double precision / NULLIF(total_rows, 0), 0) AS lift,
       SUM(n_positive) OVER (ORDER BY decile)::double precision
         / NULLIF(SUM(n_tx) OVER (ORDER BY decile), 0)                  AS cumulative_hit_rate
FROM agg
ORDER BY decile;


-- ---------------------------------------------------------------------
-- 12. Covariate-shift distributions
-- ---------------------------------------------------------------------
-- Two aligned histograms per numeric feature: a reference window (the
-- earlier 70% of steps) and a current window (the rest). Shared bin edges
-- are what make the windows comparable -- and what lets PSI be computed
-- from the result.
--
-- This is a model-health signal, read separately from any transaction-level
-- flag: it says the inputs moved, not that a transaction is wrong.
--
-- NOTE: this is the heaviest view here -- it fans each row out to five
-- feature rows. See the materialised-view section at the bottom if it is
-- slow on your data volume.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_feature_distribution AS
WITH split AS (
    SELECT MAX(step) * 0.7 AS step_split FROM transactions
),
long AS (
    SELECT f.feature, f.value, t.step
    FROM transactions t
    CROSS JOIN LATERAL (VALUES
        ('amount',          t.amount),
        ('oldbalance_org',  t.oldbalance_org),
        ('newbalance_orig', t.newbalance_orig),
        ('oldbalance_dest', t.oldbalance_dest),
        ('newbalance_dest', t.newbalance_dest)
    ) AS f(feature, value)
),
bounds AS (
    SELECT feature, MIN(value) AS lo,
           GREATEST(MAX(value), MIN(value) + 1) AS hi
    FROM long GROUP BY 1
),
binned AS (
    SELECT long.feature,
           WIDTH_BUCKET(LEAST(GREATEST(long.value, bounds.lo), bounds.hi - 1e-9),
                        bounds.lo, bounds.hi, 20)                          AS bin,
           bounds.lo, bounds.hi,
           COUNT(*) FILTER (WHERE long.step <  split.step_split)           AS reference_n,
           COUNT(*) FILTER (WHERE long.step >= split.step_split)           AS current_n
    FROM long
    JOIN bounds USING (feature)
    CROSS JOIN split
    GROUP BY 1, 2, 3, 4
),
totals AS (
    SELECT feature, SUM(reference_n) AS ref_total, SUM(current_n) AS cur_total
    FROM binned GROUP BY 1
)
SELECT binned.feature,
       binned.bin,
       binned.lo + (binned.bin - 1) * (binned.hi - binned.lo) / 20.0 AS bin_low,
       binned.lo + binned.bin       * (binned.hi - binned.lo) / 20.0 AS bin_high,
       binned.reference_n,
       binned.current_n,
       binned.reference_n::double precision
         / NULLIF(totals.ref_total, 0)                               AS reference_share,
       binned.current_n::double precision
         / NULLIF(totals.cur_total, 0)                               AS current_share
FROM binned JOIN totals USING (feature)
ORDER BY binned.feature, binned.bin;


-- ---------------------------------------------------------------------
-- 13. PSI per feature, with a status band
-- ---------------------------------------------------------------------
-- Population Stability Index: the at-a-glance drift magnitude.
-- Complementary to the KS test in `src/inference/covariate_shift.py` --
-- KS gives a p-value, PSI gives a magnitude that stays meaningful at this
-- sample size (where KS flags everything).
--
-- Bands: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant.
-- The band ships as a word, so a report never leans on colour alone.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_aml_feature_psi AS
WITH shares AS (
    SELECT feature,
           GREATEST(COALESCE(reference_share, 0), 1e-6) AS ref_share,
           GREATEST(COALESCE(current_share, 0), 1e-6)   AS cur_share
    FROM vw_aml_feature_distribution
),
psi AS (
    SELECT feature,
           SUM((cur_share - ref_share) * LN(cur_share / ref_share)) AS psi
    FROM shares GROUP BY 1
)
SELECT feature,
       psi,
       CASE WHEN psi IS NULL     THEN 'unknown'
            WHEN psi < 0.10      THEN 'stable'
            WHEN psi < 0.25      THEN 'moderate shift'
            ELSE                      'significant shift'
       END AS shift_band
FROM psi
ORDER BY psi DESC;


-- =====================================================================
--  Optional: materialise the heavy views
-- =====================================================================
--  Views 8-13 rescan `transactions` / `model_outputs` on every refresh.
--  If Power BI refresh is slow, convert the expensive ones to materialised
--  views and refresh them on your own schedule. The view names stay the
--  same, so nothing in the report changes.
--
--      DROP VIEW IF EXISTS vw_aml_feature_psi;
--      DROP VIEW IF EXISTS vw_aml_feature_distribution;
--      CREATE MATERIALIZED VIEW vw_aml_feature_distribution AS <body above>;
--      CREATE VIEW vw_aml_feature_psi AS <body above>;
--      REFRESH MATERIALIZED VIEW vw_aml_feature_distribution;
--
--  Same pattern applies to vw_aml_score_buckets, which views 9-11 all read.
-- =====================================================================


-- =====================================================================
--  Sanity check -- run after creating the views
-- =====================================================================
--  SELECT 'transactions'  AS view_name, COUNT(*) FROM vw_aml_transactions
--  UNION ALL SELECT 'daily',      COUNT(*) FROM vw_aml_daily_metrics
--  UNION ALL SELECT 'channel',    COUNT(*) FROM vw_aml_channel_metrics
--  UNION ALL SELECT 'patterns',   COUNT(*) FROM vw_aml_pattern_metrics
--  UNION ALL SELECT 'buckets',    COUNT(*) FROM vw_aml_score_buckets
--  UNION ALL SELECT 'sweep',      COUNT(*) FROM vw_aml_threshold_sweep
--  UNION ALL SELECT 'lift',       COUNT(*) FROM vw_aml_lift_deciles
--  UNION ALL SELECT 'shap',       COUNT(*) FROM vw_aml_shap_long
--  UNION ALL SELECT 'psi',        COUNT(*) FROM vw_aml_feature_psi;
--
--  SELECT * FROM vw_aml_model_metrics;
-- =====================================================================
