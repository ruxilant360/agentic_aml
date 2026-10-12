# Advanced dashboards (fraud-model analysis)

> Looking for the plain transaction dashboard? That's [`README.md`](README.md)
> — this file is the deeper fraud-model stack (SHAP, drift, precision/recall)
> built on top of it. Most people want the other one.

Two dashboards over the same two PostgreSQL tables (`transactions`,
`model_outputs`):

| | What it is | Where |
|---|---|---|
| **Streamlit** | Interactive app, six tabs, runs locally | `streamlit_app.py` |
| **Power BI** | SQL views + DAX measures + theme you assemble into a `.pbix` | `powerbi/` |

Both read the same metrics and follow the same two project rules:

- **Macro-PR is the metric of record**, and **threshold selection maximises
  precision subject to a recall floor** — never best-F1. The F1 numbers appear
  only as a labelled contrast, so the cost of the policy is visible.
- A flag is a **routing signal**, and SHAP is the **evidence** behind it.
  Neither dashboard labels a transaction; the conclusion stays the analyst's.

```
dashboards/
  streamlit_app.py     six-tab Streamlit app
  data_access.py        SQL/CSV loaders, runtime column-name resolution
  metrics.py            PR curves, threshold policy, PSI, lift (pure functions)
  theme.py              validated light/dark palette + Plotly chrome
  powerbi/
    01_views.sql        13 read-only views — run this first
    02_measures.dax     ~60 measures to paste into Power BI
    aml_theme.json      report theme, same palette as Streamlit
```

---

## 1. Streamlit — what to run

```powershell
cd C:\Users\USER\Downloads\Projects\AgenticAMLproject
.\venv\Scripts\Activate.ps1

pip install plotly==6.0.1 streamlit==1.41.1      # the only two new deps

streamlit run dashboards/streamlit_app.py
```

It opens at <http://localhost:8501>. Stop it with `Ctrl+C`.

Both packages are already added to `requirements.txt`, so
`pip install -r requirements.txt` also works — it's just slower.

### If the database isn't up

Pick **CSV sample** in the sidebar. It reads `notebooks/data.csv` directly, so
the Portfolio, Behaviour and Model health tabs work with no database at all.
The Review queue, Model performance and SHAP tabs stay inert there — scores,
flags and SHAP values only exist in `model_outputs`.

### The six tabs

| Tab | Reads | Key metrics |
|---|---|---|
| **Portfolio** | `transactions` | volume & value per day, hour-of-day profile, channel mix, order-of-magnitude amount bands, positive-label rate at every cut |
| **Review queue** | both | queue size and precision as a function of the score cut, flag rate per day, top 50 cases |
| **Model performance** | both | macro-PR AUC, PR curve, per-class score distributions, the precision-priority cut, confusion matrix, decile lift |
| **SHAP drivers** | `model_outputs` | mean \|SHAP\| (how much each feature moves the score), mean signed SHAP (which way), single-case breakdown |
| **Behaviour** | `transactions` | six account-behaviour patterns with coverage *and* hit rate, destination fan-in |
| **Model health** | `transactions` | PSI per feature between an early reference window and a recent one, distribution overlay, score drift |

### Sidebar controls

- **Queue score cut** — the cut that defines the review queue and the "flagged
  only" SHAP scope.
- **Recall floor** — the constraint in the threshold policy. Raise it and
  precision falls; the app shows the chosen cut's precision, recall and queue
  size side by side so the trade is visible. (Both implementations also guard
  against a floor no cut can reach. In practice that can't happen — recall
  reaches 1.0 at a threshold of 0 — so treat it as a safety net, not a
  feature you'll see.)
- **Chart palette** — Light / Dark. Each mode has its own validated palette
  steps, not an automatic flip. To make the app chrome match dark too, set
  `base = "dark"` in `.streamlit/config.toml` (the file lists the four colours).

### Performance

Every metric is a SQL aggregate, so the millions of rows in `transactions`
never land in pandas — the PR curve comes from a 200-bin server-side
histogram, not from pulling score/label pairs. Results are cached for 10
minutes; press `R` in the browser to force a refresh.

---

## 2. Power BI — what to run

### Step 1 — create the views

```powershell
psql $env:DATABASE_URL -f dashboards/powerbi/01_views.sql
```

No `psql`? Paste the file into pgAdmin's Query Tool, or DBeaver, and execute.
It only creates views; it never writes to, alters or drops your two tables.

**Before you run it**, confirm your column naming, because this repo
disagrees with itself — `src/db/schema.py` declares camelCase (`transID`,
`oldbalanceOrg`, and a typo'd `newbalancDest`) while
`src/training/data_extraction.py` and `src/tools/sql_query_tool.py` both query
snake_case:

```sql
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name IN ('transactions', 'model_outputs')
ORDER BY table_name, ordinal_position;
```

The SQL file is written in snake_case. If your database reports camelCase,
rename the identifiers in the file, quoting the mixed-case ones (`t."transID"`).
The Streamlit app resolves this automatically at runtime; plain SQL can't.

Sanity-check afterwards with the commented query at the bottom of the file.

### Step 2 — apply the theme

Power BI Desktop → **View → Themes → Browse for themes** →
`dashboards/powerbi/aml_theme.json`.

### Step 3 — import the views

**Home → Get Data → PostgreSQL database**, then select these 12 views. (The
script creates 13; the odd one out is `vw_aml_score_buckets`, a 101-row base
view that the curve views read in SQL. Importing it does no harm, but nothing
in the report binds to it.)

```
vw_aml_transactions          vw_aml_threshold_sweep
vw_aml_model_outputs         vw_aml_model_metrics
vw_aml_shap_long             vw_aml_lift_deciles
vw_aml_daily_metrics         vw_aml_feature_distribution
vw_aml_channel_metrics       vw_aml_feature_psi
vw_aml_pattern_metrics       vw_aml_dest_fan_in
```

Use **Import** for all of them except `vw_aml_transactions` and
`vw_aml_shap_long` — those two are row-level and large, so **DirectQuery**
keeps the file small. Everything else is already aggregated to at most a few
hundred rows.

### Step 4 — relationships

**Model view**. Only two, both one-to-many:

```
vw_aml_transactions[transaction_id]  1 → *  vw_aml_model_outputs[transaction_id]
vw_aml_model_outputs[transaction_id] 1 → *  vw_aml_shap_long[transaction_id]
```

Leave every other view unrelated. They carry pre-computed rates; relating them
lets a slicer re-filter the numerator without the denominator, which reads
wrong rather than failing.

Do **not** mark anything as a date table. `step` is hours since the simulation
started, not a calendar — `sim_day` is a plain numeric axis, and DAX time
intelligence over it returns nonsense.

### Step 5 — the recall-floor parameter

**Modeling → New parameter → Numeric range**: name `Recall Floor`, min `0.05`,
max `0.95`, increment `0.05`, default `0.5`. Power BI generates a table, a
column, and a `[Recall Floor Value]` measure — that last one is what the
threshold measures read.

### Step 6 — the measures

Open `02_measures.dax` and paste each measure in via **Modeling → New measure**.
DAX has no bulk-import format, so this is a copy-paste library. The fast path
is sections 1–6 and 12 (about 25 measures) — that covers every page below;
add the rest as you need them.

### Suggested pages

| Page | Visuals |
|---|---|
| **Overview** | Cards: `Transactions`, `Total Value`, `Positive Label Rate`, `Macro-PR AUC`. Line: `Daily Transactions` by `sim_day`. Bar: `Channel Transactions` by `channel`. |
| **Model** | Cards: `Policy Summary`, `Macro-PR AUC`. Line: `precision` by `recall` from `vw_aml_threshold_sweep`. Line: `Cases Queued at Cut` by `threshold`. Matrix: the four `Policy True/False …` measures. Text box: `Policy Trade-off`. |
| **Queue** | Slicer on `vw_aml_threshold_sweep[threshold]`. Cards: `Cases Queued at Cut`, `Precision at Cut`, `Recall at Cut`. Table from `vw_aml_model_outputs` + `vw_aml_transactions`. |
| **Explanations** | Bar: `Mean Abs SHAP` by `feature`, sorted descending. Bar: `Mean SHAP` by `feature` with a diverging conditional format around 0. Cards: `Top Driver`, `Top Driver Direction`. |
| **Behaviour** | Bar: `Pattern Coverage` by `pattern`. Bar: `Pattern Hit Rate` by `pattern`. Table: `vw_aml_dest_fan_in`. |
| **Health** | Card: `Drift Status`. Bar: `Feature PSI` by `feature`, conditional-formatted by `Drift Colour`. Line: `reference_share` and `current_share` by `bin_low` from `vw_aml_feature_distribution`, sliced by `feature`. |

### Chart conventions worth keeping

The theme supplies eight categorical colours in a fixed order — use them in
order and never generate a ninth. Three things it won't stop you doing, which
are worth not doing:

- **No dual-axis charts.** Count and value have different units; give them two
  visuals. The theme can't enforce this.
- **Cap scatter/bubble visuals at three series.** Slots 4+ stop being reliably
  distinguishable at a glance in those forms; bar and line visuals are fine to
  five.
- **One colour per bar chart** unless the colour means identity. Colouring bars
  darker-where-bigger just re-encodes the bar length.

---

## 3. Metrics reference

Everything either dashboard shows, and where it's defined:

| Metric | Definition | Defined in |
|---|---|---|
| Positive-label rate | positives ÷ **labelled** rows (`is_fraud` is nullable; unlabelled rows are excluded, not counted as negative) | `kpi_summary`, `[Positive Label Rate]` |
| Macro-PR AUC | mean of average precision for each class | `metrics.macro_pr_auc`, `vw_aml_model_metrics` |
| Policy threshold | highest cut with max precision among cuts clearing the recall floor | `metrics.select_threshold_by_precision`, `[Policy Threshold]` |
| Decile lift | hit rate in a score decile ÷ overall base rate | `metrics.lift_table`, `vw_aml_lift_deciles` |
| Mean \|SHAP\| | mean absolute contribution per feature — ranks influence | `shap_global`, `[Mean Abs SHAP]` |
| Mean signed SHAP | mean contribution per feature — shows direction | `shap_global`, `[Mean SHAP]` |
| PSI | Σ (cur% − ref%) · ln(cur% ÷ ref%) over 20 shared bins | `metrics.psi`, `vw_aml_feature_psi` |
| Ledger identity | expected post-balance is `old − amount`, except `CASH_IN` where it's `old + amount` | `typology_counts`, `vw_aml_transactions` |

Both implementations were checked against scikit-learn on synthetic labelled
scores: binned average precision lands within 0.006 of
`average_precision_score` at the bin resolution each uses, confusion-matrix
rows sum exactly to the true class totals, and the precision-priority policy
picks a visibly different cut from best-F1 (at a 50% floor on the test data:
precision 1.000 / recall 0.505, versus 0.932 / 0.735 for F1).

## 4. Interpretation notes

- **`step` is not a date.** It's hours since the simulation start, so "day" is
  `step / 24` and "hour of day" is `step % 24`. There is no calendar.
- **Behaviour patterns are descriptive, not accusatory.** Each ships with its
  coverage *and* its hit rate for a reason: "ledger identity broken" covers
  ~60% of the book at a ~0.008% hit rate — it's a data artefact, not a signal.
  "Originator balance fully drained" covers ~24% at ~0.56%, which is ~4× base
  rate and worth something.
- **PSI complements the KS test** in `src/inference/covariate_shift.py` rather
  than replacing it. KS returns a p-value that flags nearly everything at this
  sample size; PSI returns a magnitude that stays readable.
- **Covariate shift is a model-health signal**, handled separately from fraud
  flagging throughout — different trigger, different urgency, different
  downstream handling.
