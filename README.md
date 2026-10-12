# Agentic AML Project

An anti-money-laundering (AML) fraud detection system: an XGBoost model
flags suspicious transactions, a LangGraph agent reacts to those flags
(explains them with SHAP, writes results to PostgreSQL, drafts a
policy-grounded report, raises alerts), and a separate LangChain agent lets
you query the database in plain English from the terminal.

This README covers the whole project. If you just want to look at the data
in a browser, see **[`dashboards/README.md`](dashboards/README.md)** instead
— it's the fastest path to something you can actually see.

---

## Contents

- [What's in this repo](#whats-in-this-repo)
- [The dashboard](#the-dashboard-fastest-way-to-see-the-data)
- [Project structure](#project-structure)
- [Setup](#setup)
- [The model](#the-model-xgboost-fraud-classifier)
- [The agents](#the-agents)
- [Evals](#evals)
- [Project conventions](#project-conventions)
- [Current status](#current-status--what-actually-runs)

---

## What's in this repo

Three pieces, each independent:

1. **An XGBoost classifier** trained on transaction data pulled from
   PostgreSQL, tuned via `RandomizedSearchCV`, evaluated by macro-averaged
   precision-recall (not accuracy — this is a heavily imbalanced
   fraud-detection problem).
2. **A LangGraph agent** that reacts to the model's output: explains a flag
   with SHAP, writes the result to the database, drafts a report grounded in
   policy documents (RAG) when a transaction is flagged, and separately
   raises an alert if the model's input data has drifted from what it was
   trained on.
3. **A LangChain agent** that answers natural-language questions about the
   database from the terminal (read-only — it never writes anything).

Plus a `dashboards/` folder with a Streamlit app for just looking at the
numbers, and an `eval/` suite that checks the model and agents actually do
what they're supposed to.

---

## The dashboard (fastest way to see the data)

```powershell
.\venv\Scripts\Activate.ps1
pip install streamlit plotly
streamlit run dashboards/simple_dashboard.py
```

Opens at `http://localhost:8501`. Total transactions, total amount moved,
average transaction size, average account balances, and a breakdown by
transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER). Works
against either the live PostgreSQL database or `notebooks/data.csv` directly
— there's a toggle in the sidebar.

### Screenshot

<!--
  Save your screenshot to dashboards/screenshots/dashboard.png and it will
  appear right here automatically — no need to edit this file. Steps below.
-->
![Dashboard screenshot](dashboards/screenshots/dashboard.png)

*(Placeholder until you add one — see the next steps.)*

1. Run the dashboard (above) and open it at `http://localhost:8501`.
2. Take a screenshot:
   - **Windows:** `Win + Shift + S`, drag over the browser window, paste
     (`Ctrl + V`) into Paint or any image editor, save as PNG.
   - **Mac:** `Cmd + Shift + 4`, drag over the window — saves straight to
     your Desktop.
3. Save (or move) the file to this exact path, relative to the repo root:
   ```
   dashboards/screenshots/dashboard.png
   ```
   The folder already exists — just drop the file in. Nothing else to edit;
   the image tag above already points there.

Markdown can't have an image pasted directly into it — it only links to a
file sitting on disk — which is why the steps above are save-then-point
rather than paste-in.

Full details and a deeper SHAP/drift/precision-
recall version of the dashboard: **[`dashboards/README.md`](dashboards/README.md)**.

---

## Project structure

```
src/
  training/       XGBoost training: SQL extraction, encoding, train/test split,
                   RandomizedSearchCV (hyperparameters passed as args, never hardcoded)
  inference/       Scores a transaction (binary flag + score only), and detects
                   covariate shift (input drift vs. the training distribution)
  agentic/         LangGraph agent: graph definition, nodes, routing logic
  query_agent/     LangChain NL-to-SQL agent (separate from the LangGraph agent)
  tools/           Individual tool functions used by the two agents above
  rag/             Policy-document retrieval for report generation
  db/              SQLAlchemy models + connection handling for the two
                   PostgreSQL tables (transactions, model_outputs)
  cli/             Terminal entry point for the LangChain query agent
  config.py        All environment variables are read through here

eval/
  model/           Macro-PR and threshold-selection tests
  agent/           Golden evals (latency), SQL tool-calling accuracy,
                   fallback-path tests (what happens when nothing is flagged)

dashboards/
  simple_dashboard.py   Plain Streamlit dashboard (totals, averages, by-type)
  streamlit_app.py       Deeper fraud-model dashboard (SHAP, drift, PR curves)
  powerbi/                SQL views + DAX measures for a Power BI version

data/                (empty — data isn't checked into the repo; see Setup)
notebooks/            Exploratory notebooks + the raw CSV used as a fallback
                       data source when the database isn't running
```

---

## Setup

### 1. Python environment

```powershell
cd C:\Users\USER\Downloads\Projects\AgenticAMLproject
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A `venv/` already exists in this repo. If you're setting up fresh elsewhere,
create one first: `python -m venv venv`.

### 2. Environment variables

Copy `.env` (already present) and fill in / confirm these keys:

| Variable | What it's for |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (or set `PGUSER` / `PGPASSWORD` / `PGHOST` / `PGPORT` / `PGDATABASE` separately — see `src/config.py`) |
| `GROQ_API_KEY` | LLM calls for both agents (SHAP explanation, report drafting, NL-to-SQL) |
| `GROQ_MODEL_NAME` | Defaults to a sensible model if unset |
| `POLICY_DOCS_DIR` | Folder of policy documents the RAG report tool retrieves from |
| `FRAUD_SCORE_THRESHOLD` | Fallback score threshold if inference isn't given one explicitly — should come from the precision-priority threshold selection in `eval/model`, not left at an arbitrary default |

### 3. The database

Two tables are expected to **already exist** in PostgreSQL:
`transactions` and `model_outputs`. See `src/db/schema.py` for the expected
columns. Nothing in this codebase creates or alters tables — schema
ownership stays wherever the database was provisioned.

> **Heads up:** `src/db/schema.py` (camelCase: `transID`, `oldbalanceOrg`,
> ...) and the rest of the codebase (snake_case: `transaction_id`,
> `oldbalance_org`, ...) disagree on column naming. Check what your actual
> database has with:
> ```sql
> SELECT table_name, column_name FROM information_schema.columns
> WHERE table_name IN ('transactions', 'model_outputs');
> ```
> The dashboard code resolves this automatically at runtime; raw SQL (like
> the Power BI views) does not — see `dashboards/ADVANCED.md` if that
> matters to you.

### 4. The raw data

`notebooks/data.csv` (~5 million rows, the PaySim-style simulated
transaction dataset) is used as a no-database fallback for the dashboard and
for exploration. It is **not** the training source for the model — training
pulls from PostgreSQL (`src/training/data_extraction.py`), per project
convention.

---

## The model: XGBoost fraud classifier

- Trained via `RandomizedSearchCV` — every hyperparameter (`max_depth`,
  `n_estimators`, `learning_rate`, `subsample`, `min_child_weight`, ...) is
  passed in as an argument, never hardcoded in the training function.
- **Evaluation metric: macro-averaged precision-recall**, not accuracy and
  not plain ROC-AUC. The positive class (fraud) is a small fraction of a
  percent of the data — accuracy would look great on a model that flags
  nothing.
- **Threshold selection prioritizes precision over recall**: it maximizes
  precision subject to a minimum recall floor, not F1-balance. The policy is
  "miss some fraud rather than flood an analyst with false positives."
- The model's output is a **score and a binary flag only** — it never labels
  a transaction "suspicious" or similar in any text it produces. That
  framing is downstream, produced by the agent via SHAP explanation.

Relevant files: `src/training/data_extraction.py` (pull from Postgres),
`src/training/preprocessing.py` (encoding, train/test split),
`src/training/model.py` (model definition), `src/training/train.py`
(the search itself), `src/inference/pipeline.py` (scoring a transaction),
`src/inference/covariate_shift.py` (drift detection).

---

## The agents

### LangGraph agent (`src/agentic/`)

Reacts to the model's output. Two entry points:

- **A scored transaction** → explain it with SHAP → write it to the
  database → if flagged, draft a report (RAG over policy documents) and
  raise a fraud alert → otherwise, stop — no report, no alert, just the DB
  write (the "nothing happened" path, tested explicitly in
  `eval/agent/test_fallback_behavior.py`).
- **A covariate-shift signal** from the inference pipeline → raise a
  model-health alert. This is entirely separate from the fraud-flag path —
  different trigger, different urgency, no DB write, no report.

The agent's report never uses the word "suspicious" (or similar) as a
verdict — it surfaces the SHAP-driven reasoning and lets the analyst draw
that conclusion.

### LangChain query agent (`src/query_agent/`)

A single read-only tool: takes a natural-language question from the
terminal, generates SQL, queries the database, returns the result. Kept
completely separate from the LangGraph agent's tools (write/report/alert) —
intentionally not merged into one agent with five tools.

```powershell
python -m src.cli.query_cli
```

---

## Evals

```powershell
pytest eval/
```

- **`eval/model/`** — macro-PR computation, and that the chosen threshold
  actually follows the precision-priority policy (not just "best F1").
- **`eval/agent/`** — golden evals (fixed prompts, latency tracked per
  prompt), SQL tool-calling accuracy (does the query agent call the DB tool
  with the right query/args), and fallback-behavior tests (the no-flag path
  behaves correctly, not just the happy path).

---

## Project conventions

These are enforced throughout the codebase, not just suggested:

- **No hardcoded XGBoost hyperparameters** — they're always function
  arguments.
- **Never the literal word "suspicious"** (or similar definitive fraud
  language) in model or agent output. The alert is the signal; the SHAP
  explanation is the evidence; the conclusion is the analyst's.
- **Threshold selection is precision-first**, never F1/accuracy-balanced by
  default.
- **The two agents' tools are never combined** into one agent.
- Every tool function has a docstring written as user-facing text (LangChain
  / LangGraph pass these to the LLM as the tool's description).
- Tool calls that can fail (DB writes, RAG retrieval, external calls) catch
  their own exceptions and return an error string rather than raising —
  retry/routing logic lives in the graph structure, not in unhandled
  exceptions.

Full detail (including the agent-graph routing diagram in prose) lives in
**[`CLAUDE.md`](CLAUDE.md)**.

---

## Current status / what actually runs

This codebase has been built out module-by-module (training pipeline,
inference, agent graph, tools, evals) but **has not been run end-to-end** —
per the project's own working convention, code has been written and
reviewed, not executed, so there is currently no trained `model.json` you
should trust, and the agents have not been smoke-tested against a live
database. Before relying on any of this:

- Run `pytest eval/` and confirm the test suite actually passes against your
  environment.
- Run the training pipeline once end-to-end and sanity-check the resulting
  macro-PR score before trusting `model.json`.
- Confirm the database column-naming mismatch (see [Setup](#3-the-database))
  doesn't break raw SQL you write by hand.

The dashboards (`dashboards/`) are the one part of this repo that's been
actually executed and checked — see `dashboards/README.md` for what was
verified.
