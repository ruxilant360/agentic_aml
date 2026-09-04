

This file gives Claude Code context on this repository. Read it before making changes.

## Project Overview

An Anti-Money Laundering (AML) fraud detection system combining:
- An **XGBoost model** trained via RandomizedSearch (tunable hyperparameters, passed as function args) that flags suspicious transactions. the training pipeline must also contain functions to split the dataset into train and test and to encode the categorical variables. the dataset is already cleaned all you need is just the pipeline to encode function to define model and function to train model. btw, r
- A **LangGraph agent** that reacts to model outputs — explains flags via SHAP, writes results to a SQL database, generates policy-grounded reports (RAG), and raises alerts (fraud flag, covariate shift)
- A **LangChain agent** that lets a user query the SQL database via natural language from the terminal
- An **eval suite** covering both the model and the agent(s), including latency (golden evals) and tool-calling accuracy

## Repository Structure

```
src/
  training/         # XGBoost training, RandomizeSearch, hyperparameter args
  inference/         # Inference pipeline, covariate shift detection
  tools/             # LangGraph + LangChain tool function definitions
  agentic/           # LangGraph agent graph, nodes, routing logic
  rag/               # Policy document retrieval for report generation
eval/
  model/             # XGBoost eval: macro-PR, threshold selection
  agent/             # Agent eval: golden evals (latency), SQL tool-calling accuracy, fallback behavior tests
```

## Model: XGBoost Fraud Classifier

- Trained with **RandomizedCV**; hyperparameters (e.g. `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `min_child_weight`) are exposed as **function arguments**, not hardcoded — training functions should accept a params dict/kwargs and pass through to RandomizedSearch, not bake in fixed values. Note full data lives in PosgreSQL db not pandas so training logic must contain extraction step from SQL DB to pandas.  
- **Evaluation metric: macro-averaged Precision-Recall (macro-PR)**, not accuracy or plain AUC — this is an imbalanced fraud problem.
- **Threshold selection prioritizes precision over recall.** We'd rather miss some fraud (lower recall) than flood the agent/analyst with false positives (low precision). When writing threshold-selection code, default to maximizing precision subject to a minimum acceptable recall floor — don't default to F1-balanced thresholds unless explicitly asked.
- The model's output to the agent is a **binary/score flag only** — it does not itself label a transaction "suspicious" in any user-facing text. That framing (suspicious vs. not) is downstream, produced by the agent via SHAP explanation, never by the model directly.

## Agentic Layer

### LangGraph agent — tools

1. **Write to PostgreSQL database** — persists transaction to the DB. I have the db created with two different tables, one for transactions and one for model output (score, SHAP values per feature, flag)
2. **Generate report (RAG)** — triggered when a transaction is flagged. Retrieves relevant policy documents and drafts a report. Uses **SHAP values to explain *why*** a transaction was flagged — the agent never says "suspicious" verbatim; it surfaces the SHAP-driven reasoning and lets the analyst draw that conclusion, then alerts the user.
3. **Alert on covariate shift** — triggered by the inference pipeline when input feature distributions drift from training distributions. Separate concern from fraud alerting; this is a model-health signal, not a transaction-level flag.

### LangChain agent — tools

1. **NLP-to-SQL query tool** — takes natural language input from the user's terminal, queries the database, returns results. This agent is read-only / query-focused, distinct from the LangGraph agent's write/report/alert responsibilities. Keep these two agents' tool sets separate — don't merge them into one agent with five tools.

### Conventions

- Tool functions live in `src/tools/`, one file per tool (or per logical group) — not inlined in the graph-definition file.
- Every tool function needs a clear docstring (LangChain/LangGraph use these for tool descriptions passed to the LLM) — treat the docstring as user-facing, not just internal documentation.
- When a tool call can fail (DB write, RAG retrieval, external call), catch the exception and return an error string in a `ToolMessage`-compatible form rather than raising — routing/retry logic lives in the agent's system prompt and graph structure, not in unhandled exceptions.
- Fraud/flag alerting logic and covariate-shift alerting logic should be **separate tools**, not overloaded into one "alert" tool — they have different triggers, different urgency, and different downstream handling.

## Eval Suite

### `eval/model/`
- Macro-PR curve computation and threshold selection tests.
- Verify chosen threshold meets the precision-priority policy (precision maximized subject to a recall floor), not just "best F1."

### `eval/agent/`
- **Golden evals**: fixed set of representative user prompts, with response time measured per prompt — used to track latency regressions.
- **SQL tool-calling accuracy**: fixed set of NL queries with known-correct SQL/results; track how often the agent calls the DB tool correctly (right query, right args) vs. how often it gets it wrong.
- **Fallback/callback behavior**: tests for what the LangGraph agent does when the fraud threshold is *not* reached — i.e., verifying the default/no-op path (no report, no alert, correct DB write) behaves as expected, not just testing the "happy path" where fraud is flagged.

## Things to Avoid

- Don't hardcode XGBoost hyperparameters inside training functions — they must remain arguments.
- Don't let the model or the agent output the literal word "suspicious" (or similar definitive fraud language) as a label — output should be SHAP-driven explanation; the alert is the signal, not an accusation baked into text.
- Don't default threshold selection to F1/accuracy-balanced logic — precision is prioritized per the policy above.
- Don't combine the LangGraph agent's tools and the LangChain query agent's tool into a single agent.

Leave everything else as is, do not run the training or inference code just write it all 
