# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `groundcheck_evaluate_faithfulness`: claim-level faithfulness scoring against sources.
- `groundcheck_detect_hallucinations`: tight, fix-it-loop-oriented hallucination report.
- `groundcheck_evaluate_retrieval`: precision/recall/MRR/NDCG@k, gold-label (Mode A) or LLM-graded (Mode B).
- `groundcheck_compare`: pairwise answer comparison with mandatory position-bias mitigation.
- `groundcheck_run_suite` + `groundcheck_get_report`: batch evaluation with a persisted JSON report store.
- `groundcheck://reports` and `groundcheck://reports/{report_id}` resources.
- `audit_rag_response` prompt template.
- Zero-key judge via MCP sampling, with an `ANTHROPIC_API_KEY` fallback and actionable errors when neither is available.
- Stdio and Streamable HTTP transports from the same app; optional bearer-token auth for HTTP.
- Deterministic metrics module (`metrics.py`) with exhaustive edge-case unit tests.
- Eval harness (`evals/`) for tool-selection and hallucination-detection accuracy.
