# groundcheck evals

Two things get measured here, against the scenarios in `evals/scenarios/`:

1. **Tool selection** (`tool_selection.jsonl`, 12 cases): given a plain-English
   task description and the six tool name+description pairs (pulled live from
   the running server, so this always reflects the current docstrings), does
   an LLM agent pick the right tool?
2. **Hallucination detection** (`hallucination_cases.jsonl`, 12 cases: 7 with
   a planted hallucination, 5 clean): does `groundcheck_detect_hallucinations`
   catch the planted issue without over-flagging clean answers?

## Status: not yet run

This harness has not been executed against a real model yet -- the sandbox
this was built in has no `ANTHROPIC_API_KEY`. Numbers below are placeholders
to fill in after running for real:

```bash
ANTHROPIC_API_KEY=sk-... uv run python evals/run_evals.py --label baseline
```

Then tune the tool docstrings in `src/groundcheck/server.py` (and/or the judge
prompts in `src/groundcheck/judge/prompts.py`) based on which cases failed,
and re-run:

```bash
ANTHROPIC_API_KEY=sk-... uv run python evals/run_evals.py --label tuned
```

Each run writes `evals/raw_<label>.json` with full per-case detail. Fill in
the table below from those two files.

## Baseline vs tuned

| Metric | Baseline | Tuned | Delta |
|---|---|---|---|
| Tool selection accuracy | — | — | — |
| Hallucination recall (planted) | — | — | — |
| Hallucination specificity (clean) | — | — | — |

## Analysis

_Fill in after running: which cases failed at baseline, what description/prompt
change fixed them, and why. This is the actual artifact -- the before/after
numbers alone don't explain the engineering judgment behind the fix._
