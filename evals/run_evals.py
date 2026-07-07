"""Agent-facing evals for groundcheck: (a) does an agent pick the right tool for a
described task, and (b) does the hallucination-detection engine catch planted
hallucinations without over-flagging clean answers.

Requires ANTHROPIC_API_KEY (calls the real Anthropic API directly via
ApiKeyJudge -- this is an offline batch script, not run through the MCP
protocol, so there's no client to sample from).

Usage:
    ANTHROPIC_API_KEY=sk-... uv run python evals/run_evals.py --label baseline
    # tune tool docstrings in src/groundcheck/server.py, then:
    ANTHROPIC_API_KEY=sk-... uv run python evals/run_evals.py --label tuned

Writes evals/raw_<label>.json and prints a summary. Compare two label runs by
hand (or eyeball the JSON diff) and record the before/after in RESULTS.md --
this script does not write RESULTS.md for you, since the "why" behind a
descriptions change belongs in prose, not a template.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from groundcheck.engine.claims import detect_hallucinations  # noqa: E402
from groundcheck.judge.sampler import ApiKeyJudge, api_key_judge_from_env  # noqa: E402
from groundcheck.metrics import token_overlap  # noqa: E402
from groundcheck.schemas import Source  # noqa: E402
from groundcheck.server import mcp  # noqa: E402

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
OVERLAP_MATCH_THRESHOLD = 0.3

TOOL_SELECTION_SYSTEM = (
    "You are an AI agent deciding which tool to call for a task. You will be given "
    "a task description and a list of available tools (name and description). "
    "Respond with ONLY the exact name of the single best tool, nothing else -- no "
    "punctuation, no explanation."
)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


async def _tool_catalog_text() -> str:
    tools = await mcp.list_tools()
    return "\n\n".join(f"- {t.name}: {(t.description or '').strip()}" for t in tools)


async def eval_tool_selection(judge: ApiKeyJudge) -> dict:
    cases = _load_jsonl(SCENARIOS_DIR / "tool_selection.jsonl")
    catalog = await _tool_catalog_text()
    results = []
    for case in cases:
        prompt = (
            f"TASK: {case['task_description']}\n\nAVAILABLE TOOLS:\n{catalog}\n\n"
            "Which tool should the agent call?"
        )
        raw = await judge.complete(system=TOOL_SELECTION_SYSTEM, prompt=prompt, max_tokens=50)
        picked = raw.strip().strip("`").strip()
        correct = picked == case["expected_tool"]
        results.append(
            {
                "id": case["id"],
                "task_description": case["task_description"],
                "expected": case["expected_tool"],
                "picked": picked,
                "correct": correct,
            }
        )
    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0.0
    return {"accuracy": accuracy, "cases": results}


def _flagged_matches_any(flagged_span: str, expected_spans: list[str]) -> bool:
    return any(
        token_overlap(flagged_span, expected) >= OVERLAP_MATCH_THRESHOLD
        for expected in expected_spans
    )


async def eval_hallucination_detection(judge: ApiKeyJudge) -> dict:
    cases = _load_jsonl(SCENARIOS_DIR / "hallucination_cases.jsonl")
    results = []
    true_positives = false_negatives = true_negatives = false_positives = 0
    for case in cases:
        sources = [Source(**s) for s in case["sources"]]
        result = await detect_hallucinations(judge, case["answer"], sources)
        flagged = [h.answer_span for h in result.hallucinations]
        expected = case["planted_hallucination_spans"]

        if expected:
            case_correct = any(_flagged_matches_any(span, expected) for span in flagged)
            if case_correct:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            case_correct = not flagged
            if case_correct:
                true_negatives += 1
            else:
                false_positives += 1

        results.append(
            {
                "id": case["id"],
                "expected_hallucinations": expected,
                "flagged_spans": flagged,
                "correct": case_correct,
            }
        )

    total = len(cases)
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives)
        else 0.0
    )
    accuracy = (true_positives + true_negatives) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "recall_on_planted_hallucinations": recall,
        "specificity_on_clean_answers": specificity,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "cases": results,
    }


async def main_async(label: str) -> None:
    judge = api_key_judge_from_env()
    if judge is None:
        print("ANTHROPIC_API_KEY is not set. Evals need a real judge to score.", file=sys.stderr)
        raise SystemExit(1)

    tool_selection = await eval_tool_selection(judge)
    hallucinations = await eval_hallucination_detection(judge)

    output = {
        "label": label,
        "tool_selection": tool_selection,
        "hallucination_detection": hallucinations,
    }
    out_path = Path(__file__).parent / f"raw_{label}.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"=== groundcheck evals ({label}) ===")
    print(
        f"Tool selection accuracy: {tool_selection['accuracy']:.0%} "
        f"({sum(c['correct'] for c in tool_selection['cases'])}/{len(tool_selection['cases'])})"
    )
    recall = hallucinations["recall_on_planted_hallucinations"]
    specificity = hallucinations["specificity_on_clean_answers"]
    print(f"Hallucination detection accuracy: {hallucinations['accuracy']:.0%}")
    print(f"  recall on planted hallucinations: {recall:.0%}")
    print(f"  specificity on clean answers: {specificity:.0%}")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="run", help="Label for this run, e.g. baseline or tuned")
    args = parser.parse_args()
    asyncio.run(main_async(args.label))


if __name__ == "__main__":
    main()
