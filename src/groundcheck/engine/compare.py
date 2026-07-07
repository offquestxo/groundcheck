"""Pairwise answer comparison with mandatory position-bias mitigation: the judge
sees (A,B) and (B,A) in separate calls, and any disagreement between the two
orderings collapses that verdict to "tie/uncertain" rather than picking a side.
"""

from __future__ import annotations

from groundcheck.judge.parsing import extract_json
from groundcheck.judge.prompts import COMPARE_PROMPT, COMPARE_SYSTEM
from groundcheck.judge.sampler import Judge
from groundcheck.schemas import CompareResult, CriterionVerdict, Source


def _format_sources(sources: list[Source] | None) -> str:
    if not sources:
        return ""
    block = "\n\n".join(f"[{s.id}]\n{s.text}" for s in sources)
    return f"SOURCES:\n{block}\n\n"


async def _judge_one_ordering(
    judge: Judge,
    query: str,
    first_label: str,
    first_answer: str,
    second_label: str,
    second_answer: str,
    sources: list[Source] | None,
    criteria: list[str],
) -> dict:
    raw = await judge.complete(
        system=COMPARE_SYSTEM,
        prompt=COMPARE_PROMPT.format(
            query=query,
            first_label=first_label,
            first_answer=first_answer,
            second_label=second_label,
            second_answer=second_answer,
            sources_block=_format_sources(sources),
            criteria=", ".join(criteria),
        ),
        max_tokens=2048,
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Judge returned non-object compare verdict: {raw!r}")
    return parsed


async def compare_answers(
    judge: Judge,
    query: str,
    answer_a: str,
    answer_b: str,
    sources: list[Source] | None,
    criteria: list[str],
) -> CompareResult:
    forward = await _judge_one_ordering(
        judge, query, "a", answer_a, "b", answer_b, sources, criteria
    )
    backward = await _judge_one_ordering(
        judge, query, "b", answer_b, "a", answer_a, sources, criteria
    )

    forward_by_criterion = {c["criterion"]: c for c in forward["per_criterion"]}
    backward_by_criterion = {c["criterion"]: c for c in backward["per_criterion"]}

    criterion_verdicts = []
    for criterion in criteria:
        fwd = forward_by_criterion.get(criterion)
        bwd = backward_by_criterion.get(criterion)
        if fwd is None or bwd is None:
            criterion_verdicts.append(
                CriterionVerdict(
                    criterion=criterion,
                    winner="tie",
                    rationale="Judge did not return a verdict for this criterion.",
                )
            )
            continue
        if fwd["winner"] == bwd["winner"]:
            criterion_verdicts.append(
                CriterionVerdict(
                    criterion=criterion, winner=fwd["winner"], rationale=fwd["rationale"]
                )
            )
        else:
            criterion_verdicts.append(
                CriterionVerdict(
                    criterion=criterion,
                    winner="tie",
                    rationale=(
                        f"Verdict changed with presentation order (forward: {fwd['winner']}, "
                        f"backward: {bwd['winner']}) -- treating as a tie to avoid position bias."
                    ),
                )
            )

    if forward["overall_winner"] == backward["overall_winner"]:
        overall_winner = forward["overall_winner"]
        rationale = forward["rationale"]
    else:
        overall_winner = "tie/uncertain"
        rationale = (
            f"Overall verdict disagreed between presentation orders (forward: "
            f"{forward['overall_winner']}, backward: {backward['overall_winner']}); "
            "reporting as uncertain rather than picking a side."
        )

    return CompareResult(winner=overall_winner, criteria=criterion_verdicts, rationale=rationale)
