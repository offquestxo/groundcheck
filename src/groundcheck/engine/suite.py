"""Batch eval runner + JSON report store.

`GROUNDCHECK_DATA_DIR` doubles as (a) the root for persisted reports
(`$GROUNDCHECK_DATA_DIR/reports/`, falling back to `~/.groundcheck/reports/`
when unset) and (b) the allowlisted root a `dataset_path` must resolve inside
of (falling back to the current working directory when unset). Path
traversal on `dataset_path` is the one real attack surface here -- it names a
file for us to read, and that input arrives from an LLM tool call -- so it's
handled explicitly rather than trusted.
"""

from __future__ import annotations

import os
from pathlib import Path

from groundcheck.engine.claims import evaluate_faithfulness
from groundcheck.engine.retrieval import evaluate_with_gold_labels
from groundcheck.judge.prompts import PROMPT_VERSION
from groundcheck.judge.sampler import Judge
from groundcheck.schemas import CaseResult, EvalCase, Report, SuiteSummary


class PathTraversalError(ValueError):
    def __init__(self, requested: Path, allowlist_dir: Path) -> None:
        self.requested = requested
        self.allowlist_dir = allowlist_dir
        super().__init__(
            f"dataset_path '{requested}' resolves outside the allowlisted directory "
            f"'{allowlist_dir}'. Set GROUNDCHECK_DATA_DIR to the directory containing "
            "your dataset, or move the file under it."
        )


class ReportNotFoundError(KeyError):
    def __init__(self, report_id: str, available: list[str]) -> None:
        self.report_id = report_id
        self.available = available
        super().__init__(
            f"No report with id '{report_id}'. Available report ids: {available or '(none yet)'}"
        )


def _data_dir_override() -> Path | None:
    raw = os.environ.get("GROUNDCHECK_DATA_DIR")
    return Path(raw).expanduser().resolve() if raw else None


def get_reports_dir() -> Path:
    override = _data_dir_override()
    base = override if override is not None else Path.home() / ".groundcheck"
    return base / "reports"


def get_dataset_allowlist_dir() -> Path:
    override = _data_dir_override()
    return override if override is not None else Path.cwd().resolve()


def resolve_dataset_path(dataset_path: str) -> Path:
    """Resolve `dataset_path` and reject anything outside the allowlisted dir."""
    allowlist_dir = get_dataset_allowlist_dir()
    candidate = Path(dataset_path)
    candidate = candidate if candidate.is_absolute() else allowlist_dir / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(allowlist_dir)
    except ValueError:
        raise PathTraversalError(candidate, allowlist_dir) from None
    return candidate


def load_cases_from_jsonl(path: Path) -> list[EvalCase]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate_json(line))
    return cases


def save_report(report: Report) -> None:
    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{report.report_id}.json"
    report_path.write_text(report.model_dump_json(indent=2))


def list_report_ids() -> list[str]:
    reports_dir = get_reports_dir()
    if not reports_dir.exists():
        return []
    return sorted(p.stem for p in reports_dir.glob("*.json"))


def load_report(report_id: str) -> Report:
    report_path = get_reports_dir() / f"{report_id}.json"
    if not report_path.exists():
        raise ReportNotFoundError(report_id, list_report_ids())
    return Report.model_validate_json(report_path.read_text())


async def run_suite(
    judge: Judge, cases: list[EvalCase], k_values: list[int] | None = None
) -> Report:
    """Run faithfulness (+ retrieval, when `relevant_ids` is given) per case,
    aggregate, persist a report, and return it."""
    k_values = k_values or [3, 5, 10]
    case_results: list[CaseResult] = []
    for case in cases:
        faithfulness = await evaluate_faithfulness(judge, case.answer, case.sources)
        retrieval = None
        if case.relevant_ids is not None:
            retrieval = evaluate_with_gold_labels(case.sources, case.relevant_ids, k_values)
        case_results.append(CaseResult(id=case.id, faithfulness=faithfulness, retrieval=retrieval))

    mean_faithfulness = (
        sum(r.faithfulness.score for r in case_results) / len(case_results) if case_results else 0.0
    )

    ndcg_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    for r in case_results:
        if r.retrieval is not None:
            for k, value in r.retrieval.ndcg.items():
                ndcg_by_k.setdefault(k, []).append(value)
    mean_ndcg = {k: (sum(vs) / len(vs) if vs else 0.0) for k, vs in ndcg_by_k.items()}

    worst_cases = [r.id for r in sorted(case_results, key=lambda r: r.faithfulness.score)[:5]]

    report = Report(
        prompt_version=PROMPT_VERSION,
        cases=case_results,
        summary=SuiteSummary(
            report_id="placeholder",
            case_count=len(cases),
            mean_faithfulness=mean_faithfulness,
            mean_ndcg=mean_ndcg,
            worst_cases=worst_cases,
        ),
    )
    report.summary.report_id = report.report_id
    save_report(report)
    return report
