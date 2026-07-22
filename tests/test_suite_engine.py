import json

import pytest

from groundcheck.engine.suite import (
    PathTraversalError,
    ReportNotFoundError,
    list_report_ids,
    load_cases_from_jsonl,
    load_report,
    resolve_dataset_path,
    run_suite,
    save_report,
)
from groundcheck.schemas import (
    EvalCase,
    Report,
    Source,
    SuiteSummary,
)
from tests.fakes import FakeJudge


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_DATA_DIR", str(tmp_path))
    return tmp_path


class TestResolveDatasetPath:
    def test_relative_path_inside_allowlist(self, tmp_path):
        (tmp_path / "data.jsonl").write_text("{}\n")
        resolved = resolve_dataset_path("data.jsonl")
        assert resolved == (tmp_path / "data.jsonl").resolve()

    def test_absolute_path_inside_allowlist(self, tmp_path):
        target = tmp_path / "nested" / "data.jsonl"
        target.parent.mkdir()
        target.write_text("{}\n")
        resolved = resolve_dataset_path(str(target))
        assert resolved == target.resolve()

    def test_traversal_outside_allowlist_rejected(self, tmp_path):
        with pytest.raises(PathTraversalError):
            resolve_dataset_path("../../etc/passwd")

    def test_absolute_path_outside_allowlist_rejected(self):
        with pytest.raises(PathTraversalError):
            resolve_dataset_path("/etc/passwd")

    def test_dotdot_within_relative_segments_rejected(self, tmp_path):
        (tmp_path / "sub").mkdir()
        with pytest.raises(PathTraversalError):
            resolve_dataset_path("sub/../../outside.jsonl")

    def test_default_allowlist_is_cwd_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROUNDCHECK_DATA_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data.jsonl").write_text("{}\n")
        resolved = resolve_dataset_path("data.jsonl")
        assert resolved == (tmp_path / "data.jsonl").resolve()


class TestLoadCasesFromJsonl:
    def test_parses_lines_and_skips_blank(self, tmp_path):
        case = EvalCase(id="1", query="q", answer="a", sources=[Source(id="s1", text="t")])
        path = tmp_path / "cases.jsonl"
        path.write_text(case.model_dump_json() + "\n\n")
        cases = load_cases_from_jsonl(path)
        assert len(cases) == 1
        assert cases[0].id == "1"


class TestReportStore:
    def test_save_and_load_round_trip(self):
        report = Report(
            prompt_version="v1",
            cases=[],
            summary=SuiteSummary(
                report_id="abc123", case_count=0, mean_faithfulness=0.0, worst_cases=[]
            ),
        )
        report.summary.report_id = report.report_id
        save_report(report)
        loaded = load_report(report.report_id)
        assert loaded.report_id == report.report_id

    def test_unknown_id_lists_available(self):
        report = Report(
            prompt_version="v1",
            cases=[],
            summary=SuiteSummary(
                report_id="x", case_count=0, mean_faithfulness=0.0, worst_cases=[]
            ),
        )
        report.summary.report_id = report.report_id
        save_report(report)
        with pytest.raises(ReportNotFoundError) as exc_info:
            load_report("does-not-exist")
        assert report.report_id in exc_info.value.available

    def test_list_report_ids_empty_when_no_reports_dir(self):
        assert list_report_ids() == []


class TestRunSuite:
    async def test_aggregates_faithfulness_and_persists(self):
        decompose = '[{"claim": "A is true.", "answer_span": "A is true"}]'
        verify = json.dumps(
            [
                {
                    "id": "c1",
                    "verdict": "supported",
                    "source_id": "s1",
                    "quoted_span": "A is true",
                    "reason": None,
                }
            ]
        )
        judge = FakeJudge([decompose, verify])
        cases = [
            EvalCase(
                id="case1",
                query="q",
                answer="A is true.",
                sources=[Source(id="s1", text="A is true")],
                relevant_ids=["s1"],
            )
        ]
        report = await run_suite(judge, cases, k_values=[1])
        assert report.summary.case_count == 1
        assert report.summary.mean_faithfulness == 1.0
        assert report.cases[0].retrieval is not None
        assert report.summary.worst_cases == ["case1"]
        # persisted and retrievable
        loaded = load_report(report.report_id)
        assert loaded.report_id == report.report_id

    async def test_no_relevant_ids_falls_back_to_llm_graded_retrieval(self):
        # answer is blank -> decompose_claims short-circuits, no judge call needed
        # for faithfulness; the one canned response is consumed by grade_relevance.
        judge = FakeJudge(['[{"id": "s1", "grade": 2}]'])
        cases = [EvalCase(id="case1", query="q", answer="  ", sources=[Source(id="s1", text="x")])]
        report = await run_suite(judge, cases)
        assert report.cases[0].retrieval is not None
        assert report.cases[0].retrieval.mode == "llm_graded"

    async def test_empty_cases_list(self):
        judge = FakeJudge([])
        report = await run_suite(judge, [])
        assert report.summary.case_count == 0
        assert report.summary.mean_faithfulness == 0.0
        assert report.summary.worst_cases == []
