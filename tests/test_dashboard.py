import json

import pytest

from groundcheck.dashboard import generate_dashboard
from groundcheck.schemas import (
    CaseResult,
    FaithfulnessResult,
    Report,
    RetrievalMetrics,
    RetrievalResult,
    SuiteSummary,
)


def _report(report_id: str, prompt_version: str, cases: list[CaseResult]) -> Report:
    supported = sum(c.faithfulness.supported for c in cases)
    total = sum(c.faithfulness.total_claims for c in cases)
    mean_faithfulness = (supported / total) if total else 1.0
    return Report(
        report_id=report_id,
        prompt_version=prompt_version,
        cases=cases,
        summary=SuiteSummary(
            report_id=report_id,
            case_count=len(cases),
            mean_faithfulness=mean_faithfulness,
            mean_ndcg={5: 0.75},
            worst_cases=[c.id for c in cases[:5]],
        ),
    )


@pytest.fixture
def fixture_reports_dir(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    case1 = CaseResult(
        id="case1",
        query="What is the EIN?",
        faithfulness=FaithfulnessResult(
            score=0.6789, total_claims=2, supported=1, unsupported=1, contradicted=0, claims=[]
        ),
        retrieval=RetrievalResult(
            mode="gold_labels",
            mrr=1.0,
            ndcg={5: 0.9},
            metrics=[RetrievalMetrics(k=5, precision=0.6, recall=0.8)],
        ),
    )
    case2 = CaseResult(
        id="case2",
        query="What is the catering price?",
        faithfulness=FaithfulnessResult(
            score=0.0, total_claims=2, supported=0, unsupported=1, contradicted=1, claims=[]
        ),
        retrieval=None,
    )

    report_a = _report("report_a", "v1", [case1, case2])
    report_b = _report("report_b", "v2", [case1])

    (reports_dir / "report_a.json").write_text(report_a.model_dump_json())
    (reports_dir / "report_b.json").write_text(report_b.model_dump_json())
    return reports_dir


class TestGenerateDashboard:
    def test_writes_html_file(self, fixture_reports_dir, tmp_path):
        out = tmp_path / "dashboard.html"
        generate_dashboard(fixture_reports_dir, out)
        assert out.exists()
        html = out.read_text()
        assert html.startswith("<!doctype html>")
        assert "</html>" in html

    def test_embedded_data_contains_known_values(self, fixture_reports_dir, tmp_path):
        out = tmp_path / "dashboard.html"
        generate_dashboard(fixture_reports_dir, out)
        html = out.read_text()

        start = html.index("const DATA = ") + len("const DATA = ")
        end = html.index(";\n", start)
        data = json.loads(html[start:end])

        assert data["generated_reports_count"] == 2
        assert data["total_cases"] == 3
        report_ids = {r["report_id"] for r in data["reports"]}
        assert report_ids == {"report_a", "report_b"}

        report_a = next(r for r in data["reports"] if r["report_id"] == "report_a")
        case1_payload = next(c for c in report_a["cases"] if c["id"] == "case1")
        assert case1_payload["faithfulness_score"] == pytest.approx(0.6789)
        assert case1_payload["retrieval_precision_5"] == pytest.approx(0.6)
        assert case1_payload["query"] == "What is the EIN?"

        case2_payload = next(c for c in report_a["cases"] if c["id"] == "case2")
        assert case2_payload["retrieval_precision_5"] is None

    def test_no_reports_produces_empty_but_valid_dashboard(self, tmp_path):
        empty_dir = tmp_path / "empty_reports"
        empty_dir.mkdir()
        out = tmp_path / "dashboard.html"
        generate_dashboard(empty_dir, out)
        html = out.read_text()
        assert '"generated_reports_count": 0' in html
        assert "</html>" in html

    def test_missing_reports_dir_treated_as_empty(self, tmp_path):
        out = tmp_path / "dashboard.html"
        generate_dashboard(tmp_path / "does_not_exist", out)
        html = out.read_text()
        assert '"generated_reports_count": 0' in html

    def test_creates_parent_directories(self, fixture_reports_dir, tmp_path):
        out = tmp_path / "nested" / "dir" / "dashboard.html"
        generate_dashboard(fixture_reports_dir, out)
        assert out.exists()
