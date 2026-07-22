"""Generates a single self-contained HTML dashboard from the report store.

Static generation, not a running app: zero deployment/hosting burden, opens
straight from disk (`file://`), trivially screenshot-able. Chart.js is loaded
from a CDN; all report data is embedded inline as a JSON blob so there's no
fetch() call that would fail under a file:// origin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from groundcheck.engine.suite import get_reports_dir
from groundcheck.schemas import Report

DATA_PLACEHOLDER = "__GROUNDCHECK_DASHBOARD_DATA__"


def _load_reports(reports_dir: Path) -> list[Report]:
    if not reports_dir.exists():
        return []
    paths = sorted(reports_dir.glob("*.json"))
    reports = [Report.model_validate_json(p.read_text()) for p in paths]
    return sorted(reports, key=lambda r: r.created_at)


def _precision_at(case_metrics: list[dict[str, Any]], k: int) -> float | None:
    for m in case_metrics:
        if m["k"] == k:
            return m["precision"]
    return None


def _build_payload(reports: list[Report]) -> dict[str, Any]:
    report_payloads = []
    total_cases = 0
    faithfulness_sum = 0.0
    precision_values: list[float] = []

    for report in reports:
        cases_payload = []
        for case in report.cases:
            precision_5 = None
            if case.retrieval is not None:
                metrics = [m.model_dump() for m in case.retrieval.metrics]
                precision_5 = _precision_at(metrics, 5)
                if precision_5 is not None:
                    precision_values.append(precision_5)
            cases_payload.append(
                {
                    "id": case.id,
                    "query": case.query,
                    "faithfulness_score": case.faithfulness.score,
                    "unsupported_claims": case.faithfulness.unsupported,
                    "contradicted_claims": case.faithfulness.contradicted,
                    "retrieval_mode": case.retrieval.mode if case.retrieval else None,
                    "retrieval_precision_5": precision_5,
                }
            )

        total_cases += report.summary.case_count
        faithfulness_sum += report.summary.mean_faithfulness * max(report.summary.case_count, 1)

        report_payloads.append(
            {
                "report_id": report.report_id,
                "created_at": report.created_at,
                "prompt_version": report.prompt_version,
                "case_count": report.summary.case_count,
                "mean_faithfulness": report.summary.mean_faithfulness,
                "mean_ndcg": report.summary.mean_ndcg,
                "worst_cases": report.summary.worst_cases,
                "cases": cases_payload,
            }
        )

    mean_faithfulness_overall = (faithfulness_sum / total_cases) if total_cases else 0.0
    mean_precision_overall = (
        sum(precision_values) / len(precision_values) if precision_values else None
    )

    return {
        "generated_reports_count": len(reports),
        "total_cases": total_cases,
        "mean_faithfulness_overall": mean_faithfulness_overall,
        "mean_precision_overall": mean_precision_overall,
        "reports": report_payloads,
    }


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>groundcheck dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 960px;
    margin: 2rem auto;
    padding: 0 1rem;
    color: #1a1a1a;
    background: #fff;
  }
  @media (prefers-color-scheme: dark) {
    body { color: #e6e6e6; background: #14161a; }
    .card { background: #1e2126 !important; border-color: #333 !important; }
    table { border-color: #333 !important; }
    th, td { border-color: #333 !important; }
    th { background: #24272d !important; }
    tr:hover { background: #24272d !important; }
  }
  h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
  .subtitle { opacity: 0.7; margin-top: 0; margin-bottom: 1.5rem; font-size: 0.9rem; }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.75rem;
    margin-bottom: 2rem;
  }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 0.9rem 1rem; background: #fafafa; }
  .card .value { font-size: 1.6rem; font-weight: 600; }
  .card .label { font-size: 0.8rem; opacity: 0.7; }
  section { margin-bottom: 2.5rem; }
  h2 { font-size: 1.1rem; border-bottom: 1px solid #ddd; padding-bottom: 0.4rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid #eee; }
  th { cursor: pointer; user-select: none; background: #f5f5f5; white-space: nowrap; }
  th:hover { opacity: 0.8; }
  th.sorted-asc::after { content: " \\2191"; }
  th.sorted-desc::after { content: " \\2193"; }
  tr:hover { background: #f9f9f9; }
  select { padding: 0.3rem 0.5rem; font-size: 0.9rem; margin-bottom: 0.75rem; }
  .worst-list { padding-left: 1.2rem; }
  .worst-list li { margin-bottom: 0.3rem; }
  footer {
    opacity: 0.6;
    font-size: 0.8rem;
    margin-top: 3rem;
    border-top: 1px solid #ddd;
    padding-top: 1rem;
  }
  .empty { opacity: 0.6; font-style: italic; }
</style>
</head>
<body>

<h1>groundcheck dashboard</h1>
<p class="subtitle">Generated from the local report store.
  No server required -- open this file directly.</p>

<div class="cards" id="summary-cards"></div>

<section>
  <h2>Faithfulness by report</h2>
  <canvas id="faithfulness-chart" height="90"></canvas>
</section>

<section>
  <h2>Per-case breakdown</h2>
  <select id="report-select"></select>
  <table id="case-table">
    <thead>
      <tr>
        <th data-key="query">Query</th>
        <th data-key="faithfulness_score">Faithfulness</th>
        <th data-key="unsupported_claims">Unsupported</th>
        <th data-key="contradicted_claims">Contradicted</th>
        <th data-key="retrieval_precision_5">Precision@5</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
</section>

<section>
  <h2>Worst 5 cases (selected report)</h2>
  <ul class="worst-list" id="worst-cases"></ul>
</section>

<footer id="footer"></footer>

<script>
const DATA = __GROUNDCHECK_DASHBOARD_DATA__;

function fmtPct(x) {
  return x === null || x === undefined ? "-" : (x * 100).toFixed(0) + "%";
}

function renderSummaryCards() {
  const cards = [
    { label: "Total reports", value: DATA.generated_reports_count },
    { label: "Total cases evaluated", value: DATA.total_cases },
    { label: "Mean faithfulness (all reports)", value: fmtPct(DATA.mean_faithfulness_overall) },
    { label: "Mean precision@5 (where available)", value: fmtPct(DATA.mean_precision_overall) },
  ];
  const el = document.getElementById("summary-cards");
  el.innerHTML = cards.map(c =>
    `<div class="card"><div class="value">${c.value}</div><div class="label">${c.label}</div></div>`
  ).join("");
}

function renderFaithfulnessChart() {
  const ctx = document.getElementById("faithfulness-chart");
  if (!DATA.reports.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No reports yet.";
    ctx.replaceWith(p);
    return;
  }
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: DATA.reports.map(r => r.report_id),
      datasets: [{
        label: "Mean faithfulness",
        data: DATA.reports.map(r => r.mean_faithfulness),
        backgroundColor: "#5b8def",
      }],
    },
    options: {
      scales: { y: { beginAtZero: true, max: 1 } },
      plugins: { legend: { display: false } },
    },
  });
}

let currentSort = { key: null, dir: 1 };

function renderCaseTable(report) {
  const tbody = document.querySelector("#case-table tbody");
  if (!report || !report.cases.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No cases in this report.</td></tr>';
    return;
  }
  let rows = [...report.cases];
  if (currentSort.key) {
    rows.sort((a, b) => {
      const av = a[currentSort.key], bv = b[currentSort.key];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      if (av < bv) return -1 * currentSort.dir;
      if (av > bv) return 1 * currentSort.dir;
      return 0;
    });
  }
  tbody.innerHTML = rows.map(c => `
    <tr>
      <td>${c.query}</td>
      <td>${fmtPct(c.faithfulness_score)}</td>
      <td>${c.unsupported_claims}</td>
      <td>${c.contradicted_claims}</td>
      <td>${fmtPct(c.retrieval_precision_5)}</td>
    </tr>
  `).join("");
}

function renderWorstCases(report) {
  const el = document.getElementById("worst-cases");
  if (!report || !report.worst_cases.length) {
    el.innerHTML = '<li class="empty">None recorded.</li>';
    return;
  }
  el.innerHTML = report.worst_cases.map(id => `<li>${id}</li>`).join("");
}

function renderFooter(report) {
  const el = document.getElementById("footer");
  if (!report) { el.textContent = ""; return; }
  el.textContent =
    `Selected report ${report.report_id} was judged with prompt version ${report.prompt_version}.`;
}

function selectReport(reportId) {
  const report = DATA.reports.find(r => r.report_id === reportId) || null;
  renderCaseTable(report);
  renderWorstCases(report);
  renderFooter(report);
}

function renderReportSelect() {
  const select = document.getElementById("report-select");
  if (!DATA.reports.length) {
    select.innerHTML = '<option>No reports yet</option>';
    return;
  }
  select.innerHTML = DATA.reports
    .map(r => `<option value="${r.report_id}">${r.report_id}</option>`)
    .join("");
  select.addEventListener("change", () => selectReport(select.value));
  const latestId = DATA.reports[DATA.reports.length - 1].report_id;
  select.value = latestId;
  selectReport(latestId);
}

document.querySelectorAll("#case-table th[data-key]").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    document.querySelectorAll("#case-table th")
      .forEach(h => h.classList.remove("sorted-asc", "sorted-desc"));
    if (currentSort.key === key) {
      currentSort.dir *= -1;
    } else {
      currentSort = { key, dir: 1 };
    }
    th.classList.add(currentSort.dir === 1 ? "sorted-asc" : "sorted-desc");
    const select = document.getElementById("report-select");
    selectReport(select.value);
  });
});

renderSummaryCards();
renderFaithfulnessChart();
renderReportSelect();
</script>
</body>
</html>
"""


def generate_dashboard(reports_dir: Path, output_path: Path) -> None:
    """Read every report in `reports_dir` and render a self-contained HTML
    dashboard to `output_path`."""
    reports = _load_reports(reports_dir)
    payload = _build_payload(reports)
    html = _HTML_TEMPLATE.replace(DATA_PLACEHOLDER, json.dumps(payload))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a static groundcheck eval dashboard.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/dashboard.html"),
        help="Output HTML file path (default: docs/dashboard.html)",
    )
    args = parser.parse_args()
    generate_dashboard(get_reports_dir(), args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
