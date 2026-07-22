"""FastMCP app: registers all groundcheck tools/resources/prompts. Stdio entry point.

Also used as the shared app for the Streamable HTTP entry (see http.py).
"""

from __future__ import annotations

import json
import logging
import os
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from groundcheck.engine import compare as compare_engine
from groundcheck.engine import retrieval as retrieval_engine
from groundcheck.engine import suite as suite_engine
from groundcheck.engine.claims import detect_hallucinations, evaluate_faithfulness
from groundcheck.judge.sampler import get_judge
from groundcheck.schemas import (
    CompareResult,
    EvalCase,
    FaithfulnessResult,
    HallucinationResult,
    Report,
    ResponseFormat,
    RetrievalResult,
    Source,
    SuiteSummary,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("groundcheck")

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

mcp = FastMCP(
    "groundcheck",
    instructions=(
        "Evaluates RAG (retrieval-augmented generation) outputs: faithfulness scoring, "
        "hallucination detection, and retrieval quality metrics. Judge-backed tools use "
        "MCP sampling (your own model) by default -- no API key needed."
    ),
    host=os.environ.get("GROUNDCHECK_HTTP_HOST", "127.0.0.1"),
    port=int(os.environ.get("GROUNDCHECK_HTTP_PORT", "8000")),
    # Stateless: no reliance on protocol sessions. run_suite returns a report_id,
    # get_report accepts it -- reports are the only state, and they live on disk.
    stateless_http=True,
    json_response=True,
)


def _apply_faithfulness_format(
    result: FaithfulnessResult, response_format: ResponseFormat
) -> FaithfulnessResult:
    if response_format == "detailed":
        return result
    problem_claims = [c for c in result.claims if c.verdict != "supported"]
    return result.model_copy(update={"claims": problem_claims})


@mcp.tool(annotations=_READ_ONLY)
async def groundcheck_evaluate_faithfulness(
    answer: str,
    sources: list[Source],
    ctx: Context,
    response_format: ResponseFormat = "concise",
) -> FaithfulnessResult:
    """Score how well `answer` is supported by `sources`, claim by claim.

    Use this when you need a faithfulness score (0-1) and want to see every
    claim's verdict, not just the problems -- for that, use
    groundcheck_detect_hallucinations instead, which is tighter and cheaper.

    Args:
        answer: the RAG-generated answer text to check.
        sources: list of {id, text} chunks the answer was generated from,
            e.g. [{"id": "doc1#chunk3", "text": "..."}].
        response_format: "concise" (default) returns the score, claim counts,
            and only unsupported/contradicted claims. "detailed" returns
            every claim's verdict.

    Returns a score (supported/total claims), counts by verdict, and a claims
    list (filtered per response_format). Costs 2 model calls via your client's
    sampling (decompose, then verify) -- no API key needed if your client
    supports sampling.
    """
    judge = get_judge(ctx)
    result = await evaluate_faithfulness(judge, answer, sources)
    return _apply_faithfulness_format(result, response_format)


@mcp.tool(annotations=_READ_ONLY)
async def groundcheck_detect_hallucinations(
    answer: str, sources: list[Source], ctx: Context
) -> HallucinationResult:
    """Find only the unsupported or contradicted claims in `answer`.

    Use this in a fix-it loop where you only care about what's wrong, not a
    full faithfulness report -- for the full picture with a score and every
    claim's verdict, use groundcheck_evaluate_faithfulness instead.

    Args:
        answer: the RAG-generated answer text to check.
        sources: list of {id, text} chunks the answer was generated from.

    Returns an empty list if the answer is clean. Otherwise, each entry has
    the exact answer span, the closest source passage that fails to support
    it, and a one-line reason. Costs 2 model calls via your client's
    sampling -- no API key needed if your client supports sampling.
    """
    judge = get_judge(ctx)
    return await detect_hallucinations(judge, answer, sources)


@mcp.tool(annotations=_READ_ONLY)
async def groundcheck_evaluate_retrieval(
    query: str,
    retrieved: list[Source],
    ctx: Context,
    relevant_ids: list[str] | None = None,
    k_values: list[int] | None = None,
) -> RetrievalResult:
    """Score retrieval quality for `retrieved` chunks against `query`.

    Two modes, chosen automatically:
    - Mode A (pass `relevant_ids`): precision@k, recall@k, MRR, NDCG computed
      by pure math from your gold relevance labels. Instant, no model calls.
    - Mode B (omit `relevant_ids`): no gold labels available, so each chunk
      is graded 0-3 for relevance via one sampling call, then the same
      metrics are computed from those grades. Use this when you don't have
      a labeled relevant-docs set for this query.

    Args:
        query: the search query the chunks were retrieved for.
        retrieved: ranked list of {id, text} chunks, in retrieval order (rank matters).
        relevant_ids: ids of chunks known to be relevant. Supply this whenever
            you have gold labels -- Mode A is free and exact.
        k_values: cutoffs to compute metrics at (default [3, 5, 10]).

    Output states which mode ran. Mode A: instant. Mode B: 1 model call, no
    API key needed if your client supports sampling.
    """
    judge = None
    if relevant_ids is None:
        judge = get_judge(ctx)
    return await retrieval_engine.evaluate_retrieval(
        judge, query, retrieved, relevant_ids, k_values or [3, 5, 10]
    )


@mcp.tool(annotations=_READ_ONLY)
async def groundcheck_compare(
    query: str,
    answer_a: str,
    answer_b: str,
    ctx: Context,
    sources: list[Source] | None = None,
    criteria: list[str] | None = None,
) -> CompareResult:
    """Judge which of two candidate answers to `query` is better.

    Use this to A/B two RAG configurations (prompts, retrievers, models) on
    the same query. Position bias is mitigated automatically: the judge sees
    (A,B) and (B,A) in separate calls, and any criterion whose verdict flips
    with presentation order is reported as a tie rather than a pick.

    Args:
        query: the shared query both answers respond to.
        answer_a: first candidate answer.
        answer_b: second candidate answer.
        sources: optional list of {id, text} chunks, used to judge faithfulness.
        criteria: judged criteria (default ["faithfulness", "completeness", "relevance"]).

    Returns a winner ("a"/"b"/"tie/uncertain"), a verdict per criterion, and a
    brief rationale. Costs 2 model calls via your client's sampling -- no API
    key needed if your client supports sampling.
    """
    judge = get_judge(ctx)
    return await compare_engine.compare_answers(
        judge,
        query,
        answer_a,
        answer_b,
        sources,
        criteria or ["faithfulness", "completeness", "relevance"],
    )


@mcp.tool(annotations=_READ_ONLY)
async def groundcheck_run_suite(
    ctx: Context,
    cases: list[EvalCase] | None = None,
    dataset_path: str | None = None,
    k_values: list[int] | None = None,
) -> SuiteSummary:
    """Run faithfulness + retrieval over a batch of cases.

    Use this to evaluate a whole RAG pipeline run rather than one answer at a
    time. Supply exactly one of `cases` or `dataset_path`. Retrieval mode is
    chosen per case exactly like groundcheck_evaluate_retrieval: gold labels
    (Mode A, free) when a case has `relevant_ids`, LLM-graded relevance
    (Mode B, 1 model call) otherwise.

    Args:
        cases: inline list of {id, query, answer, sources, relevant_ids?}.
        dataset_path: path to a JSONL file of the same case objects, one per
            line. Must resolve inside the allowlisted data directory (env
            GROUNDCHECK_DATA_DIR, default cwd) -- paths outside it are rejected.
        k_values: retrieval cutoffs (default [3, 5, 10]).

    Persists a full report and returns a summary (mean faithfulness, mean
    NDCG, worst 5 cases, report_id). Fetch the full report with
    groundcheck_get_report(report_id). Cost scales with case count: ~2-3 model
    calls per case via your client's sampling (2 for faithfulness, +1 for
    retrieval when a case has no `relevant_ids`).
    """
    if (cases is None) == (dataset_path is None):
        raise ValueError("Provide exactly one of `cases` or `dataset_path`, not both/neither.")
    if dataset_path is not None:
        resolved = suite_engine.resolve_dataset_path(dataset_path)
        cases = suite_engine.load_cases_from_jsonl(resolved)
    judge = get_judge(ctx)
    report = await suite_engine.run_suite(judge, cases or [], k_values)
    return report.summary


@mcp.tool(annotations=_READ_ONLY)
def groundcheck_get_report(report_id: str, response_format: ResponseFormat = "concise") -> Report:
    """Fetch a previously persisted evaluation report by id.

    Args:
        report_id: the id returned by groundcheck_run_suite.
        response_format: "concise" returns just the summary (scores,
            aggregates, worst cases). "detailed" returns every case's full
            faithfulness and retrieval results.

    Raises an error listing available report ids if `report_id` is unknown.
    No model calls -- reads from the local report store.
    """
    report = suite_engine.load_report(report_id)
    if response_format == "concise":
        return report.model_copy(update={"cases": []})
    return report


@mcp.resource("groundcheck://reports")
def list_reports_resource() -> str:
    """All persisted report ids, as JSON."""
    return json.dumps({"report_ids": suite_engine.list_report_ids()})


@mcp.resource("groundcheck://reports/{report_id}")
def get_report_resource(report_id: str) -> str:
    """A single persisted report, as JSON."""
    return suite_engine.load_report(report_id).model_dump_json(indent=2)


@mcp.prompt()
def audit_rag_response(answer: str, sources: str) -> str:
    """Walks through supplying an answer + sources and requests a full faithfulness audit."""
    return (
        "I have a RAG-generated answer and the sources it was based on. Please run a "
        "full faithfulness audit using groundcheck_evaluate_faithfulness with "
        'response_format="detailed", then summarize any unsupported or contradicted '
        f"claims.\n\nANSWER:\n{answer}\n\nSOURCES:\n{sources}"
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
