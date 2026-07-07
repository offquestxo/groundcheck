"""Versioned judge prompt templates. Bump PROMPT_VERSION on any wording change
that could shift verdicts, so reports stay attributable to the prompt that produced them.
"""

PROMPT_VERSION = "v1"

DECOMPOSE_SYSTEM = (
    "You are a precise fact-extraction assistant used inside an automated RAG "
    "evaluation pipeline. You decompose a piece of text into atomic factual claims."
)

DECOMPOSE_PROMPT = """Decompose the ANSWER below into a list of atomic factual claims.

Rules:
- Each claim must be a single, self-contained factual assertion (one fact per claim).
- Split compound sentences into separate claims.
- Skip greetings, hedges, and pure opinions/questions that assert no fact.
- "claim": preserve enough context to be checked independently (resolve
  pronouns; e.g. "It has 12 layers" -> "The X model has 12 layers").
- "answer_span": the exact verbatim substring of ANSWER this claim was drawn from.
- Respond with ONLY a JSON array of objects, no prose, no markdown fences.
  Each object: {{"claim": str, "answer_span": str}}

ANSWER:
{answer}
"""

VERIFY_SYSTEM = (
    "You are a precise fact-verification assistant used inside an automated RAG "
    "evaluation pipeline. You check factual claims against source documents."
)

VERIFY_PROMPT = """For each CLAIM below (given by id), decide whether the SOURCES support it,
contradict it, or neither.

Rules:
- "supported": at least one source directly confirms the claim.
- "contradicted": at least one source directly conflicts with the claim.
- "unsupported": the sources neither confirm nor conflict with the claim (the
  claim isn't addressed at all, or the support is too indirect/vague).
- Always include the source id and a short verbatim quoted span backing the
  verdict when one exists. For "unsupported", still include the closest
  related passage if one exists, even though it doesn't fully support the
  claim; use null for source_id/quoted_span only if the sources are entirely
  silent on the topic.
- For any non-"supported" verdict, include a one-line reason.
- Respond with ONLY a JSON array of objects, no prose, no markdown fences.
  Each object: {{"id": str, "verdict": "supported"|"unsupported"|"contradicted",
  "source_id": str|null, "quoted_span": str|null, "reason": str|null}}

CLAIMS (id: claim):
{claims}

SOURCES:
{sources}
"""

GRADE_RELEVANCE_SYSTEM = (
    "You are a precise retrieval-relevance grading assistant used inside an "
    "automated RAG evaluation pipeline."
)

GRADE_RELEVANCE_PROMPT = """Grade how relevant each retrieved CHUNK is to the QUERY, on a
0-3 scale:
- 3: directly and fully answers the query
- 2: relevant, contains information that helps answer the query
- 1: marginally related, mostly tangential
- 0: not relevant at all

QUERY: {query}

CHUNKS:
{chunks}

Respond with ONLY a JSON array of objects, no prose, no markdown fences.
Each object: {{"id": str, "grade": 0|1|2|3}}
"""

COMPARE_SYSTEM = (
    "You are an impartial judge comparing two candidate answers to the same "
    "query on specific criteria, used inside an automated RAG evaluation pipeline."
)

COMPARE_PROMPT = """QUERY: {query}

ANSWER {first_label}:
{first_answer}

ANSWER {second_label}:
{second_answer}

{sources_block}
Judge which answer is better on each of these criteria: {criteria}.
For "faithfulness", prefer the answer better supported by the sources (if given).
For "completeness", prefer the answer that more fully addresses the query.
For "relevance", prefer the answer more directly on-topic.

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"per_criterion": [{{"criterion": str, "winner": "{first_label}"|"{second_label}"|"tie",
"rationale": str}}], "overall_winner": "{first_label}"|"{second_label}"|"tie",
"rationale": str}}
"""
