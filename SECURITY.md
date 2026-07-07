# Security

## Threat model

groundcheck is a **read-only, compute-only** MCP server. It scores text you
give it; it does not act on your behalf.

- **No shell access.** The server never spawns subprocesses or executes
  arbitrary code from tool input.
- **No network egress**, except one opt-in path: if `ANTHROPIC_API_KEY` is
  set, `ApiKeyJudge` calls `https://api.anthropic.com/v1/messages` directly.
  This is a fallback for clients that don't support MCP sampling -- it is
  never required, and is off by default.
- **No filesystem writes** outside the report store
  (`$GROUNDCHECK_DATA_DIR/reports/`, default `~/.groundcheck/reports/`). Every
  other tool is pure computation over its inputs.
- **All tool inputs are treated as untrusted.** They arrive from LLM-generated
  tool calls, which in turn may be shaped by content the LLM read from
  elsewhere (retrieved documents, a prior tool result, etc). Nothing here is
  assumed to be operator-supplied just because it came through the protocol.

## Path traversal (the one real attack surface)

`groundcheck_run_suite`'s `dataset_path` parameter names a file on disk for
the server to read -- the one place an untrusted string controls a
filesystem path. It's handled explicitly:

1. `GROUNDCHECK_DATA_DIR` (default: current working directory) defines an
   allowlisted root.
2. `dataset_path` is resolved to an absolute, symlink-free real path
   (`Path.resolve()`), whether given as relative or absolute.
3. The resolved path **must** be inside the allowlisted root
   (`Path.relative_to`), or the tool call is rejected with an error naming
   the rejected path and the allowlist directory -- no partial read happens.

This is covered by tests in `tests/test_suite_engine.py`, including relative
traversal (`../../etc/passwd`), absolute paths outside the root, and
traversal that only resolves outside the root after combining relative
segments (`sub/../../outside.jsonl`).

## Tool annotations

Every tool declares `readOnlyHint=True`, `destructiveHint=False`,
`openWorldHint=False` in its MCP annotations, so clients that surface these
hints to users (or gate auto-approval on them) can do so accurately. Per the
MCP spec, annotations are hints from the server, not a security boundary --
don't rely on them alone.

## stdio vs Streamable HTTP exposure

- **stdio** (the default, `groundcheck` command): the server only talks to
  the process that spawned it (Claude Desktop, Claude Code, Cursor, etc).
  There's no network exposure to reason about.
- **Streamable HTTP** (`groundcheck-http`): this is a real network listener.
  **Do not expose it publicly without setting `GROUNDCHECK_HTTP_TOKEN`.**
  When set, every request must include `Authorization: Bearer <token>` or
  the server returns 401. Without a token set, the HTTP server accepts
  unauthenticated requests -- fine for `localhost`-only local development,
  not fine for anything reachable from the internet. The server also binds
  to `127.0.0.1` by default (override with `GROUNDCHECK_HTTP_HOST` only if
  you understand the exposure you're creating).

## Reporting a vulnerability

Please open a GitHub issue at
[offquestxo/groundcheck/issues](https://github.com/offquestxo/groundcheck/issues)
for non-sensitive reports. For anything you'd rather not post publicly
first, use GitHub's private vulnerability reporting on the repository
(Security tab -> "Report a vulnerability").
