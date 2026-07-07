# Using groundcheck with Claude Code

```bash
claude mcp add groundcheck -- uvx groundcheck
```

That's it -- Claude Code will start `groundcheck` over stdio the next time it
needs one of its tools. No API key required: Claude Code supports MCP
sampling, so `groundcheck_evaluate_faithfulness`, `groundcheck_detect_hallucinations`,
`groundcheck_evaluate_retrieval` (Mode B), and `groundcheck_compare` all use
Claude Code's own model as the judge.

To confirm it's connected:

```bash
claude mcp list
```

## Cursor

Cursor uses the same `mcpServers` config shape as Claude Desktop. Add this to
`.cursor/mcp.json` (project-level) or your global Cursor MCP settings:

```json
{
  "mcpServers": {
    "groundcheck": {
      "command": "uvx",
      "args": ["groundcheck"]
    }
  }
}
```

Cursor does not yet support MCP sampling for all models, so if a
judge-backed tool errors with "no judge available," set `ANTHROPIC_API_KEY`
in your environment as a fallback.
