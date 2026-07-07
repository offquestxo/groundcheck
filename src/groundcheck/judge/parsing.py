"""Shared helper for parsing judge responses that are supposed to be raw JSON."""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> object:
    """Strip markdown code fences models sometimes add despite instructions."""
    cleaned = _FENCE_RE.sub("", text).strip()
    return json.loads(cleaned)
