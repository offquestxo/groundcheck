"""Streamable HTTP entry point. Same tools/resources/prompts as stdio (server.py),
served statelessly over HTTP. Not safe to expose publicly without
GROUNDCHECK_HTTP_TOKEN set -- see SECURITY.md.
"""

from __future__ import annotations

import os

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from groundcheck.server import mcp


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Rejects any request lacking `Authorization: Bearer <token>`."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        super().__init__(app)
        self._expected = f"Bearer {token}"

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("authorization") != self._expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def create_app() -> Starlette:
    app = mcp.streamable_http_app()
    token = os.environ.get("GROUNDCHECK_HTTP_TOKEN")
    if token:
        app.add_middleware(BearerAuthMiddleware, token=token)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port)


if __name__ == "__main__":
    main()
