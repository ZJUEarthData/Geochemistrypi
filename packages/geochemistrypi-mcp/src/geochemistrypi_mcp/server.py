"""Official-SDK stdio server construction."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server import Server

from .api.tools import build_tool_handlers
from .config.constants import SERVER_NAME, SERVER_VERSION
from .config.settings import McpSettings
from .runtime.runs import RunManager

SERVER_INSTRUCTIONS = """
Help geochemists analyze local data with GeochemistryPi. Accept short,
ordinary-language requests; the user does not need to know MCP tool names, JSON,
schema fields, CLI commands, model names, or machine-learning terminology.
Choose the tools and internal fields yourself. Discover capabilities when
needed, locate and inspect only the dataset the user identifies, and never
invent a file or column name. Treat every explicit scientific choice as
authoritative, including the task, data, column roles, model, tuning mode, and
parameter values. Infer or default only choices the user omitted. Validate every
explicit choice against the reported capabilities and safety limits; if it is
unsupported or invalid, explain the conflict and ask the user rather than
silently substituting another choice. If the file, scientific goal, or column
meaning is ambiguous, ask one brief question at a time in domain language,
using real names and clear choices from the inspection. Explain the proposed
analysis in plain language and wait for confirmation before execution. Then
start the exact validated request and normally use one bounded result wait;
request status only when progress detail is needed, never poll in a tight loop.
Summarize the original GeochemistryPi results and output location. Use stable
experiment IDs internally for history and start the local MLflow UI only when
explicitly requested. Never expose implementation details or claim support
beyond the reported capabilities. All scientific outputs come from the existing
GeochemistryPi CLI.
""".strip()


def create_server(settings: McpSettings | None = None, run_manager: RunManager | None = None) -> Server:
    """Build a strict, client-neutral low-level server with one run manager."""
    resolved_settings = settings or McpSettings.from_environment()
    runs = run_manager or RunManager(resolved_settings)
    list_tools, call_tool = build_tool_handlers(resolved_settings, runs)

    @asynccontextmanager
    async def lifespan(_: Server) -> AsyncIterator[dict[str, object]]:
        try:
            yield {"run_manager": runs}
        finally:
            runs.close()

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


logging.basicConfig(level=logging.INFO)
mcp = create_server()
