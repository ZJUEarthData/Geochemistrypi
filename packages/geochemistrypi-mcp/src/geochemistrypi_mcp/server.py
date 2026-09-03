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
needed; use the compact task-filtered view when the task is known and request
the full evidence inventory only when it is actually needed. The task-filtered
view contains the exact validate_analysis request contract: follow its field
locations and discriminators once instead of probing with guessed requests.
Locate only the
dataset the user identifies. Inspect it only when exact columns, shape, hashes,
values, or inferred types are still needed and are not already available from
the user's request or a trusted earlier tool result; do not add an inspection
call as a ritual before validation. Use names-only inspection unless values or
inferred types are required, and never invent a file or column name. Treat
every explicit scientific choice as authoritative, including the task, data,
column roles, model, tuning mode, and
parameter values. Infer or default only choices the user omitted. Validate every
explicit choice against the reported capabilities and safety limits; if it is
unsupported or invalid, explain the conflict and ask the user rather than
silently substituting another choice. If the file, scientific goal, or column
meaning is ambiguous, ask one brief question at a time in domain language,
using real names and clear choices from the inspection. Explain the proposed
analysis in plain language. If the user has not already authorized execution,
wait for confirmation; an explicit request to run or execute the analysis is
already confirmation and must not trigger a second confirmation turn. Then
start the exact validation receipt without reconstructing the request only when
it reports execution_ready=true. Never start a blocked validation receipt;
if its compact view reports truncated content, read that receipt once with its
exact validation_id, request_hash, and detail=full instead of repeating the
same validation or guessing omitted blockers. Explain its complete blocking
issues and validate again only after they are resolved. Then
normally use one bounded result wait;
request status only when progress detail is needed, never poll in a tight loop.
If that bounded wait returns a pending receipt, treat it as continuing work, not
as a tool or scientific failure, and make one later bounded result call rather
than starting a recovery flow.
Do not fetch the same terminal page twice. Request a returned
next_artifact_offset only when additional artifact receipts are actually
needed. Never describe partial_failure or an incomplete artifact contract as a
complete result. For a successful complete result, an external confirmation
should use the returned SHA-256 with the conditional result field so metrics
and artifacts are not replayed.
For a failed or cancelled receipt, report its bounded failure without inferring
scientific validity or artifact completeness.
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
