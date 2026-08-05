"""Console entry point for the local stdio MCP server."""

import asyncio

from mcp.server.stdio import stdio_server

from .server import mcp as server


async def run_stdio() -> None:
    """Run stdio with stdout reserved exclusively for protocol traffic."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Start the local MCP protocol loop."""
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
