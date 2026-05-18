from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from config import settings
from loguru import logger
import os


app = Server("local-filesystem")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read the content of a file at the given path",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        Tool(
            name="list_dir",
            description="List files and directories at the given path",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    ]


def _safe_path(path: str) -> str:
    allowed = os.path.realpath(settings.mcp_allowed_path)
    full = os.path.realpath(os.path.join(allowed, path))
    if not full.startswith(allowed):
        raise ValueError(f"Path traversal denied: {path}")
    return full


@app.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:
    if name == "read_file":
        path = _safe_path(args["path"])
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"read_file: {path} ({len(content)} chars)")
        return [TextContent(type="text", text=content)]
    elif name == "list_dir":
        path = _safe_path(args.get("path", "."))
        entries = os.listdir(path)
        logger.info(f"list_dir: {path} ({len(entries)} entries)")
        return [TextContent(type="text", text=str(entries))]
    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
