import pytest


@pytest.mark.asyncio
async def test_calculator_tool():
    from tools.calculator import calculator
    result = calculator.invoke("12 * 13")
    assert "156" in result


@pytest.mark.asyncio
async def test_datetime_tool():
    from tools.datetime_tool import datetime_tool
    result = datetime_tool.invoke("today")
    assert len(result) == 10  # YYYY-MM-DD


@pytest.mark.asyncio
async def test_mcp_server_tools():
    from mcp_server import app
    tools = await app.list_tools()  # type: ignore
    assert len(tools) == 2
    tool_names = {t.name for t in tools}
    assert tool_names == {"read_file", "list_dir"}
