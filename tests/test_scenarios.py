import pytest


class TestCalculator:
    def test_basic(self):
        from tools.calculator import calculator
        assert "156" in calculator("12 * 13")

    def test_expression_with_spaces(self):
        from tools.calculator import calculator
        assert "3" in calculator("1 + 2")

    def test_division(self):
        from tools.calculator import calculator
        result = calculator("10 / 3")
        assert "3.33" in result

    def test_math_functions(self):
        from tools.calculator import calculator
        assert "12.0" in calculator("sqrt(144)")

    def test_error_handling(self):
        from tools.calculator import calculator
        result = calculator("import os")
        assert "Error" in result


class TestDatetime:
    def test_now(self):
        from tools.datetime_tool import datetime_tool
        result = datetime_tool(action="now")
        assert len(result) >= 19  # YYYY-MM-DD HH:MM:SS

    def test_today(self):
        from tools.datetime_tool import datetime_tool
        result = datetime_tool(action="today")
        assert len(result) == 10  # YYYY-MM-DD

    def test_default_action(self):
        from tools.datetime_tool import datetime_tool
        result = datetime_tool()
        assert len(result) >= 19  # defaults to "now"

    def test_timestamp(self):
        from tools.datetime_tool import datetime_tool
        result = datetime_tool(action="timestamp")
        assert int(result) > 0


def test_mcp_server_has_tools():
    """Verify the MCP server is configured with read_file and list_dir tools."""
    import asyncio
    from mcp_server import list_tools
    tools = asyncio.run(list_tools())
    assert len(tools) == 2
    tool_names = {t.name for t in tools}
    assert tool_names == {"read_file", "list_dir"}


def test_mcp_path_safety():
    from mcp_server import _safe_path
    import os
    # Should allow paths within the allowed directory
    result = _safe_path(".")
    assert os.path.isabs(result)
    # Should reject path traversal
    with pytest.raises(ValueError):
        _safe_path("../../../etc/passwd")
