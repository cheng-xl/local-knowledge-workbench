from langchain_core.tools import tool
from datetime import datetime, timezone, timedelta


@tool
def datetime_tool(action: str = "now", timezone_offset: int = 8) -> str:
    """Get current date/time or calculate date differences.
    Actions: 'now' | 'today' | 'weekday' | 'timestamp'."""
    now = datetime.now(timezone(timedelta(hours=timezone_offset)))
    actions = {
        "now": lambda: now.strftime("%Y-%m-%d %H:%M:%S"),
        "today": lambda: now.strftime("%Y-%m-%d"),
        "weekday": lambda: now.strftime("%A"),
        "timestamp": lambda: str(int(now.timestamp())),
    }
    fn = actions.get(action, actions["now"])
    return fn()
