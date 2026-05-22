from datetime import datetime, timezone, timedelta


def datetime_tool(action: str = "now", timezone_offset: int = 8) -> str:
    """Get current date/time or calculate date differences.

    Args:
        action: One of 'now' | 'today' | 'weekday' | 'timestamp'.
        timezone_offset: UTC offset in hours (default 8 = China Standard Time).

    Returns:
        Formatted date/time string.
    """
    now = datetime.now(timezone(timedelta(hours=timezone_offset)))
    actions = {
        "now": lambda: now.strftime("%Y-%m-%d %H:%M:%S"),
        "today": lambda: now.strftime("%Y-%m-%d"),
        "weekday": lambda: now.strftime("%A"),
        "timestamp": lambda: str(int(now.timestamp())),
    }
    fn = actions.get(action, actions["now"])
    return fn()
