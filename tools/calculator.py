def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A string like '12 * 13' or 'sqrt(144) + abs(-5)'.

    Returns:
        The numeric result as a string, or an error message.
    """
    import math
    allowed = {
        k: v for k, v in vars(math).items() if not k.startswith("_")
    }
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
