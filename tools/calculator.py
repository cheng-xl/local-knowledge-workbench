import math


def calculator(expression: str) -> str:
    allowed = {k: v for k, v in vars(math).items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"
