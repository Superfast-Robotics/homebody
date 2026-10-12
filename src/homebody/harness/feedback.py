"""How a customized skill's run is reported back to the agent, as a skill result."""

MAX_LOG_LINES = 40


def result(env, returned, error):
    """A skill result dict {code, message, details} from what `run` returned or raised."""
    log = env._log[-MAX_LOG_LINES:]
    if error is not None:
        return {"code": "SKILL_ERROR", "message": error[:500], "details": {"log": log}}
    if isinstance(returned, dict) and isinstance(returned.get("code"), str):
        code, message = returned["code"], str(returned.get("message", ""))[:500]
    else:
        code, message = "OK", f"returned {str(returned)[:300]}"
    return {"code": code, "message": message, "details": {"log": log}}
