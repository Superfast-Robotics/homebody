"""How a dry run is reported back to the program's author: text plus images to attach."""
import json

MAX_LOG_LINES = 60


def report(env, returned, error, seconds):
    """(text, images) for one dry run. IMAGES maps a file name to an RGB array."""
    state = env.state()
    lines = [f"Returned: {str(returned)[:300]}", f"Error: {error or 'none'}", f"Simulated seconds: {seconds:.1f}",
             f"End state: {json.dumps(state)}", f"Visible at the end: {json.dumps(env.visible())}"]
    log = env._log[-MAX_LOG_LINES:]
    lines.append("Log (last lines):\n" + ("\n".join(log) if log else "(empty)"))
    frame = env._frame()
    images = {"end_head.jpg": frame.rgb}
    return "\n".join(lines), images
