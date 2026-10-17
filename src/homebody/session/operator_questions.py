"""Ask the operator one question while held in a pause; the explorer accepts the answer for
the task it was asked in and records every step."""
import time
import uuid

from homebody.primitives.navigation.errors import Stopped


def ask_question(runtime, rec):
    runtime.check()
    question = {"id": str(uuid.uuid4()), "decision": rec["n"],
                "question": rec["question"], "why": rec["why"], "status": "pending",
                "brief": runtime._current_brief(), "generation": runtime.task_region_generation,
                "visual_epoch": getattr(runtime.visual_history, "epoch", None),
                "asked_wall_s": time.time()}
    with runtime.lock:
        runtime.pending_question = question
        runtime.operator_questions.append(question)
        runtime._persist_question(question, "asked")
        runtime.plan, runtime.next_wp, runtime.pending_face = [], None, None
        if not runtime.request_pause() and not runtime.source.pause_flag.is_set():
            runtime._invalidate_question("question could not enter held pause")
            raise Stopped("operator question could not confirm pause")
    try:
        runtime._service_pause()
        runtime.check()
        runtime.decision_pose()
    finally:
        runtime._invalidate_question("question interrupted")
    rec.update(outcome="answered", operator_question=dict(question))
    runtime._save()
