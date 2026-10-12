"""The inner loop for one task: astra writes `run(env)` against the candidate's harness, dry-runs it
on a saved simulation, reads the harness's report, and finally runs one program for real, which
the frozen physics scorer judges. Runs on the remote with the candidate's code on PYTHONPATH.

    inner.py --root CHECKOUT --scene SCENE_DIR --targets FILE --out DIR [--attempts 4] [--video]

The program sees only what the harness gives it. The score never reaches the program's author.
"""
import argparse
import copy
import json
import math
import re
import runpy
import signal
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image

from homebody import harness
from homebody.backends.mujoco import MujocoBackend
from homebody.helpers.kinematics import Arm
from homebody.session.settings import Settings
from homebody.skills.contract import Context

SAFE_BUILTINS = {name: getattr(__builtins__, name, None) if not isinstance(__builtins__, dict) else __builtins__[name]
                 for name in ("abs", "min", "max", "range", "len", "enumerate", "zip", "float", "int", "list", "dict",
                              "tuple", "sorted", "sum", "round", "isinstance", "any", "all", "reversed", "map", "filter",
                              "bool", "str", "set", "frozenset", "divmod", "pow", "repr", "hasattr", "getattr", "Exception",
                              "ValueError", "RuntimeError", "KeyError", "IndexError", "ZeroDivisionError", "TypeError",
                              "AttributeError", "StopIteration", "next", "iter", "print")}
EXECUTE = re.compile(r"^\s*EXECUTE\s+(\d+)\s*$", re.MULTILINE)
CODE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
SETTLE_SECONDS = 3.0
LIFT_M = 0.05
VIDEO_PERIOD = 0.04


class ProgramTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise ProgramTimeout("the program ran out of time")


signal.signal(signal.SIGALRM, _alarm)


class Recording:
    """Real-time 25 fps room and head videos while switched on."""

    def __init__(self, backend, folder):
        self.backend, self.folder, self.on, self.next, self.writers = backend, folder, False, 0.0, {}

    def capture(self):
        if not self.on or self.backend.time < self.next:
            return
        for name, rgb in (("room", self.backend.observer_frame()), ("head", self.backend.snapshot().rgb)):
            if name not in self.writers:
                writer = imageio_ffmpeg.write_frames(str(self.folder / f"{name}.mp4"), (rgb.shape[1], rgb.shape[0]),
                                                     fps=1 / VIDEO_PERIOD, codec="libx264", pix_fmt_in="rgb24",
                                                     pix_fmt_out="yuv420p", macro_block_size=1, ffmpeg_log_level="error")
                writer.send(None)
                self.writers[name] = writer
            self.writers[name].send(np.ascontiguousarray(rgb))
        self.next = self.backend.time + VIDEO_PERIOD

    def close(self):
        for writer in self.writers.values():
            writer.close()


class Robot:
    """The backend as the Robot the Context drives, recording after every advance."""

    def __init__(self, backend, recording):
        self._backend, self._recording = backend, recording

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def advance(self, epoch, seconds):
        self._backend.advance(epoch, seconds)
        self._recording.capture()


class Task:
    def __init__(self, args):
        self.args = args
        root = Path(args.root)
        self.settings = Settings.load(root / "configs/simulation.toml", scene=args.scene)
        self.backend = MujocoBackend(root, self.settings, scene=args.scene)
        args.out.mkdir(parents=True, exist_ok=True)
        self.recording = Recording(self.backend, args.out)
        urdf = root / "assets/robot/g1_29dof_with_hand.urdf"
        calibration = json.loads((root / "assets/robot/calibration.json").read_text())
        robot = Robot(self.backend, self.recording)
        self.ctx = Context(observations=robot, actions=robot, arms={s: Arm(urdf, s) for s in ("left", "right")},
                           settings=self.settings, jaw_samples=calibration["jaw_samples"])
        semantics = json.loads((args.scene / "semantics.json").read_text())["entities"]
        self.memory = [{"id": e["id"], "label": e["label"], "location": e.get("location_description", ""),
                        "anchor_xyz": [round(v, 3) for v in e["reference_anchor_ws_map_m"]],
                        "portable": bool(e.get("portable_candidate"))} for e in semantics if e["id"] in self.backend.label_ids]
        self.labels = {item["id"]: self.backend.label_ids[item["id"]] for item in self.memory}
        self.task_text = json.loads((args.scene / "task_scene.json").read_text())["default_task"]
        for _ in range(4):
            self.backend.advance(self.backend.epoch, 0.5)

    # --- simulation snapshots (sim-only affordance of the frozen runner) ---
    def snapshot(self):
        b = self.backend
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(b.model, spec))
        mujoco.mj_getState(b.model, b.data, state, spec)
        policy = b.controller.policy
        memo = {id(obj): obj for obj in (b.model, b.data, policy._network, policy._adapter, policy._np, policy._torch)}
        counters = {k: copy.deepcopy(getattr(b, k)) for k in ("sequence", "epoch", "_valid", "_velocity_expiry")}
        return state, copy.deepcopy(b.controller, memo), counters, copy.deepcopy(self.ctx.carried)

    def restore(self, saved):
        b = self.backend
        state, controller, counters, carried = saved
        mujoco.mj_setState(b.model, b.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(b.model, b.data)
        b.controller = controller
        for key, value in counters.items():
            setattr(b, key, value)
        b._frame_cache = None
        self.ctx.carried.clear()
        self.ctx.carried.update(carried)

    # --- one program ---
    def run_program(self, code, seconds):
        env = harness.build_env(self.ctx, self.memory, self.labels)
        namespace = {"__builtins__": {**SAFE_BUILTINS, "print": env.log}, "np": np, "math": math}
        started = self.backend.time
        returned, error = None, None
        signal.alarm(int(seconds))
        try:
            exec(compile(code, "run.py", "exec"), namespace)  # noqa: S102 -- the model's program, sandboxed names only
            if "run" not in namespace:
                raise NameError("the code must define run(env)")
            returned = namespace["run"](env)
        except Exception as exc:  # noqa: BLE001 -- the program's failure is its feedback
            tb = traceback.extract_tb(exc.__traceback__)
            where = next((f" (line {f.lineno}: {f.line})" for f in reversed(tb) if f.filename == "run.py"), "")
            error = f"{type(exc).__name__}: {exc}{where}"
        finally:
            signal.alarm(0)
        return env, returned, error, self.backend.time - started

    def ask(self, prompt, images, number):
        folder = Path(tempfile.mkdtemp(prefix=f"inner-{number}-"))
        names = []
        for name, rgb in images.items():
            Image.fromarray(rgb).save(folder / name, quality=85)
            names += ["--image", str(folder / name)]
        out = folder / "reply.txt"
        command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--sandbox",
                   "read-only", "--model", self.args.model, "-c", f'model_reasoning_effort="{self.args.effort}"',
                   "-c", 'web_search="disabled"', "--output-last-message", str(out), "--color", "never", *names, "-"]
        started = time.monotonic()
        done = subprocess.run(command, cwd=folder, input=prompt.encode(), capture_output=True, timeout=1200, check=False)
        reply = out.read_text() if out.exists() else ""
        if not reply:
            raise RuntimeError(f"codex returned nothing ({done.returncode}): {done.stderr.decode(errors='replace')[-500:]}")
        return reply, time.monotonic() - started

    # --- the loop ---
    def main(self):
        transcript = self.args.out / "transcript"
        transcript.mkdir(exist_ok=True)
        attempts, codes, chosen, log = [], {}, None, []
        images = {"start_head.jpg": self.backend.snapshot().rgb}
        Image.fromarray(images["start_head.jpg"]).save(transcript / "start_head.jpg", quality=85)
        for number in range(1, self.args.attempts + 1):
            prompt = harness.render_prompt(self.task_text, self.memory, attempts, self.args.attempts)
            (transcript / f"{number:02d}_prompt.md").write_text(prompt)
            reply, asked = self.ask(prompt, images, number)
            (transcript / f"{number:02d}_reply.md").write_text(reply)
            execute = EXECUTE.search(reply)
            if execute and int(execute.group(1)) in codes:
                chosen = int(execute.group(1))
                log.append({"number": number, "kind": "execute", "chosen": chosen, "codex_seconds": round(asked, 1)})
                break
            match = CODE.search(reply)
            if not match:
                attempts.append({"number": number, "report": "Your reply had no ```python``` code block defining run(env)."})
                log.append({"number": number, "kind": "no_code", "codex_seconds": round(asked, 1)})
                continue
            codes[number] = match.group(1)
            saved = self.snapshot()
            wall = time.monotonic()
            env, returned, error, sim = self.run_program(codes[number], self.args.program_seconds)
            text, extra = harness.report(env, returned, error, sim)
            self.restore(saved)
            attempts.append({"number": number, "report": text})
            images = {"start_head.jpg": images.get("start_head.jpg", extra.get("end_head.jpg")), **extra}
            (transcript / f"{number:02d}_report.md").write_text(text)
            log.append({"number": number, "kind": "dry_run", "codex_seconds": round(asked, 1), "error": error,
                        "returned": str(returned)[:200], "sim_seconds": round(sim, 1),
                        "wall_seconds": round(time.monotonic() - wall, 1)})
        if chosen is None and codes:
            chosen = max(codes)
        result = {"task": self.task_text, "attempts": log, "executed": chosen}
        if chosen is not None:
            rows = []
            self.backend.set_evaluation_sink(rows.append)
            self.recording.on = self.args.video
            env, returned, error, sim = self.run_program(codes[chosen], self.args.program_seconds)
            for _ in range(int(SETTLE_SECONDS / 0.5)):
                self.ctx.advance(self.backend.epoch, 0.5)
            self.recording.on = False
            self.recording.close()
            self.backend.set_evaluation_sink(None)
            result.update(execution={"error": error, "returned": str(returned)[:300], "sim_seconds": round(sim, 1),
                                      "log": env._log[-40:]}, **self.score(rows))
        else:
            result.update(success=False, stage=0.0, objects={})
        result["wall_seconds"] = round(time.monotonic() - STARTED, 1)
        (self.args.out / "result.json").write_text(json.dumps(result, indent=1, default=str))
        print(f"RESULT success={result['success']} stage={result['stage']} executed={chosen}", flush=True)

    def score(self, rows):
        spec = json.loads(self.args.targets.read_text())
        goals = ({s["fixture_id"]: s["evaluator_target"] for s in spec["steps"]} if "steps" in spec else spec["objects"])
        evaluator = runpy.run_path(str(Path(self.args.root) / "evaluation/evaluate.py"))
        verdict = evaluator["score_attempt"](rows, {"schema_version": 1, "frame": "ws_map", "objects": goals},
                                             require_settled=True)
        objects = {}
        for name, goal in goals.items():
            reason = verdict["objects"].get(name, {}).get("terminal_reason", "missing")
            start = rows[0]["objects"][name]["position"] if rows else None
            lifted = bool(rows) and any(r["objects"][name]["position"][2] > start[2] + LIFT_M and r["objects"][name]["hand_contacts"]
                                        for r in rows)
            end = np.asarray(rows[-1]["objects"][name]["position"]) if rows else None
            low, high = (np.asarray(b) for b in goal["bounds"])
            inside = end is not None and bool(np.all(end >= low) and np.all(end <= high))
            stage = 3 if reason is None else 2 if inside else 1 if lifted else 0
            objects[name] = {"stage": stage, "reason": reason}
        stage = sum(o["stage"] for o in objects.values()) / max(1, len(objects))
        return {"success": bool(verdict["success"]), "stage": round(stage, 3), "objects": objects,
                "untargeted": verdict.get("untargeted"), "invalid": verdict.get("invalid_reasons")}


STARTED = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--program-seconds", type=float, default=420)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.scene, args.targets, args.out = args.scene.resolve(), args.targets.resolve(), args.out.resolve()
    Task(args).main()


if __name__ == "__main__":
    main()
