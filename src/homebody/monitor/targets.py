"""What the monitor's models take and give: the vocabulary model's request and answer, the decision model's
request and reading, and the verdict the rules draw from readings. All frozen; all recorded as JSON.

The monitor sees what the agent may see: the task, the plan, the robot's own skill and stage names, its hands and
its camera images; never object poses or evaluator targets.
"""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

KINDS = ("progress", "success", "failure")  # what a dynamic label is
NONE_OF_THESE = "none_of_these"  # always offered, so probability is never forced onto the nearest label
STATIC = (NONE_OF_THESE, "unsafe", "human_needed")
PROGRESS_LEVELS = ("no_change", "some", "clear")
PHASES = ("idle", "planning", "tool_call")
ACTIONS = ("continue", "flag", "abort_skill", "reopen_goal")


@dataclass(frozen=True)
class Label:
    """One action the robot may be seen doing for a goal: its kind, how it looks and the skill stages it matches."""
    name: str
    kind: str
    description: str
    stages: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    """A visible fact a done goal made true that must stay true, e.g. "the cup is in the left hand"."""
    goal_id: str
    text: str


@dataclass(frozen=True)
class Vocabulary:
    """A goal's dynamic labels and standing claims; `source` is "model" or "fallback"."""
    goal_id: str
    labels: tuple[Label, ...]
    standing: tuple[Claim, ...] = ()
    expected_seconds: float | None = None
    source: str = "model"

    def label(self, name: str) -> Label | None:
        return next((label for label in self.labels if label.name == name), None)

    def failures(self) -> tuple[str, ...]:
        return tuple(label.name for label in self.labels if label.kind == "failure")


@dataclass(frozen=True)
class NodeInfo:
    """A goal of the plan as the monitor reads it; `context` is the path of node types above it."""
    goal_id: str
    goal: str
    check: dict | None = None
    status: str = "pending"
    budget_left: int | None = None
    context: str = ""


@dataclass(frozen=True)
class Phase:
    """What the harness is doing: idle, planning (the agent decides) or tool_call (a skill runs, at a stage);
    `call` numbers the skill calls of a task, so a reading can stop only the call it watched."""
    name: str = "idle"
    skill: str | None = None
    stage: str | None = None
    call: int = 0


@dataclass(frozen=True)
class Shot:
    """One captured camera image at a simulation time."""
    rgb: np.ndarray = field(repr=False, compare=False)
    sim_time: float
    view: str = "head"


@dataclass(frozen=True)
class VocabularyRequest:
    """What the vocabulary model reads when a goal opens or a skill starts in it."""
    task: str
    tree: str
    node: NodeInfo
    done: tuple[NodeInfo, ...]
    skills: tuple[str, ...]
    hands: str
    images: tuple[Path, ...] = ()
    stages: dict = field(default_factory=dict)  # skill -> stage names seen this session
    skill: str | None = None
    arguments: dict | None = None
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionRequest:
    """One decision call: the newest frames with the task, the goal, the phase, the hands and the vocabulary."""
    task: str
    node: NodeInfo
    phase: Phase
    hands: str
    vocabulary: Vocabulary
    shots: tuple[Shot, ...]


@dataclass(frozen=True)
class Reading:
    """A decision call's answers as probabilities. A question the model refused is None (or absent) and named in
    `refusals`."""
    goal_id: str
    sim_time: float
    activity: dict = field(default_factory=dict)  # label -> probability
    unsafe: float | None = None
    human_needed: float | None = None
    still_true: dict = field(default_factory=dict)  # goal id -> probability its claim still holds
    progress: dict = field(default_factory=dict)  # level -> probability
    refusals: tuple[str, ...] = ()
    phase: Phase = field(default_factory=Phase)
    latency_s: float = 0.0
    lag_s: float = 0.0
    input_tokens: int = 0
    cost_usd: float = 0.0

    def top(self) -> tuple[str | None, float]:
        """The most probable activity label and its probability."""
        if not self.activity:
            return None, 0.0
        name = max(self.activity, key=lambda label: self.activity[label])
        return name, self.activity[name]


@dataclass(frozen=True)
class Verdict:
    """What the rules decided from the readings so far, for one goal."""
    action: str
    goal_id: str | None
    label: str | None = None
    p: float = 0.0
    reason: str = ""
    sim_time: float = 0.0

    def text(self) -> str:
        """The verdict as the agent and the plan read it."""
        return f"{self.label} p={self.p:.2f}" + (f": {self.reason}" if self.reason else "")


def request_json(request: VocabularyRequest, root: Path) -> dict:
    """REQUEST as recorded: images as paths relative to ROOT (the session folder) where they lie inside it."""
    data = asdict(request)
    data["images"] = [str(Path(image).relative_to(root)) if Path(image).is_relative_to(root) else str(image)
                      for image in request.images]
    return data


def node_from(data: dict) -> NodeInfo:
    return NodeInfo(**data)


def request_from(data: dict, root: Path) -> VocabularyRequest:
    from homebody.rollout.paths import inside  # a recorded request's images stay inside its rollout

    """A recorded request (request_json) read back, its images resolved against ROOT."""
    return VocabularyRequest(**{**data, "node": node_from(data["node"]),
                                "done": tuple(node_from(node) for node in data.get("done", ())),
                                "skills": tuple(data.get("skills", ())),
                                "images": tuple(path for image in data.get("images", ())
                                                if (path := inside(root, image)) is not None),
                                "findings": tuple(data.get("findings", ()))})


def reading_from(data: dict) -> Reading:
    """A line of readings.jsonl read back (its `action` field is the verdict's, not the reading's)."""
    names = {item.name for item in fields(Reading)}
    values = {key: value for key, value in data.items() if key in names}
    return Reading(**{**values, "phase": Phase(**values.get("phase", {})),
                      "refusals": tuple(values.get("refusals", ()))})


def verdict_from(data: dict) -> Verdict:
    return Verdict(**{key: value for key, value in data.items() if key in {item.name for item in fields(Verdict)}})
