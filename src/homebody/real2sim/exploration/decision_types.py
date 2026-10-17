"""Frozen requests yielded by exploration policy to the Session owner."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    x: float
    y: float
    yaw: float
    rejection: str = ""
    continuation: dict | None = None


@dataclass(frozen=True)
class Recovery:
    record: dict


@dataclass(frozen=True)
class ActionRequest:
    name: str
    arguments: dict


@dataclass(frozen=True)
class PrepareAction:
    name: str
    arguments: dict


@dataclass(frozen=True)
class DecisionContext:
    images: tuple
    prompt: str
    on_partial: object
    number: int
    tools: tuple
    limits: dict
    transforms: tuple
    permission: dict
    generation: int
