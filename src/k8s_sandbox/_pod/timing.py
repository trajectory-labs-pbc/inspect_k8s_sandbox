from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PodOperationTiming:
    connect_s: float
    command_s: float


POD_OPERATION_TIMING: ContextVar[PodOperationTiming | None] = ContextVar(
    "pod_operation_timing", default=None
)
