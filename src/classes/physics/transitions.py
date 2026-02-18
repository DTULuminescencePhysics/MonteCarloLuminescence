from __future__ import annotations
from typing import Callable, ClassVar, Dict, List
from dataclasses import dataclass, field
from src.helper_functions import Builder3, ArrayLike, _filter_kwargs


# Transition registry
@dataclass
class Transitions:

    _fill: Callable[..., ArrayLike] = field(init=False, repr=False)

    FILL_REGISTRY: ClassVar[Dict[str, Builder3]] = {}
    PROCESS_REGISTRY: ClassVar[Dict[str, Callable]] = {}


    def __init__(self, fill_kind: str, tran_kind: List[str], **kwargs):

        fill_kind = fill_kind.lower()
        if fill_kind not in self.FILL_REGISTRY:
            raise ValueError(f"Unknown fill kind '{fill_kind}'. Available: {sorted(self.FILL_REGISTRY)}")

        for tran in tran_kind:
            if tran not in self.PROCESS_REGISTRY:
                raise ValueError(f"Unknown transition kind '{tran}'. Available: {sorted(self.PROCESS_REGISTRY.keys())}")

        f_builder = self.FILL_REGISTRY[fill_kind]
        object.__setattr__(self, "_fill", f_builder(**_filter_kwargs(f_builder, kwargs)))

    @classmethod
    def register_fill(cls, kind: str) -> Callable[[Builder3], Builder3]:
        kind = kind.lower()
        def deco(fn: Builder3) -> Builder3:
            if not callable(fn):
                raise TypeError("fill builder must be callable")
            cls.FILL_REGISTRY[kind] = fn
            return fn
        return deco

    @classmethod
    def register_process(cls, kind: str) -> Callable:
        kind = kind.lower()
        def deco(fn: Callable) -> Callable:
            if not callable(fn):
                raise TypeError("process builder must be callable")
            cls.PROCESS_REGISTRY[kind] = fn
            return fn
        return deco

    def build_processes(self, tran_kind: List[str], **kwargs) -> list:
        """Build the ordered list of active TransitionProcess instances
        from the registered process builders."""
        processes: list = []
        for tran_name in tran_kind:
            if tran_name not in self.PROCESS_REGISTRY:
                continue
            builder = self.PROCESS_REGISTRY[tran_name]
            result = builder(**_filter_kwargs(builder, kwargs))
            if isinstance(result, list):
                processes.extend(result)
            else:
                processes.append(result)
        return processes

    @classmethod
    def available_physics(cls) -> dict[str, list[str]]:
        return {
            "fill":    sorted(cls.FILL_REGISTRY),
            "process": sorted(cls.PROCESS_REGISTRY),
        }
