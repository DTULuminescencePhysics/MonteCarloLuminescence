from __future__ import annotations
from typing import Callable, ClassVar, Dict, List
from dataclasses import dataclass, field
import numpy as np 
from functools import partial
from src.helper_functions import Builder, Builder3, Builder4

from src.classes.physics.system_physics import CrystalPhysics
from src.helper_functions import ArrayLike, _return_like_input, _filter_kwargs
from src.classes.constants import cnst


# Transition registry
@dataclass
class Transitions:

    _fill: Callable[..., ArrayLike] = field(init=False, repr=False)
    _gs_tun: Callable[..., ArrayLike] | None = field(init=False, repr=False)
    _es_tun: Callable[..., ArrayLike] | None = field(init=False, repr=False)
    _gs_con: Callable[..., ArrayLike] | None = field(init=False, repr=False)
    _es_con: Callable[..., ArrayLike] | None = field(init=False, repr=False)
    _cb_mob: Callable[..., ArrayLike] | None = field(init=False, repr=False)

    FILL_REGISTRY: ClassVar[Dict[str, Builder3]] = {}
    TRAN_REGISTRY: ClassVar[Dict[str, Builder4]] = {}
    PROCESS_REGISTRY: ClassVar[Dict[str, Callable]] = {}


    def __init__(self, fill_kind: str, tran_kind: List[str], **kwargs):
                 
        fill_kind = fill_kind.lower()
        if fill_kind not in self.FILL_REGISTRY:
            raise ValueError(f"Unknown fill kind '{fill_kind}'. Available: {sorted(self.FILL_REGISTRY)}")
        
        for tran in tran_kind:
            if tran not in self.TRAN_REGISTRY:
                raise ValueError(f"Unknown transition kind '{tran}'. Available: {sorted(self.TRAN_REGISTRY.keys())}")

        f_builder = self.FILL_REGISTRY[fill_kind]
        fi_kwargs = _filter_kwargs(f_builder, kwargs)
        object.__setattr__(self, "_fill", f_builder(**fi_kwargs))


        for tran in tran_kind:
            attr_name = self.TRAN_REGISTRY[tran][0]
            e_builder = self.TRAN_REGISTRY[tran][1]
            e_kwargs = _filter_kwargs(e_builder, kwargs)
            object.__setattr__(self, attr_name, e_builder(**e_kwargs))

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
    def register_tran(cls, kind: str, attr:str) -> Callable[[Builder], Builder]:
        kind = kind.lower()
        def deco(fn: Builder) -> Builder:
            if not callable(fn):
                raise TypeError("transition builder must be callable")
            cls.TRAN_REGISTRY[kind] = (attr, fn)
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
            attr_name = self.TRAN_REGISTRY[tran_name][0]
            rate_fn = getattr(self, attr_name)
            builder = self.PROCESS_REGISTRY[tran_name]
            result = builder(rate_fn=rate_fn, **_filter_kwargs(builder, kwargs))
            if isinstance(result, list):
                processes.extend(result)
            else:
                processes.append(result)
        return processes

    @classmethod
    def available_physics(cls) -> dict[str, list[str]]:
        return {
            "fill": sorted(cls.FILL_REGISTRY),
            "tran": sorted(cls.TRAN_REGISTRY),
            "process": sorted(cls.PROCESS_REGISTRY),
        }
