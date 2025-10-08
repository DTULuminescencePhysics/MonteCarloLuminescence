from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Union, ClassVar
import inspect
import numpy as np

ArrayLike = Union[float, np.ndarray]
Builder = Callable[..., Callable[[ArrayLike], ArrayLike]]


@dataclass
class TimeTempProfile:
    _f: Callable[[ArrayLike], ArrayLike]

    # ----- Registry -----
    _REGISTRY: ClassVar[Dict[str, Builder]] = {} 

    def __call__(self, t: ArrayLike) -> ArrayLike:
        return self._f(t)

    def __init__(self, kind: str, **kwargs):
        kind = kind.lower()
        try:
            builder = self._REGISTRY[kind]
        except KeyError as e:
            raise ValueError(
                f"Unknown profile kind '{kind}'. Available: {sorted(self._REGISTRY)}"
            ) from e
        
        sig = inspect.signature(builder)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # object.__setattr__(self, "_f", builder(**filtered))
        self._f = builder(**filtered)

    # Decorator to register new kinds
    @classmethod
    def register(cls, kind: str) -> Callable[[Builder], Builder]:
        kind = kind.lower()
        def decorator(fn: Builder) -> Builder:
            if not callable(fn):
                raise TypeError("Registered object must be callable.")
            cls._REGISTRY[kind] = fn
            return fn
        return decorator

    @classmethod
    def available_kinds(cls):
        return sorted(cls._REGISTRY)
    

   