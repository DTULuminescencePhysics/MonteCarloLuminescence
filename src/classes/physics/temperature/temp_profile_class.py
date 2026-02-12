from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, ClassVar

from src.helper_functions import ArrayLike, Builder, _filter_kwargs

@dataclass
class TimeTempProfile:
    _f: Callable[[ArrayLike], ArrayLike] = field(init=False, repr=False)

    # ----- Registry -----
    _REGISTRY: ClassVar[Dict[str, Builder]] = {} 

    def __call__(self, t: ArrayLike) -> ArrayLike:
        return self._f(t)

    def __init__(self, **kwargs):
        kind = kwargs.pop("kind")
        if kind is None: 
            kind = 'linearsteps'
        kind = kind.lower()
        try:
            builder = self._REGISTRY[kind]
        except KeyError as e:
            raise ValueError(
                f"Unknown profile kind '{kind}'. Available: {sorted(self._REGISTRY)}"
            ) from e
        filtered = _filter_kwargs(builder, kwargs or {})
       
        object.__setattr__(self, "_f", builder(**filtered))

    @classmethod
    def register(cls, kind: str | None) -> Callable[[Builder], Builder]:
        if kind is None: 
            kind = 'linearsteps'
        kind = kind.lower()
        def decorator(fn: Builder) -> Builder:
            if not callable(fn):
                raise TypeError("Registered object must be callable.")
            cls._REGISTRY[kind] = fn
            return fn
        return decorator

    @classmethod
    def available_temperature_profiles(cls):
        return sorted(cls._REGISTRY)
    

   