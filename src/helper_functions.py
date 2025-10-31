from __future__ import annotations
from typing import Callable, Dict, Any, Union
from numpy import ndarray, atleast_1d, asarray, ndim
import inspect 

ArrayLike = Union[float, ndarray]
Builder = Callable[..., Callable[[ArrayLike], ArrayLike]]
Builder2 = Callable[..., Callable[[ArrayLike, ArrayLike], ArrayLike]]
Builder3 = Callable[..., Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]



def _filter_kwargs(builder: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs the builder accepts."""
    sig = inspect.signature(builder)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def _as_1d(t: ArrayLike) -> ndarray:
    """Convert any scalar/array input to a 1D float array for internal math."""
    return atleast_1d(asarray(t, dtype=float))

def _return_like_input(t_in: ArrayLike, out_1d: ndarray) -> ArrayLike:
    """Return a scalar if input was scalar; otherwise the 1D array."""
    return out_1d.item() if ndim(t_in) == 0 else out_1d