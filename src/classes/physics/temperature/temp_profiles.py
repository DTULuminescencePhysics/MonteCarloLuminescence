from __future__ import annotations
import numpy as np 
from typing import Iterable, Callable
from src.helper_functions import ArrayLike, _as_1d, _return_like_input
from src.classes.physics.temperature.temp_profile_class import TimeTempProfile


# Constant temperature
@TimeTempProfile.register("constant")
def _build_constant(T0: float) -> Callable[[ArrayLike], ArrayLike]:
    """Produces a function that returns a constant temperature"""
    def f(t: ArrayLike) -> ArrayLike:
        return T0 
    return f

# Linear: T(t) = T0 + slope * t   (slope < 0 → cooling)
@TimeTempProfile.register("linear")
def _build_linear(T0: float, dT: float) -> Callable[[ArrayLike], ArrayLike]:
    def f(t: ArrayLike) -> ArrayLike:
        """Produces a function that is changing linearly"""
        tt = _as_1d(t)
        out = T0 - dT * tt
        return _return_like_input(t, out)
    return f


@TimeTempProfile.register("linearsteps")
def build_piecewise_linear_with_eps(times: ArrayLike, temps: ArrayLike) -> Callable[[ArrayLike], ArrayLike]:
    """Produces a function that can output a complete temperature profile with linear changes and constant
    temperatures"""
    t_min =   times[0]
    t_max =   times[-1]
    T_left =  temps[0]
    T_right = temps[-1]

    def f(t: ArrayLike) -> ArrayLike:
        tt = _as_1d(t)

        out = np.interp(tt, times, temps)  
        out = np.where(tt < t_min, T_left, out)
        out = np.where(tt > t_max, T_right, out)

        return _return_like_input(t, out)

    return f





# 2) Single step down at time t_step by dT
@TimeTempProfile.register("step")
def _build_step(T0: float, times: float, dT_step: float) -> Callable[[ArrayLike], ArrayLike]:
    def f(t: ArrayLike) -> ArrayLike:
        tt = _as_1d(t)
        out = T0 - dT_step * np.heaviside(tt - times, 1.0)
        return _return_like_input(t, out)
    return f

# 3) Multiple steps (sizes > 0 means step down)
@TimeTempProfile.register("steps")
def _build_steps(T0: float, times: Iterable[float], 
                 dT_step: Iterable[float]) -> Callable[[ArrayLike], ArrayLike]:
    times = np.asarray(list(times), float)
    dT_step = np.asarray(list(dT_step), float)
    def f(t: ArrayLike) -> ArrayLike:
        tt = _as_1d(t)                                 # (T,)
        H = np.heaviside(tt[None, :] - times[:, None], 1.0)  # (S, T)
        out = T0 - (dT_step[:, None] * H).sum(axis=0)    # (T,)
        return _return_like_input(t, out)
    return f