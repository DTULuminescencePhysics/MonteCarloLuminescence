from __future__ import annotations
import numpy as np 
from typing import Iterable, Callable
from src.helper_functions import ArrayLike, _as_1d, _return_like_input
from src.classes.physics.temperature.temp_profile_class import TimeTempProfile




# 1) Constant temperature
@TimeTempProfile.register("constant")
def _build_constant(T0: float) -> Callable[[ArrayLike], ArrayLike]:
    def f(t: ArrayLike) -> ArrayLike:
        tt = _as_1d(t)
        out = np.full_like(tt, T0, dtype=float)
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

# 4) Linear: T(t) = T0 + slope * t   (slope < 0 → cooling)
@TimeTempProfile.register("linear")
def _build_linear(T0: float, dT: float) -> Callable[[ArrayLike], ArrayLike]:
    def f(t: ArrayLike) -> ArrayLike:
        tt = _as_1d(t)
        out = T0 - dT * tt
        return _return_like_input(t, out)
    return f

# 5) Linear steps: at given starts; each adds an instant drop and then linear rate
@TimeTempProfile.register("linearsteps")
def _build_linear_steps(T0: float, times: Iterable[float],dT_step: Iterable[float],
                        dT: Iterable[float], t0: float = 0.0) -> Callable[[ArrayLike], ArrayLike]:
    
    if isinstance(times, np.ndarray) and times.size == 0: 
        r = np.asarray(dT, dtype=float)
        
        if r[0] == 0:
            def f(t: ArrayLike) -> ArrayLike:
                tt = _as_1d(t)
                out = np.full_like(tt, T0, dtype=float)
                return _return_like_input(t, out)
        else: 
            def f(t: ArrayLike) -> ArrayLike:
                tt = _as_1d(t)
                out = T0 - dT * tt
                return _return_like_input(t, out)
        
        return f

    elif isinstance(times, float) or isinstance(times, np.floating):
        s = np.asarray(times, dtype=float)
        a = np.asarray(dT_step, dtype=float)
        r = np.asarray(dT,      dtype=float)
    else:
        s = np.asarray(list(times), dtype=float)
        a = np.asarray(list(dT_step), dtype=float)
        r = np.asarray(list(dT),      dtype=float)

    if s.size != a.size:
        raise ValueError("dT_step must have the same length as times")
    if r.size != s.size + 1:
        raise ValueError("dT must have length len(times) + 1")
    if np.any(a < 0):
        raise ValueError("temp_step must be nonnegative")

  
    if s.size:
        order = np.argsort(s)
        s = s[order]
        a = a[order]

    seg_times = np.concatenate(([t0], s))                   
    seg_ends   = np.concatenate((s, [np.inf]))             
    seg_lengths = seg_ends - seg_times                 

    def f(t: ArrayLike) -> ArrayLike:
        tt = _as_1d(t)                                       

        dt = tt[None, :] - seg_times[:, None]              
        time_in_section = np.clip(dt, 0.0, seg_lengths[:, None])  
        linear_drop = (r[:, None] * time_in_section).sum(axis=0) 

        if s.size:
            H = np.heaviside(tt[None, :] - s[:, None], 1.0)  
            step_drop = (a[:, None] * H).sum(axis=0)     
        else:
            step_drop = np.zeros_like(tt)

        out = T0 - step_drop - linear_drop
        return _return_like_input(t, out)

    return f

# 6) Exponential: T(t) = T_inf + (T0 - T_inf) * exp(-k t)
@TimeTempProfile.register("exponential")
def _build_exponential(T0: float, T_inf: float, k: float) -> Callable[[ArrayLike], ArrayLike]:
    def f(t: ArrayLike) -> ArrayLike:
        tt = _as_1d(t)
        out = T_inf + (T0 - T_inf) * np.exp(-k * tt)
        return _return_like_input(t, out)
    return f

# 7) Linear drop events
# @TimeTempProfile.register("lineardrops")
# def _build_linear_drop_profile(T0: float, times: Iterable[float], 
#                                dT_step: Iterable[float],dT: Iterable[float]) -> Callable[[ArrayLike], ArrayLike]:
#     s = np.asarray(list(times), dtype=float)
#     a = np.asarray(list(dT_step), dtype=float)
#     r = np.asarray(list(dT),  dtype=float)
#     if not (len(s) == len(a) == len(r)):
#         raise ValueError("times, amounts, dT must have the same length")
#     if np.any(a < 0) or np.any(r < 0):
#         raise ValueError("amounts and dT must be nonnegative")

#     def f(t: ArrayLike) -> ArrayLike:
#         tt = _as_1d(t)                
#         dt = tt[None, :] - s[:, None] 
#         ramp = np.maximum(0.0, dt) * r[:, None]
#         clipped = np.minimum(ramp, a[:, None])
#         out = T0 - clipped.sum(axis=0) 
#         return _return_like_input(t, out)
#     return f

# 7) Linear drop events
@TimeTempProfile.register("lineardrops")
def _build_linear_drop_profile(T0: float, times: Iterable[float], 
                               dT_step: Iterable[float], dT: Iterable[float]) -> Callable[[ArrayLike], ArrayLike]:
    s = np.asarray(list(times), dtype=float)
    a = np.asarray(list(dT_step), dtype=float)
    r = np.asarray(list(dT), dtype=float)

    if s.size == 0:
        return _build_constant(T0)

    if not (len(s) == len(a) == len(r)):
        raise ValueError("times, amounts, dT must have the same length")
    if np.any(a < 0) or np.any(r < 0):
        raise ValueError("amounts and dT must be nonnegative")

    order = np.argsort(s)
    s = s[order]
    a = a[order]
    r = r[order]

    if np.any(np.diff(s) <= 0):
        raise ValueError("times must be strictly increasing")

    rates = r.copy()
    for idx in range(s.size):
        drop = a[idx]
        start = s[idx]
        next_time = s[idx + 1] if idx + 1 < s.size else np.inf
        rate = rates[idx]

        if drop == 0.0:
            rates[idx] = 0.0
            continue

        if not np.isfinite(rate) or rate == 0.0:
            if np.isfinite(next_time):
                duration = next_time - start
                if duration <= 0.0:
                    raise ValueError("times must be strictly increasing")
                rates[idx] = drop / duration
            else:
                raise ValueError("Final segment with non-zero drop requires a positive rate")
            rate = rates[idx]

        duration = drop / rate
        if np.isfinite(next_time) and duration > (next_time - start) + 1e-9:
            raise ValueError(
                f"Rate {rate} is too small to reach the requested drop {drop} before the next time point"
            )

    def f(t: ArrayLike) -> ArrayLike:
        tt = _as_1d(t)
        dt = tt[None, :] - s[:, None]
        ramp = np.maximum(0.0, dt) * rates[:, None]
        clipped = np.minimum(ramp, a[:, None])
        out = T0 - clipped.sum(axis=0)
        return _return_like_input(t, out)
   
    return f
