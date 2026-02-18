from __future__ import annotations
from typing import Callable
import numpy as np

from src.classes.physics.transitions import Transitions
from src.classes.physics.transition_process import (
    TunnelingRecombination, TunnelingRetrapping, ConductionBandExcitation,
    RecombinationOutcome, RetrappingOutcome,
    EVENT_CODES,
)
from src.helper_functions import ArrayLike, _return_like_input
from src.classes.constants import cnst

# ── Filling Registry ──────────────────────────────────────────────────────────

@Transitions.register_fill("dose")
def build_fill_dose(D0: float) -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    """Dosing equation from A.Larsen et al. Radiation Measurements 44 (2009) 467-471
    which is equivalent to the dosing equation from
    G.E. King et al.  Quaternary Geochronology 33 (2016) 76-87 if N is fixed"""
    def f(N: ArrayLike, e: ArrayLike, D_dot: ArrayLike) -> ArrayLike:
        diff = N - e
        if diff <= 0:
            lam = 1e-20
        else:
            lam = (D_dot * diff / D0)
        return _return_like_input(N, np.array(lam))
    return f

@Transitions.register_fill("none")
def build_fill_none() -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    def f(N: ArrayLike, e: ArrayLike, D_dot) -> ArrayLike:
        return -1e20
    return f


# ── Process Registry ──────────────────────────────────────────────────────────
# Each builder receives raw physics kwargs and constructs both the rate
# callable and the TransitionProcess objects in a single step.
# State mutations are delegated to RecombinationOutcome / RetrappingOutcome.

@Transitions.register_process("ground_state_tunnel")
def _build_gs_tunnel_procs(alpha_GS: float, b: float,
                            retrap_pre_tun: float = 0.01, **kw):
    rate_fn = lambda r: b * np.exp(-alpha_GS * r)
    return [
        TunnelingRecombination("GS tunneling", rate_fn, "F1",
            outcome=RecombinationOutcome(EVENT_CODES["GS_tun_recom"])),
        TunnelingRetrapping("GS tunneling", rate_fn, "F1", retrap_pre_tun,
            outcome=RetrappingOutcome(EVENT_CODES["GS_tun_retrap"])),
    ]

@Transitions.register_process("excited_state_tunnel")
def _build_es_tunnel_procs(alpha: float, b: float,
                            retrap_pre_tun: float = 0.01, **kw):
    rate_fn = lambda r: b * np.exp(-alpha * r)
    return [
        TunnelingRecombination("ES tunneling", rate_fn, "F2",
            outcome=RecombinationOutcome(EVENT_CODES["ES_tun_recom"])),
        TunnelingRetrapping("ES tunneling", rate_fn, "F2", retrap_pre_tun,
            outcome=RetrappingOutcome(EVENT_CODES["ES_tun_retrap"])),
    ]

@Transitions.register_process("ground_state_to_cb")
def _build_gs_cb_proc(E_cb: float, s: float, mu: float,
                      retrap_pre_CB: float = 1.0, **kw):
    rate_fn = lambda T: s * np.exp(-E_cb / (cnst.k_b_ev * T))
    mob_fn  = lambda r: np.exp(-(r / mu) ** 2)
    return [ConductionBandExcitation("GS->CB", rate_fn, "F1", mob_fn,
                                     retrap_pre_CB,
                                     recom_outcome=RecombinationOutcome(EVENT_CODES["GS_CB_recom"]),
                                     retrap_outcome=RetrappingOutcome(EVENT_CODES["GS_CB_retrap"]))]

@Transitions.register_process("excited_state_to_cb")
def _build_es_cb_proc(E_loc: float, E_cb: float, s: float, mu: float,
                      retrap_pre_CB: float = 1.0, **kw):
    rate_fn = lambda T: s * np.exp(-(E_cb - E_loc) / (cnst.k_b_ev * T))
    mob_fn  = lambda r: np.exp(-(r / mu) ** 2)
    return [ConductionBandExcitation("ES->CB", rate_fn, "F2", mob_fn,
                                     retrap_pre_CB,
                                     recom_outcome=RecombinationOutcome(EVENT_CODES["ES_CB_recom"]),
                                     retrap_outcome=RetrappingOutcome(EVENT_CODES["ES_CB_retrap"]))]
