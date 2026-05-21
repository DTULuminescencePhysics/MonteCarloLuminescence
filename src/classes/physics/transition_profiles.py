from __future__ import annotations
from typing import Callable
import numpy as np

from src.classes.physics.transitions import Transitions
from src.classes.physics import transition_process as tp
from src.classes.physics.transition_process import EVENT_CODES
from src.helper_functions import ArrayLike, _return_like_input
from src.classes.constants import cnst


# Fill registry

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
            # lam = D0 / D_dot * np.log(diff/(diff-1))
        return _return_like_input(N, np.array(lam))
    return f

@Transitions.register_fill("none")
def build_fill_none() -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
    def f(N: ArrayLike, e: ArrayLike, D_dot) -> ArrayLike:
        return -1e20
    return f


# Process registry

@Transitions.register_process("ground_state_tunnel")
def _build_gs_tunnel_procs(alpha_GS: float, b: float, R_tun: float = 0.01,
                            VRH: bool = False, **kw):
    rate_fn = lambda r: b * np.exp(-alpha_GS * r)
    energy_key = "dg_ee_E" if VRH else None
    return [
        tp.TunnelingRecombination("GS tunneling", rate_fn, "F1",
            operation=tp.RecombinationOperation(EVENT_CODES["GS_tun_recom"])),
        tp.TunnelingRetrapping("GS tunneling", rate_fn, "F1", R_tun,
            operation=tp.RetrappingOperation(EVENT_CODES["GS_tun_retrap"]),
            energy_key=energy_key),
    ]

@Transitions.register_process("excited_state_tunnel")
def _build_es_tunnel_procs(alpha_ES: float, b: float, R_tun: float = 0.01,
                          VRH: bool = False, **kw):
    rate_fn = lambda r: b * np.exp(-alpha_ES * r)
    energy_key = "de_ee_E" if VRH else None
    return [
        tp.TunnelingRecombination("ES tunneling", rate_fn, "F2",
            operation=tp.RecombinationOperation(EVENT_CODES["ES_tun_recom"])),
        tp.TunnelingRetrapping("ES tunneling", rate_fn, "F2", R_tun,
            operation=tp.RetrappingOperation(EVENT_CODES["ES_tun_retrap"]),
            energy_key=energy_key),
    ]

@Transitions.register_process("ground_state_to_cb")
def _build_gs_cb_proc(s: float, mu: float, R_CB: float = 1.0, **kw):
    rate_fn = lambda box: s * np.exp(-box.E_cb_occ / (cnst.k_b_ev * box.T))
    mob_fn  = lambda r: np.exp(-(r / mu) ** 2)
    return [tp.ConductionBandExcitation("GS->CB", rate_fn, "F1", mob_fn, R_CB,
                                     recom_operation=tp.RecombinationOperation(EVENT_CODES["GS_CB_recom"]),
                                     retrap_operation=tp.RetrappingOperation(EVENT_CODES["GS_CB_retrap"]))]

@Transitions.register_process("excited_state_to_cb")
def _build_es_cb_proc(s: float, mu: float, R_CB: float = 1.0, **kw):
    rate_fn = lambda box: s * np.exp(-(box.E_cb_occ - box.E_loc_occ) / (cnst.k_b_ev * box.T))
    mob_fn  = lambda r: np.exp(-(r / mu) ** 2)
    return [tp.ConductionBandExcitation("ES->CB", rate_fn, "F2", mob_fn, R_CB,
                                     recom_operation=tp.RecombinationOperation(EVENT_CODES["ES_CB_recom"]),
                                     retrap_operation=tp.RetrappingOperation(EVENT_CODES["ES_CB_retrap"]))]


@Transitions.register_process("band_tail")
def _build_bt_procs(b_BT: float, alpha_BT: float, **kw):
    """Six processes implementing band-tail transitions via shallow defects.

    Two deep-pool sub-channels for deep -> shallow excitation (GS / ES on
    the deep side, weighted by F1 / F2), one shallow-pool tunnelling-retrap,
    one shallow-pool radiative recombination, and two shallow-pool
    sub-channels for shallow -> deep de-excitation (GS / ES on the deep
    target side).
    """
    rate_fn = lambda r: b_BT * np.exp(-alpha_BT * r)

    op_d_to_s    = tp.DeepToShallowOperation(EVENT_CODES["Deep_to_Shallow"])
    op_s_to_s    = tp.ShallowToShallowOperation(EVENT_CODES["Shallow_to_Shallow"])
    op_s_recom   = tp.ShallowRecombineOperation(EVENT_CODES["Shallow_tun_recom"])
    op_s_to_d    = tp.ShallowToDeepOperation(EVENT_CODES["Shallow_to_Deep"])

    return [
        tp.DeepToShallow("Deep_GS->Shallow", rate_fn, "F1",
                      energy_key="dE_t_sh_unocc_GS",
                      operation=op_d_to_s, source_pool="deep"),
        tp.DeepToShallow("Deep_ES->Shallow", rate_fn, "F2",
                      energy_key="dE_t_sh_unocc_ES",
                      operation=op_d_to_s, source_pool="deep"),
        tp.ShallowTunRetrap("Shallow->Shallow", rate_fn,
                         operation=op_s_to_s, source_pool="shallow"),
        tp.ShallowTunRecom("Shallow->Hole", rate_fn,
                        operation=op_s_recom, source_pool="shallow"),
        tp.ShallowToDeep("Shallow->Deep_GS", rate_fn,
                      energy_key="dE_sh_t_unocc_GS",
                      operation=op_s_to_d, source_pool="shallow"),
        tp.ShallowToDeep("Shallow->Deep_ES", rate_fn,
                      energy_key="dE_sh_t_unocc_ES",
                      operation=op_s_to_d, source_pool="shallow"),
    ]
