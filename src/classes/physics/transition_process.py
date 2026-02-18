from __future__ import annotations
from typing import Protocol, Callable, runtime_checkable, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from src.helper_functions import ArrayLike

if TYPE_CHECKING:
    from src.classes.physics.crystal import Box


# Even codes >= 2 are recombination (luminescence); odd codes are retrapping.
# Code 0 = no event / max_dt step; Code 1 = fill (trap_new_electron), etc.

EVENT_CODES = {
    "no_event":         0,
    "fill":             1,
    "GS_tun_recom":     2,
    "GS_tun_retrap":    3,
    "ES_tun_recom":     4,
    "ES_tun_retrap":    5,
    "GS_CB_recom":      6,
    "GS_CB_retrap":     7,
    "ES_CB_recom":      8,
    "ES_CB_retrap":     9,
}

EVENT_NAMES = {v: k for k, v in EVENT_CODES.items()}    # Reverse eveent codes


def is_luminescence(code: int) -> bool:
    """Return True if the event code corresponds to an event where charge carriers annihilate."""
    return code >= 2 and code % 2 == 0


@runtime_checkable
class TransitionProcess(Protocol):
    """
    Protocol for a single transition channel in the MC simulation.

    Each process knows:
      1. How to compute its rate array for one electron (rates)
      2. How to compute summed rates for ALL electrons at once (bulk_rates_sum)
      3. How to execute the state change when selected (execute)
    """

    name: str

    def rates(self, box: Box, exec_index: int) -> np.ndarray:
        """Return 1-D rate array for the electron at *exec_index*.

        For distance-dependent channels this returns one rate per target
        (hole or unoccupied trap). For scalar channels (CB excitation)
        this returns a 1-element array. If inapplicable, return empty array.
        """
        pass

    def bulk_rates_sum(self, box: Box) -> np.ndarray:
        """Return shape-(n_occupied_traps,) array: sum of rates across all
        targets for every occupied electron.  Used by select_electron for
        vectorised performance."""
        pass

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        """Mutate *box* state for the selected event.

        exec_index : index into occupied traps (same as passed to rates)
        local_idx  : which element within this process's rate array was chosen
        """
        pass


@dataclass(slots=True)
class TunnelingRecombination:
    """Distance-dependent tunneling recombination with a hole."""

    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]
    pop_factor_key: str                               # "F1"/"F2" depending on GS/ES
    event_code: int = EVENT_CODES["GS_tun_recom"]

    def rates(self, box: Box, exec_index: int) -> np.ndarray:
        if box.d.size == 0:
            return np.empty(0)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d[exec_index, :])

    def bulk_rates_sum(self, box: Box) -> np.ndarray:
        if box.d.size == 0:
            return np.zeros(box.t_cnt)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d).sum(axis=1)

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        t_index = np.flatnonzero(box.occ_trap)[exec_index]
        h_index = np.flatnonzero(box.occ_hole)[local_idx]
        box.occ_hole[h_index] = 0
        box.occ_trap[t_index] = 0
        box.t_cnt -= 1
        box.h_cnt -= 1
        box.event_code = self.event_code


@dataclass(slots=True)
class TunnelingRetrapping:
    """Distance-dependent tunneling retrapping into an unoccupied trap."""

    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]
    pop_factor_key: str
    pre_factor: float
    event_code: int = EVENT_CODES["GS_tun_retrap"]

    def rates(self, box: Box, exec_index: int) -> np.ndarray:
        if box.d_ee.size == 0:
            return np.empty(0)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d_ee[exec_index, :]) * self.pre_factor

    def bulk_rates_sum(self, box: Box) -> np.ndarray:
        if box.d_ee.size == 0:
            return np.zeros(box.t_cnt)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d_ee).sum(axis=1) * self.pre_factor

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        t_index = np.flatnonzero(box.occ_trap)[exec_index]
        dest_index = np.where(box.occ_trap == 0)[0][local_idx]
        box.occ_trap[t_index] = 0
        box.occ_trap[dest_index] = 1
        box.event_code = self.event_code


@dataclass(slots=True)
class ConductionBandExcitation:
    """Thermal excitation to the conduction band, followed by a
    distance-dependent sub-selection (recombination vs retrapping)
    using CB mobility."""

    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]    # _gs_con or _es_con
    pop_factor_key: str
    cb_mob_fn: Callable[[ArrayLike], ArrayLike]  # _cb_mob
    retrap_pre_CB: float
    recom_event_code: int = EVENT_CODES["GS_CB_recom"]
    retrap_event_code: int = EVENT_CODES["GS_CB_retrap"]

    def rates(self, box: Box, exec_index: int) -> np.ndarray:
        F = getattr(box, self.pop_factor_key)
        return np.atleast_1d(F * self.rate_fn(box.T))

    def bulk_rates_sum(self, box: Box) -> np.ndarray:
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.T) * np.ones(box.t_cnt)

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        t_index = np.flatnonzero(box.occ_trap)[exec_index]

        cb_rates = np.array([], dtype=float)
        n_recom = 0

        if box.d.size != 0:
            recom_rates = self.cb_mob_fn(box.d[exec_index, :])
            cb_rates = np.append(cb_rates, recom_rates)
            n_recom = recom_rates.size

        if box.d_ee.size != 0:
            retrap_rates = (self.cb_mob_fn(box.d_ee[exec_index, :])
                            * self.retrap_pre_CB)
            cb_rates = np.append(cb_rates, retrap_rates)

        prob = np.cumsum(cb_rates) / np.sum(cb_rates)
        idx2 = int(np.min(np.argwhere(prob >= box.rng.random())))

        if idx2 < n_recom:
            h_index = np.flatnonzero(box.occ_hole)[idx2]
            box.occ_trap[t_index] = 0
            box.occ_hole[h_index] = 0
            box.t_cnt -= 1
            box.h_cnt -= 1
            box.event_code = self.recom_event_code
        else:
            dest_idx = np.where(box.occ_trap == 0)[0][idx2 - n_recom]
            box.occ_trap[t_index] = 0
            box.occ_trap[dest_idx] = 1
            box.event_code = self.retrap_event_code
