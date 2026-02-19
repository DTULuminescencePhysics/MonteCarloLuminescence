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

EVENT_NAMES = {v: k for k, v in EVENT_CODES.items()}    # Reverse event codes


def is_luminescence(code: int) -> bool:
    """Return True if the event code corresponds to an event where charge carriers annihilate."""
    return code >= 2 and code % 2 == 0


# Protocol for operations on trapped charge carriers

@runtime_checkable
class Operation(Protocol):
    """Protocol for a terminal state mutation after a transition is selected."""
    event_code: int

    def execute(self, box: Box, t_index: int, local_index: int) -> None:
        """
        Mutate box state.

        t_index     : absolute index into occ_trap (already resolved)
        local_index : index of the target sites into the operation
                    (occupied holes for recombination, empty traps for retrapping)
        """
        pass


# Concrete classes for operation on charge carriers

@dataclass(slots=True)
class RecombinationOperation:
    """Electron and hole annihilate; both are removed from the crystal."""
    event_code: int

    def execute(self, box: Box, t_index: int, local_index: int) -> None:
        h_index = np.flatnonzero(box.occ_hole)[local_index]
        box.occ_hole[h_index] = 0
        box.occ_trap[t_index] = 0
        box.t_cnt -= 1
        box.h_cnt -= 1
        box.event_code = self.event_code


@dataclass(slots=True)
class RetrappingOperation:
    """Electron moves from its current trap to an unoccupied trap."""
    event_code: int

    def execute(self, box: Box, t_index: int, local_index: int) -> None:
        dest_index = np.where(box.occ_trap == 0)[0][local_index]
        box.occ_trap[t_index] = 0
        box.occ_trap[dest_index] = 1
        box.event_code = self.event_code


# Protocol for all transition processes

@runtime_checkable
class TransitionProcess(Protocol):
    """
    Protocol for a single transition process in the MC simulation.

    Each transition possesses three member functions:
    rates(self, Box, int) -> Arraylike: Compute transistion rates for a given electron to all possible targets
    bulk_rates_sum(self, Box) -> Arraylike: Compute cumulative rates for all possible transition events
    execute(self, Box, int, int) -> None: Choose and execute a transition based on random number generation
    """

    name: str

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        """Return 1-D rate array for the electron at *exec_index*.

        For distance-dependent channels this returns one rate per target
        (hole or unoccupied trap). For scalar channels (CB excitation)
        this returns a 1-element array. If inapplicable, return empty array.
        """
        pass

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
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


# Concrete classes for transition processes

@dataclass(slots=True)
class TunnelingRecombination:
    """Distance-dependent tunneling recombination with a hole."""

    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]
    pop_factor_key: str                               # "F1"/"F2" depending on GS/ES
    operation: Operation

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        if box.d.size == 0:
            return np.empty(0)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d[exec_index, :])

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        if box.d.size == 0:
            return np.zeros(box.t_cnt)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d).sum(axis=1)

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        t_index = np.flatnonzero(box.occ_trap)[exec_index]
        self.operation.execute(box, t_index, local_idx)


@dataclass(slots=True)
class TunnelingRetrapping:
    """Distance-dependent tunneling retrapping into an unoccupied trap."""

    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]
    pop_factor_key: str
    pre_factor: float
    operation: Operation

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        if box.d_ee.size == 0:
            return np.empty(0)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d_ee[exec_index, :]) * self.pre_factor

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        if box.d_ee.size == 0:
            return np.zeros(box.t_cnt)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d_ee).sum(axis=1) * self.pre_factor

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        t_index = np.flatnonzero(box.occ_trap)[exec_index]
        self.operation.execute(box, t_index, local_idx)


@dataclass(slots=True)
class ConductionBandExcitation:
    """Thermal excitation to the conduction band, followed by a
    distance-dependent sub-selection (recombination vs retrapping)
    using CB mobility."""

    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]    # _gs_con or _es_con
    pop_factor_key: str
    cb_mob_fn: Callable[[ArrayLike], ArrayLike]  # CB mobility
    retrap_pre_CB: float
    recom_operation: Operation
    retrap_operation: Operation

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        F = getattr(box, self.pop_factor_key)
        return np.atleast_1d(F * self.rate_fn(box.T))

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.T) * np.ones(box.t_cnt)

    def _build_cb_rates(self, box: Box, exec_index: int):
        """Return (cb_rates_array, n_recom_entries) for CB sub-selection."""
        cb_rates = np.array([], dtype=float)
        n_recom = 0
        if box.d.size != 0:
            recom_rates = self.cb_mob_fn(box.d[exec_index, :])
            cb_rates = np.append(cb_rates, recom_rates)
            n_recom = recom_rates.size
        if box.d_ee.size != 0:
            retrap_rates = self.cb_mob_fn(box.d_ee[exec_index, :]) * self.retrap_pre_CB
            cb_rates = np.append(cb_rates, retrap_rates)
        return cb_rates, n_recom

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        t_index = np.flatnonzero(box.occ_trap)[exec_index]

        cb_rates, n_recom = self._build_cb_rates(box, exec_index)
        if cb_rates.size == 0:
            return

        prob = np.cumsum(cb_rates) / cb_rates.sum()
        idx2 = int(np.min(np.argwhere(prob >= box.rng.random())))

        if idx2 < n_recom:
            self.recom_operation.execute(box, t_index, idx2)
        else:
            self.retrap_operation.execute(box, t_index, idx2 - n_recom)
