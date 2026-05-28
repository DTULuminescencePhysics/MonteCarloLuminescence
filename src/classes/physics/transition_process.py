from __future__ import annotations
from typing import Protocol, Callable, runtime_checkable, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from src.helper_functions import ArrayLike
from src.classes.constants import cnst

if TYPE_CHECKING:
    from src.classes.physics.crystal import Box

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
    "bleach":           10,
    "Deep_to_Shallow":           11,    # deep -> shallow
    "Shallow_to_Shallow":        12,    # shallow -> shallow
    "Shallow_to_Deep":           13,    # shallow -> deep
    "Shallow_tun_recom":         14,    # shallow -> hole
}

EVENT_NAMES = {v: k for k, v in EVENT_CODES.items()}    # Reverse event codes

# Luminescence events emit a photon (charge carriers annihilate).
LUMINESCENCE_CODES: set[int] = {
    EVENT_CODES["GS_tun_recom"],
    EVENT_CODES["ES_tun_recom"],
    EVENT_CODES["GS_CB_recom"],
    EVENT_CODES["ES_CB_recom"],
    EVENT_CODES["Shallow_tun_recom"],
}


def is_luminescence(code: int) -> bool:
    """Return True if the event code corresponds to an event where charge carriers annihilate."""
    return code in LUMINESCENCE_CODES


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
        box.deep_cnt -= 1
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
    source_pool: str = "deep"

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        if box.d.size == 0:
            return np.empty(0)
        F = getattr(box, self.pop_factor_key)
        return F[exec_index] * self.rate_fn(box.d[exec_index, :])

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        if box.d.size == 0:
            return np.zeros(box.deep_cnt)
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box.d).sum(axis=1)

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        t_index = np.flatnonzero(box.occ_trap)[exec_index]
        self.operation.execute(box, t_index, local_idx)


@dataclass(slots=True)
class TunnelingRetrapping:
    """Distance-dependent tunneling retrapping into an unoccupied trap.
    When `energy_key` is set, the spatial rate is multiplied by the
    Miller-Abrahams VRH factor:
        exp( -(dE + |dE|) / (2 * k_b * T) )
    """

    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]
    pop_factor_key: str
    pre_factor: float
    operation: Operation
    energy_key: str | None = None
    source_pool: str = "deep"

    def _ma_factor(self, box: Box, exec_index: int | None = None) -> ArrayLike:
        dE = getattr(box, self.energy_key)
        if exec_index is not None:
            dE = dE[exec_index, :]
        return np.exp(-(dE + np.abs(dE)) / (2.0 * cnst.k_b_ev * box.T))

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        if box.d_ee.size == 0:
            return np.empty(0)
        F = getattr(box, self.pop_factor_key)
        spatial = self.rate_fn(box.d_ee[exec_index, :])
        if self.energy_key is not None:
            spatial = spatial * self._ma_factor(box, exec_index)
        return F[exec_index] * spatial * self.pre_factor

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        if box.d_ee.size == 0:
            return np.zeros(box.deep_cnt)
        F = getattr(box, self.pop_factor_key)
        spatial = self.rate_fn(box.d_ee)
        if self.energy_key is not None:
            spatial = spatial * self._ma_factor(box)
        return F * spatial.sum(axis=1) * self.pre_factor

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
    R_CB: float
    recom_operation: Operation
    retrap_operation: Operation
    source_pool: str = "deep"

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        F = getattr(box, self.pop_factor_key)
        bulk = self.rate_fn(box)                              # shape (t_cnt,)
        return np.atleast_1d(F[exec_index] * bulk[exec_index])

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        F = getattr(box, self.pop_factor_key)
        return F * self.rate_fn(box)                          # (t_cnt,) * (t_cnt,)

    def _build_cb_rates(self, box: Box, exec_index: int):
        """Return (cb_rates_array, n_recom_entries) for CB sub-selection."""
        cb_rates = np.array([], dtype=float)
        n_recom = 0
        if box.d.size != 0:
            recom_rates = self.cb_mob_fn(box.d[exec_index, :])
            cb_rates = np.append(cb_rates, recom_rates)
            n_recom = recom_rates.size
        if box.d_ee.size != 0:
            retrap_rates = self.cb_mob_fn(box.d_ee[exec_index, :]) * self.R_CB
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


@dataclass(slots=True)
class DeepToShallowOperation:
    """Electron leaves a deep trap and lands in a shallow site.
    Total trapped count t_cnt is unchanged (pool transfer only)."""
    event_code: int

    def execute(self, box: Box, t_index: int, sh_index: int) -> None:
        box.occ_trap[t_index] = 0
        box.occ_sh[sh_index] = 1
        box.deep_cnt -= 1
        box.sh_cnt += 1
        box.event_code = self.event_code


@dataclass(slots=True)
class ShallowToShallowOperation:
    """Electron hops from one shallow site to another shallow site."""
    event_code: int

    def execute(self, box: Box, sh_src_index: int, sh_dst_index: int) -> None:
        box.occ_sh[sh_src_index] = 0
        box.occ_sh[sh_dst_index] = 1
        box.event_code = self.event_code


@dataclass(slots=True)
class ShallowRecombineOperation:
    """Shallow electron annihilates with a hole (radiative recombination)."""
    event_code: int

    def execute(self, box: Box, sh_index: int, h_index: int) -> None:
        box.occ_sh[sh_index] = 0
        box.occ_hole[h_index] = 0
        box.sh_cnt -= 1
        box.t_cnt -= 1
        box.h_cnt -= 1
        box.event_code = self.event_code


@dataclass(slots=True)
class ShallowToDeepOperation:
    """Shallow electron de-excites into an unoccupied deep trap.
    Total trapped count t_cnt is unchanged (pool transfer only)."""
    event_code: int

    def execute(self, box: Box, sh_index: int, t_index: int) -> None:
        box.occ_sh[sh_index] = 0
        box.occ_trap[t_index] = 1
        box.sh_cnt -= 1
        box.deep_cnt += 1
        box.event_code = self.event_code


def _ma_factor_from_dE(dE: ArrayLike, T: float) -> ArrayLike:
    """Miller-Abrahams VRH factor: exp(-(dE + |dE|) / (2 k_b T))."""
    return np.exp(-(dE + np.abs(dE)) / (2.0 * cnst.k_b_ev * T))


@dataclass(slots=True)
class DeepToShallow:
    """Source = deep electron (GS or ES sub-level chosen via pop_factor_key); 
    target = unoccupied shallow site."""
    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]    # r -> b_BT * exp(-alpha_BT * r)
    pop_factor_key: str                          # "F1" (GS) or "F2" (ES)
    energy_key: str                              # "dE_t_sh_unocc_GS" or "..._ES"
    operation: Operation
    source_pool: str = "deep"

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        if box.d_t_sh_unocc.size == 0:
            return np.empty(0)
        F = getattr(box, self.pop_factor_key)
        spatial = self.rate_fn(box.d_t_sh_unocc[exec_index, :])
        dE = getattr(box, self.energy_key)[exec_index, :]
        return F[exec_index] * spatial * _ma_factor_from_dE(dE, box.T)

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        if box.d_t_sh_unocc.size == 0:
            return np.zeros(box.deep_cnt)
        F = getattr(box, self.pop_factor_key)
        spatial = self.rate_fn(box.d_t_sh_unocc)
        dE = getattr(box, self.energy_key)
        return F * (spatial * _ma_factor_from_dE(dE, box.T)).sum(axis=1)

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        t_index  = np.flatnonzero(box.occ_trap)[exec_index]
        sh_index = np.flatnonzero(box.occ_sh == 0)[local_idx]
        self.operation.execute(box, t_index, sh_index)


@dataclass(slots=True)
class ShallowTunRetrap:
    """Source = shallow electron; target = unoccupied shallow site."""
    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]
    operation: Operation
    source_pool: str = "shallow"

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        if box.d_sh_sh_unocc.size == 0:
            return np.empty(0)
        spatial = self.rate_fn(box.d_sh_sh_unocc[exec_index, :])
        dE = box.dE_sh_sh_unocc[exec_index, :]
        return spatial * _ma_factor_from_dE(dE, box.T)

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        if box.d_sh_sh_unocc.size == 0:
            return np.zeros(box.sh_cnt)
        spatial = self.rate_fn(box.d_sh_sh_unocc)
        dE = box.dE_sh_sh_unocc
        return (spatial * _ma_factor_from_dE(dE, box.T)).sum(axis=1)

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        sh_src = np.flatnonzero(box.occ_sh)[exec_index]
        sh_dst = np.flatnonzero(box.occ_sh == 0)[local_idx]
        self.operation.execute(box, sh_src, sh_dst)


@dataclass(slots=True)
class ShallowTunRecom:
    """Source = shallow electron; target = occupied hole.
    No M-A factor: the transition releases ~E_cb and is treated as downhill."""
    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]
    operation: Operation
    source_pool: str = "shallow"

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        if box.d_sh_h.size == 0:
            return np.empty(0)
        return self.rate_fn(box.d_sh_h[exec_index, :])

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        if box.d_sh_h.size == 0:
            return np.zeros(box.sh_cnt)
        return self.rate_fn(box.d_sh_h).sum(axis=1)

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        sh_index = np.flatnonzero(box.occ_sh)[exec_index]
        h_index  = np.flatnonzero(box.occ_hole)[local_idx]
        self.operation.execute(box, sh_index, h_index)


@dataclass(slots=True)
class ShallowToDeep:
    """Source = shallow electron; target = unoccupied deep trap.
    Two sub-channels (GS vs ES landing level) selected by energy_key."""
    name: str
    rate_fn: Callable[[ArrayLike], ArrayLike]
    energy_key: str                              # "dE_sh_t_unocc_GS" or "..._ES"
    operation: Operation
    source_pool: str = "shallow"

    def rates(self, box: Box, exec_index: int) -> ArrayLike:
        if box.d_sh_t_unocc.size == 0:
            return np.empty(0)
        spatial = self.rate_fn(box.d_sh_t_unocc[exec_index, :])
        dE = getattr(box, self.energy_key)[exec_index, :]
        return spatial * _ma_factor_from_dE(dE, box.T)

    def bulk_rates_sum(self, box: Box) -> ArrayLike:
        if box.d_sh_t_unocc.size == 0:
            return np.zeros(box.sh_cnt)
        spatial = self.rate_fn(box.d_sh_t_unocc)
        dE = getattr(box, self.energy_key)
        return (spatial * _ma_factor_from_dE(dE, box.T)).sum(axis=1)

    def execute(self, box: Box, exec_index: int, local_idx: int) -> None:
        sh_index = np.flatnonzero(box.occ_sh)[exec_index]
        t_index  = np.flatnonzero(box.occ_trap == 0)[local_idx]
        self.operation.execute(box, sh_index, t_index)
