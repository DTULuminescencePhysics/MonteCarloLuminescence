from __future__ import annotations
from typing import Callable, ClassVar, Dict
import numpy as np
from scipy.special import voigt_profile

from src.classes.constants import cnst

SigmaFn = Callable[[float, float], float]


def photon_energy_eV(wavelength_m: float) -> float:
   
    return cnst.h * cnst.c / (wavelength_m * cnst.ev_to_j)


def photon_wavelength_m(E_eV: float) -> float:
    
    return cnst.h * cnst.c / (E_eV * cnst.ev_to_j)


class CrossSection:

    REGISTRY: ClassVar[Dict[str, Callable[..., SigmaFn]]] = {}

    @classmethod
    def register(cls, kind: str) -> Callable[[Callable[..., SigmaFn]], Callable[..., SigmaFn]]:
        kind = kind.lower()

        def deco(fn: Callable[..., SigmaFn]) -> Callable[..., SigmaFn]:
            if not callable(fn):
                raise TypeError("cross-section builder must be callable")
            cls.REGISTRY[kind] = fn
            return fn

        return deco

    @classmethod
    def build(cls, kind: str | None, **params) -> SigmaFn:
        kind = (kind or "const").lower()
        if kind not in cls.REGISTRY:
            raise ValueError(
                f"Unknown cross-section model '{kind}'. Available: {sorted(cls.REGISTRY)}")
        return cls.REGISTRY[kind](**params)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls.REGISTRY)


@CrossSection.register("const")
def _build_const(sigma0: float | None = 0.0, **kw) -> SigmaFn:

    s0 = float(sigma0 or 0.0)

    def sigma(E_eV: float | None, T: float) -> float:
        return s0

    return sigma


@CrossSection.register("lucovsky")
def _build_lucovsky(sigma0: float, E0: float, **kw) -> SigmaFn:
    """Lucovsky (1965) short-range (delta) potential photoionisation cross-section."""
    s0 = float(sigma0 or 0.0)
    E0 = float(E0)

    def sigma(E_eV: float, T: float) -> float:
        if E_eV is None or E_eV <= E0:
            return 0.0
        return s0 * (4.0 * (E_eV - E0) * E0) ** 1.5 / E_eV ** 3

    return sigma


@CrossSection.register("grimmeiss")
def _build_grimmeiss(sigma0: float, E0: float, m_eff: float = 1.0, **kw) -> SigmaFn:
    """Grimmeiss-Ledebo (1975) effective-mass photoionisation cross-section."""
    s0 = float(sigma0 or 0.0)
    E0 = float(E0)
    m = float(m_eff)

    def _shape(E: float) -> float:
        if E <= E0:
            return 0.0
        return (E - E0) ** 1.5 / (E * (E + E0 * (1. / m - 1.0)) ** 2)

    grid = np.linspace(E0 + 1e-6, E0 + 12.0, 4000)
    peak = max((_shape(float(e)) for e in grid), default=0.0)
    norm = 1.0 / peak if peak > 0 else 0.0

    def sigma(E_eV: float, T: float) -> float:
        if E_eV is None:
            return 0.0
        return s0 * norm * _shape(E_eV)

    return sigma


@CrossSection.register("baril_voigt")
def _build_baril_voigt(sigma0: float, E1: float = 1.44, sigma_g1: float = 0.136,
                       gamma_l1: float = 0.010, A1: float = 1.0,
                       E2: float | None = 1.57, sigma_g2: float = 0.21,
                       A2: float = 0.333, E3: float | None = None,
                       sigma_g3: float = 0.0, A3: float = 0.0, **kw) -> SigmaFn:
    """Baril & Huntley (2003) IR intra-centre (GS->ES) excitation cross-section."""

    s0 = float(sigma0 or 0.0)
    E1 = float(E1)
    sg1 = float(sigma_g1)
    gl1 = float(gamma_l1)
    a1 = float(A1)
    v1_peak = float(voigt_profile(0.0, sg1, gl1))

    def _gauss_pk(E: float, E0: float, sg: float) -> float:
        if sg <= 0:
            return 0.0
        return float(np.exp(-(E - E0) ** 2 / (2.0 * sg ** 2)))

    def sigma(E_eV: float, T: float) -> float:
        if E_eV is None:
            return 0.0
        s = a1 * (float(voigt_profile(E_eV - E1, sg1, gl1)) / v1_peak
                  if v1_peak > 0 else 0.0)
        if E2 is not None and A2:
            s += float(A2) * _gauss_pk(E_eV, float(E2), float(sigma_g2))
        if E3 is not None and A3:
            s += float(A3) * _gauss_pk(E_eV, float(E3), float(sigma_g3))
        return s0 * s

    return sigma
