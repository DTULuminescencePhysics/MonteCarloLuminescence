from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class PhysicsProfile(BaseModel):
    # Crystal geometry / lattice
    uc_h: float = Field(default=1e-10, gt=0, description="Unit cell height in meters.",)
    uc_w: float = Field(default=1e-10,gt=0, description="Unit cell width in meters.",)
    uc_l: float = Field(default=1e-10,gt=0,description="Unit cell length in meters.",)
    dimension: float | None = Field(default=7.5e-9,gt=0,description="Approximate test-crystal dimension. None means default 7.5 nm.",)
    rho: float | None = Field(default = 5.22e25,ge=0,description="Density parameter.",)
    urho: float | None = Field(default = None,ge=0,description="Optional uncertainty or variation in rho.",)
    # Trap energy levels
    E_loc: float = Field(default=1.2,gt=0,description="Localized trap energy level in eV.",)
    E_cb: float = Field(default=2.0, gt=0, description="Conduction band energy level in eV.",)
    E_loc_sigma: float = Field(default=0.0, ge=0, description="Standard deviation of Gaussian E_loc distribution in eV. 0 means fixed.",)
    E_cb_sigma: float = Field(default=0.0, ge=0, description="Standard deviation of Gaussian E_cb distribution in eV. 0 means fixed.",)

    # Filling process / dosing
    enable_fill: bool = Field(default=False, description="Enable filling/dosing stage.",)
    D0: float = Field(default=400.0, ge=0, description="Initial or total radiation dose.",)
    D_dot: float = Field( default=1.0, ge=0, description="Radiation dose rate per time unit.",)
    Dd_unit: Literal["s", "m", "h", "d", "y", "ka", "ma"] = Field(default="s", description="Dose-rate time unit.",)
    combine_when_fill: bool = Field(default=False, description="Whether to combine events during filling.",)
    recom_pre_fill: float = Field(default=0.0, ge=0, description="P(recombine) / P(fill vacancy) before filling.",)

    # Localized transitions
    enable_tunneling: bool = Field( default=True, description="Enable localized tunnelling transitions.",)
    b: float | None = Field(default=1e12, gt=0, description="Tunnelling frequency factor.",)
    alpha_GS: float | None = Field(default=9e12, gt=0, description="Ground-state localization parameter.",)
    alpha_ES: float | None = Field(default=9e9, gt=0, description="Excited-state localization parameter.",)
    R_tun: float = Field(default=0.0, ge=0, le=1, description="Retrapping ratio for tunnelling.",)
    VRH: bool = Field(default=False, description="Use Miller-Abrahams variable-range hopping instead of original tunnelling rate.",)

    # Delocalized transitions
    enable_cb: bool = Field(default=True, description="Enable conduction-band transitions.",)
    s: float | None = Field(default=1e12, gt=0, description="Frequency factor for conduction-band transitions.",)
    mu: float | None = Field(default=0.1, gt=0, description="Mobility parameter.",)
    R_CB: float = Field(default=0.0, ge=0, le=1, description="Retrapping ratio for conduction-band transitions.",)
    retrap_mask_factor: float = Field(default=0.9, ge=0, le=1, description="Factor constraining retrapping in adjacent pairs.",)

    # Band-tail transitions
    enable_BT: bool = Field(default=False, description="Enable band-tail shallow-defect transitions.",)
    shallow_deep_ratio: float | None = Field(default=5.0, gt=0,description="N_sh = int(shallow_deep_ratio * N).",)
    threshold_depth: float | None = Field(default=0.5, gt=0, description="Maximum depth below conduction band for shallow defects in eV.",)
    E_u: float | None = Field(default=0.3, gt=0, description="Urbach energy in eV.",)
    b_BT: float | None = Field(default=None, gt=0, description="Band-tail frequency factor. None means defaults to b.",)
    alpha_BT: float | None = Field(default=1.5e9, gt=0, description="Band-tail localization parameter. None means defaults to alpha_GS.",)
    init_shallow: bool = Field(default=True, description="If true, initial electrons may sit in shallow defects.",)
    sh_pcnt: float | None = Field(default=0.2, ge=0, le=1, description="Fraction of shallow traps occupied at t=0.",)


    @model_validator(mode="after")
    def validate_conditional_physics(self):
       
        if self.rho is None and self.urho is None:
            raise ValueError("The density of electron traps must have a value either with or without units.")
        elif self.rho is not None and self.urho is not None:
            raise ValueError("Only one of the density of electron traps values should be set either with or without units.")

        if self.enable_tunneling:
            if self.b is None:
                raise ValueError("b is required when enable_tunneling is True.")
            if self.alpha_GS is None:
                raise ValueError("alpha_GS is required when enable_tunneling is True.")
            if self.alpha_ES is None:
                raise ValueError("alpha_ES is required when enable_tunneling is True.")

        if self.enable_cb:
            if self.s is None:
                raise ValueError("s is required when enable_cb is True.")
            if self.mu is None:
                raise ValueError("mu is required when enable_cb is True.")

        if self.enable_BT:
            required = {
                "shallow_deep_ratio": self.shallow_deep_ratio,
                "threshold_depth": self.threshold_depth,
                "E_u": self.E_u,
                "alpha_BT": self.alpha_BT,
                "sh_pcnt": self.sh_pcnt,
            }

            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"{', '.join(missing)} required when enable_BT is True."
                )

        return self