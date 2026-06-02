from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator
FloatPair = list[float] | None


class RJMCMCParametersConfig(BaseModel):
    p_birth: float = Field(default=0.20, ge=0, le=1, description="Birth probability")
    p_death: float = Field(default=0.20, ge=0, le=1, description="Death propbability")
    p_move_time: float = Field(default=0.20, ge=0, le=1, description="Move a time point probability")
    p_move_temp: float = Field(default=0.20, ge=0, le=1, description="Move a temperature point probability")
    p_move_endpoints: float = Field(default=0.20, ge=0, le=1, description="Move a start or end temperature probability")

    sigma_birth: float = Field(default=20.0, gt=0, description="Sigma for creating new time point")
    sigma_t_birth: float = Field(default=15.0, gt=0, description="Sigma for creating new temperature point")
    sigma_temp: float = Field(default=20.0, gt=0, description="Sigma for changing temperature point")
    sigma_time_frac: float = Field(default=15.0, gt=0, description="Sigma for changing time point")
    sigma_endpoints: float = Field(default=5.0, gt=0, description="Sigma for changing start or end temperature")

    @model_validator(mode="after")
    def validate_move_probabilities(self):
        total = (
            self.p_birth
            + self.p_death
            + self.p_move_time
            + self.p_move_temp
            + self.p_move_endpoints
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "RJMCMC move probabilities must sum to 1. "
                f"Current sum is {total}."
            )

        return self


class BurnInConfig(BaseModel):
    burn: bool = Field(default=False)

    max_steps: int = Field(default=20000, ge=1, description="Maximum number of burn in steps")
    window: int = Field(default=100, ge=1, description="Number of steps used to check agreement and alter parameters")
    patience_windows: int = Field(default=2, ge=1, description="Number of windows that must pass without alteration to end burn-in")

    adjustment_factor: float = Field(default=0.05, gt=0, description="Factor to scale etas at each window")
    eta_sigma: float = Field(default=2.0, gt=0, description="Value used to adjust sigmas")
    eta_prob: float = Field(default=0.08, gt=0, description="Value used to adjust proposal probabilities")
    
    
    move_bounds: list[float] | None = Field(default_factory=lambda: [0.03, 0.26], description="Proposal probabilities bounds")
    overall_check: bool = Field(default=False, description="Just check the overall acceptance rate")
    individual_check: bool = Field(default=True, description="Just check individual move acceptance rates")

    overall_accept_target: list[float] | None = Field(default_factory=lambda: [0.2, 0.4], description="Overall acceptance rate bounds")
    birth_accept_target: list[float] | None = Field(default_factory=lambda: [0.001, 0.05], description="Birth move acceptance rate bounds")
    death_accept_target: list[float] | None = Field(default_factory=lambda: [0.001, 0.05], description="Death move acceptance rate bounds")
    move_time_accept_target: list[float] | None = Field(default=None, description="Change time move acceptance rate bounds")
    move_temp_accept_target: list[float] | None = Field(default=None, description="Change temperature move acceptance rate bounds")
    move_endpoints_accept_target: list[float] | None = Field(default=None, description="Change start/end temperature move acceptance rate bounds")

    sigma_birth_bounds: list[float] | None = Field(default=None, description="Sigma time birth bounds")
    sigma_birth_t_bounds: list[float] | None = Field(default=None, description="Sigma temperature birth bounds")
    sigma_time_bounds: list[float] | None = Field(default=None, description="Sigma change time bounds")
    sigma_temp_bounds: list[float] | None = Field(default=None, description="Sigma change temperature bounds")
    sigma_endpoints_bounds: list[float] | None = Field(default=None, description="Sigma change start/end temperature bounds")

    verbose: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_burn_in(self):
       
        pair_fields = [
            "move_bounds",
            "overall_accept_target",
            "birth_accept_target",
            "death_accept_target",
            "move_time_accept_target",
            "move_temp_accept_target",
            "move_endpoints_accept_target",
            "sigma_birth_bounds",
            "sigma_birth_t_bounds",
            "sigma_time_bounds",
            "sigma_temp_bounds",
            "sigma_endpoints_bounds",
        ]

        for field_name in pair_fields:
            value = getattr(self, field_name)
            if value is None:
                continue

            if len(value) != 2:
                raise ValueError(f"{field_name} must contain exactly two values.")

            if value[0] > value[1]:
                raise ValueError(f"{field_name} lower bound cannot exceed upper bound.")

        return self

class RJMCMCConfig(BaseModel):
    logLikeSigma: float | None = Field(default=0.01,gt=0,description="Likelihood scale parameter.",)
    parameters: RJMCMCParametersConfig = Field(default_factory=RJMCMCParametersConfig)
    burn_in: BurnInConfig = Field(default_factory=BurnInConfig)


class ChronologyProfile(BaseModel):

    method: Literal["RJMCMC", "IVMC"] = Field(default="RJMCMC", description="Chronology inversion method.")
    iters: int = Field(default=1000,ge=1,description="Number of iteration steps.",)
    T_Target: float = Field(default=30.0,description="Target final temperature.",)
    T0_lo: float = Field(default=120.0,description="Minimum T0 value.",)
    T0_hi: float = Field(default=140.0,description="Maximum T0 value.",)
    T_tolerance: float = Field(default=1.0,ge=0,description="Allowed tolerance around minimum and maximum temperatures.",)
    monotonic: Literal["free", "increasing", "decreasing"] = Field(default="free",description="Monotonicity constraint for the time-temperature path.",)
    min_internal: int = Field(default=0,ge=0,description="Minimum number of internal time-temperature nodes.",)
    max_internal: int = Field(default=20,ge=0,description="Maximum number of internal time-temperature nodes.",)
    rjmcmc: RJMCMCConfig  = Field(default_factory=RJMCMCConfig,description="RJMCMC-specific settings.",)

    @model_validator(mode="after")
    def validate_chronology(self):
        if self.T0_lo > self.T0_hi:
            raise ValueError("T0_lo cannot be greater than T0_hi.")

        if self.min_internal > self.max_internal:
            raise ValueError("min_internal cannot be greater than max_internal.")

        if self.method == "RJMCMC" and self.rjmcmc is None:
            raise ValueError("rjmcmc settings are required when method is RJMCMC.")

        return self
