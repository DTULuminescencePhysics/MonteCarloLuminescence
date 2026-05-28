from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class SetupConfig(BaseModel):
    project: str = Field(
        default="",
        description="Project name used in the Hydra run directory.",
    )
    seed: int = Field(
        default=1,
        ge=0,
        description="Random seed for reproducible Monte Carlo simulations.",
    )

    max_dt: float = Field(
        default=0.5,
        gt=0,
        description="Maximum timestep used by the simulation.",
    )

    t_pcnt: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of filled electron traps.",
    )

    h_pcnt: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of recombination holes that are unfilled.",
    )

    reps: int = Field(
        default=100,
        ge=1,
        description="Number of Monte Carlo simulations to repeat.",
    )

    boundary: Literal["padded", "periodic"] = Field(
        default="periodic",
        description="Boundary condition used by the simulation.",
    )

    n_jobs: int = Field(
        default=10,
        ge=1,
        description="Number of parallel workers. Use 1 for sequential execution.",
    )

    mc: bool = Field(
        default=True,
        description="Run Monte Carlo simulations.",
    )

    ac: bool = Field(
        default=False,
        description="Run analytic model.",
    )

    TC: bool = Field(
        default=False,
        description="Run thermochronology model.",
    )

    @model_validator(mode="after")
    def require_at_least_one_model(self):
        if not (self.mc or self.ac or self.TC):
            raise ValueError("At least one of mc, ac, or TC must be true.")
        return self


