from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class TemperatureProfile(BaseModel):
    unit: Literal["s", "m","h","d","y","Ka","Ma"] = Field(default="s", description="Unit of time used in the simulation.",)
    duration: float = Field(gt=0.0,default=120,description="Duration of the experiment in the specified time unit.")
    celsius: bool = Field(default=True,description="Units of temperatures true means celsius used ")
    T0: float = Field(description="Starting temperature in specified units.",default=25)
    kind: Literal["Constant","Linear","Other"] = Field(default="Constant",description="Type of temperature profile")
    times: list[float] | None = Field(default=None,description=("Time points for the temperature profile. "
                                                                "Must start at 0 and end at duration."),)
    temps: list[float] | None = Field(default=None, description=("Temperature values corresponding to times. "
                                                                 "Must have the same length as times and start at T0."),)
    dT: float | None = Field(default=None,description="Temperature step size or increment.",)

    @model_validator(mode="after")
    def validate_temperature_profile(self):
        if self.kind == "Constant":
            self.times = [0,self.duration]
            self.temps = [self.T0,self.T0]
            self.dT = None
            return self
        elif self.kind == "Linear":
            if self.times is None and self.temps is None and self.dT is None:
                raise ValueError("times and temps or dT is required but all were set to None")
            elif self.times is None and self.temps is None and self.dT is not None:
                self.times = [0,self.duration]
                self.temps = [self.T0,self.T0+(self.duration*self.dT)]
            elif self.times is not None and self.temps is not None and self.dT is None:
                if len(self.times) != 2:
                    raise ValueError("times must contain at least two values: 0 and duration.")
                if len(self.temps) != len(self.times):
                    raise ValueError("temps must have the same length as times.") 
                self.dT = (self.T0 - self.temps[1])/self.duration
        else:
            if self.times is None:
                raise ValueError(f"times is required when kind is '{self.kind}'.")
            if self.temps is None:
                raise ValueError(f"temps is required when kind is '{self.kind}'.")

            if len(self.times) < 2:
                raise ValueError("times must contain at least two values: 0 and duration.")
            if len(self.temps) != len(self.times):
                raise ValueError("temps must have the same length as times.")
            if self.times[0] != 0:
                raise ValueError("times must start with 0.")
            if self.times[-1] != self.duration:
                raise ValueError("times must end with duration.")
            if self.temps[0] != self.T0:
                raise ValueError("temps must start with T0.")
            if any(t2 < t1 for t1, t2 in zip(self.times, self.times[1:])):
                raise ValueError("times must be increasing.")
        return self


    