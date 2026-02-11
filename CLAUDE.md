# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monte Carlo simulation framework for modeling luminescence processes (thermoluminescence, anomalous fading) in crystalline materials. Used for geochronology and dosimetry research. Simulates quantum mechanical transitions of electrons between traps and holes in a 3D crystal lattice, including tunneling, thermal excitation, and conduction band transport.

Key references: Jain et al. 2012/2015 (kinetic models), King et al. 2016 (geological application), Larsen et al. 2009 (dosing equation).

## Running the Simulation

```bash
conda activate lum
python main.py                    # Main MC simulation (Hydra entry point)
python analytic_model_run.py      # Standalone analytic kinetic models
```

The project must be run from the repository root directory. There is no package installation; imports use `src.` paths directly.

Hydra config overrides can be passed via CLI (e.g., `python main.py physics=phys_crib`). Output goes to timestamped directories under `run/`.

There are no tests, no linter configuration, and no CI/CD.

## Configuration System (Hydra)

`conf/config.yaml` composes three config groups:
- **`physics/`** — Crystal physics parameters (E_loc, b, alpha, s, rho, etc.)
- **`mc/`** — Monte Carlo parameters (reps, trap/hole percentages, mc/ac flags)
- **`temp/`** — Temperature profile (kind: constant/linear/step/steps/exponential/etc.)

Multi-experiment support: physics configs can contain lists for parameters, automatically expanded into separate experiments via `cfg_list_check()` in `src/helper_functions.py`.

## Architecture

### Entry Flow

`main.py` → `src.main_function.main()` (Hydra-decorated) → `MC_analytic_control.monte_carlo_control_functions()` → `MCBase.from_config(cfg)` → simulation loop

### Class Hierarchy

```
TimeTempProfile (registry, callable T(t))
  ← _time (time tracking)
    ← _temp (temperature as f(time))

CrystalPhysics / Transitions (registry, physics callables)
  ← _ThermalParameters (parameter setup)
    ← Box(_temp, _ThermalParameters): 3D crystal lattice with traps/holes
      ← MCBase: Monte Carlo simulation driver
```

The `core_development` branch introduces `Transitions` (in `transitions.py` / `transition_profiles.py`) as a replacement for `CrystalPhysics`, with per-transition tracking and a more flexible registry.

### Registry Pattern

Decorator-based registries are the central extensibility mechanism:
- `@TimeTempProfile.register("linear")` — temperature profiles in `temp_profiles.py`
- `@CrystalPhysics.register_fill("dose")` / `register_fade(...)` — physics in `physics_profiles.py`
- `@Transitions.register_fill(...)` / `register_tran(...)` — transition-based physics in `transition_profiles.py`

Each registered function is a **builder**: takes physics parameters, returns a **callable** implementing the equation.

### Key Source Files

| Purpose | Path |
|---------|------|
| Entry point | `main.py` |
| Hydra main function | `src/main_function.py` |
| MC/analytic orchestration | `src/MC_analytic_control.py` |
| MC simulation loop | `src/classes/monte_carlo.py` |
| 3D crystal lattice (Box) | `src/classes/physics/crystal.py` |
| Physics registry (legacy) | `src/classes/physics/system_physics.py` |
| Physics profiles (legacy) | `src/classes/physics/physics_profiles.py` |
| Transitions registry (new) | `src/classes/physics/transitions.py` |
| Transition profiles (new) | `src/classes/physics/transition_profiles.py` |
| Parameter setup | `src/classes/physics/set_system.py` |
| Temperature profiles | `src/classes/physics/temperature/temp_profiles.py` |
| Analytic ODE model | `src/classes/analytic_model.py` |
| Physical constants | `src/classes/constants.py` |
| Plotting/post-processing | `src/process_plot.py` |
| RJMCMC inverse modeling | `src/classes/thermoC/RJMCMC.py` |

### MC Simulation Loop (in `MCBase`)

1. `Box` generates random 3D trap/hole coordinates, computes full distance matrices (`scipy.spatial.distance.cdist`)
2. Loop: compute fill time and transition execution times → select minimum dt → execute event
3. `operate_electron()` probabilistically selects transition channel: tunneling recombination, tunneling retrapping, or conduction band excitation (with sub-selection)
4. Results stored in `np.memmap`, post-processed to CSV and plots

### Physics Transition Types

1. **Ground state tunneling** — `b * exp(-alpha_GS * r)`
2. **Excited state tunneling** — `b * exp(-alpha * r)`
3. **Ground state → conduction band** — `s * exp(-E_cb / (k_b * T))`
4. **Excited state → conduction band** — `s * exp(-(E_cb - E_loc) / (k_b * T))`
5. **Conduction band mobility** — `exp(r/mu)^2` retrapping probability

## Code Conventions

- All files use `from __future__ import annotations`
- Dataclasses used for all domain objects (often `@dataclass(slots=True)`)
- Type hints throughout (Python 3.10+ union syntax `float | None`)
- No `__init__.py` files; full `src.` import paths
- Physical constants via frozen dataclass singletons (`cnst`, `mp` in `constants.py`)
- Custom logging: `ErrorOutputHandler` splits `output.log` and `error.log` in run directories
