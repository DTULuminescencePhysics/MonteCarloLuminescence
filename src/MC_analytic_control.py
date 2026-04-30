from __future__ import annotations
from src.errors import ErrorOutputHandler
from omegaconf import DictConfig
from src.classes.monte_carlo import MCBase
from src.classes.analytic_model import analytical_crystal
from src.classes.output.process_plot import clean_up_results
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.classes.output.results_file import output_file
    from src.classes.output.graph import MainPlot

def monte_carlo_control_functions(cfg: DictConfig | list[DictConfig], 
                                  experiments: int, results: output_file, err: ErrorOutputHandler) -> None:
    """Function that can run either a single or multiple Monte Carlo experiments"""

    if experiments == 1 and isinstance(cfg, DictConfig):
        err.output("Setting up simulation crystal...")
        MC = MCBase.from_config(cfg)
        err.output("Crystal setup complete.")
        err.output(MC.crystal.__repr__())
        MC.full_monte_carlo_simulation(err)
        clean_up_results(MC.results, MC.crystal, results)
        MC.clean_up()
    
    else:
        for i in range(experiments):
            err.output("Setting up simulation crystal...")
            MC = MCBase.from_config(cfg[i])
            MC.result_csv_path += f"_{i+1}"
            err.output("Crystal setup complete.")
            err.output(MC.crystal.__repr__())
            MC.full_monte_carlo_simulation(err)
            clean_up_results(MC.results,MC.crystal, results)
            MC.clean_up()
          

def analytic_control_functions(cfg: DictConfig | list[DictConfig], 
                                  experiments: int, pl: MainPlot, err: ErrorOutputHandler) -> None:
    """Function that can run either a single or multiple analytical experiments"""

    if experiments == 1 and isinstance(cfg, DictConfig):
        err.output("Setting up analytical crystal...")
        AC = analytical_crystal.from_config(cfg)
        err.output("Crystal setup complete.")
        err.output(AC.__repr__())
        AC.get_analytical_solution()
        pl.set_analytic_ratio_file(AC.result_csv_path)
    else:
        for i in range(experiments):
            err.output("Setting up analytical crystal...")
            AC = analytical_crystal.from_config(cfg[i])
            AC.result_csv_path += f"_{i+1}"
            err.output("Crystal setup complete.")
            err.output(AC.__repr__())
            AC.get_analytical_solution()
            pl.set_multi_analytic_ratio_file(AC.result_csv_path)
