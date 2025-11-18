from __future__ import annotations
from src.errors import ErrorOutputHandler
from omegaconf import DictConfig
from src.classes.monte_carlo import MCBase
from src.classes.analytic_model import analytical_crystal
from src.process_plot import clean_up_results, plot_forward_results, plot_forward_multi_experiment
from src.process_plot import plot_analtyical_results, plot_forward_multi_analytical_experiment


def monte_carlo_control_functions(cfg: DictConfig | list[DictConfig], 
                                  experiments: int, err: ErrorOutputHandler) -> str | list[str]:
    """Function that can run either a single or multiple Monte Carlo experiments"""

    if experiments == 1 and isinstance(cfg, DictConfig):
        err.output("Setting up simulation crystal...")
        MC = MCBase.from_config(cfg)
        err.output("Crystal setup complete.")
        err.output(MC.crystal.__repr__())
        MC.full_monte_carlo_simulation(err)
        ratio_file, lum_file = clean_up_results(MC.results, MC.result_csv_path,MC.crystal)
        plot_forward_results(ratio_file, lum_file, MC.crystal.unit, MC.crystal.kind)
        MC.clean_up()
        del lum_file
        return ratio_file
    else:
        file_names = []
        for i in range(experiments):
            err.output("Setting up simulation crystal...")
            MC = MCBase.from_config(cfg[i])
            MC.result_csv_path += f"_{i+1}"
            err.output("Crystal setup complete.")
            err.output(MC.crystal.__repr__())
            MC.full_monte_carlo_simulation(err)
            ratio_file, lum_file = clean_up_results(MC.results, MC.result_csv_path,MC.crystal)
            plot_forward_results(ratio_file, lum_file, MC.crystal.unit, MC.crystal.kind)
            MC.clean_up()
            file_names.append(ratio_file)
            del ratio_file, lum_file
        plot_forward_multi_experiment(file_names, "All_experiments_ratio_comp.png",cfg[0].temp.unit)
        return file_names

def analytic_control_functions(cfg: DictConfig | list[DictConfig], 
                                  experiments: int, err: ErrorOutputHandler) -> str | list[str]:
    """Function that can run either a single or multiple analytical experiments"""

    if experiments == 1 and isinstance(cfg, DictConfig):
        err.output("Setting up analytical crystal...")
        AC = analytical_crystal.from_config(cfg)
        err.output("Crystal setup complete.")
        err.output(AC.__repr__())
        AC.get_analytical_solution()
        plot_analtyical_results(AC.result_csv_path, AC.unit)
        return f"{AC.result_csv_path}.csv"
    else:
        file_names = []
        for i in range(experiments):
            err.output("Setting up analytical crystal...")
            AC = analytical_crystal.from_config(cfg[i])
            AC.result_csv_path += f"_{i+1}"
            err.output("Crystal setup complete.")
            err.output(AC.__repr__())
            AC.get_analytical_solution()
            plot_analtyical_results(AC.result_csv_path, AC.unit)
            file_names.append(f"{AC.result_csv_path}.csv")
       
        plot_forward_multi_analytical_experiment(file_names, "All_analytical_experiments_ratio_comp.png",cfg[0].temp.unit)
        return file_names
