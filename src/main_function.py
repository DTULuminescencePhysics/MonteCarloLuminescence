from __future__ import annotations
import hydra 
from omegaconf import DictConfig
from src.errors import ErrorOutputHandler
from src.filesystem import CONFIG_DIR
from src.classes.monte_carlo import MCBase
from src.classes.RJMCMC import ReverseJmpMCMC
from src.classes.physics.analytic_model import analytical_crystal
from src.process_plot import plot_analytic_comp_MC,plot_analytic_comp_MC_smoothed, comp_smooth
# from src.input_check import check_inputs

import logging


@hydra.main(config_path=CONFIG_DIR,config_name="config", version_base=None)
def main(cfg: DictConfig):
    root = logging.getLogger()
    err = next(h for h in root.handlers if isinstance(h, ErrorOutputHandler))

    err.output("Setting up simulation crystal...")
    MC = MCBase.from_config(cfg)
    err.output("Crystal setup complete.")
    err.output(MC.crystal.__repr__())
    MC.full_monte_carlo_simulation(err)

    comp_smooth("MC_results_ratio.csv",MC.crystal.unit)

    AC = analytical_crystal.from_config(cfg)
    err.output(AC.__repr__())
    AC.get_analytical_solution()

    plot_analytic_comp_MC("Analytical_results.csv","MC_results_ratio.csv",MC.crystal.unit)
    plot_analytic_comp_MC_smoothed("Analytical_results.csv","MC_results_ratio.csv",MC.crystal.unit)
    
    # import numpy as np 
    # val = np.loadtxt("MC_results.csv", delimiter=",", skiprows=1)
    # val[:,1] = val[:,2]
    # val = val[:,0:2]
   
    # rjmcmc = ReverseJmpMCMC.from_config(val,1000,cfg)
    # rjmcmc.rjmcmc_temperature(err)

