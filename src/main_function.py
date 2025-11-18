from __future__ import annotations
import hydra 
import logging
from omegaconf import DictConfig
from src.errors import ErrorOutputHandler
from src.filesystem import CONFIG_DIR
from src.helper_functions import cfg_list_check
from src.MC_analytic_control import monte_carlo_control_functions, analytic_control_functions
from src.process_plot import plot_analytic_comp_MC

from src.classes.RJMCMC import ReverseJmpMCMC


@hydra.main(config_path=CONFIG_DIR,config_name="config", version_base=None)
def main(cfg: DictConfig):
    root = logging.getLogger()
    err = next(h for h in root.handlers if isinstance(h, ErrorOutputHandler))

    run_num, runs = cfg_list_check(cfg,err)
    err.checkpoint()
    if cfg.mc.mc:
        mc_file = monte_carlo_control_functions(runs,run_num,err)
    if cfg.mc.ac:
        ac_file = analytic_control_functions(runs,run_num,err)

    if 'mc_file' in locals() and 'ac_file' in locals():
        plot_analytic_comp_MC(run_num,ac_file,mc_file,cfg.temp.unit)
    

    exit()

   

    # rjmcmc = ReverseJmpMCMC.from_config(val,1000,cfg)
    # rjmcmc.rjmcmc_temperature(err)

