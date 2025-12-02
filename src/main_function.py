from __future__ import annotations
import hydra 
import logging
from omegaconf import DictConfig
from src.errors import ErrorOutputHandler
from src.filesystem import CONFIG_DIR
from src.helper_functions import cfg_list_check
from src.MC_analytic_control import monte_carlo_control_functions, analytic_control_functions
from src.process_plot import plot_analytic_comp_MC
from src.back_tracing import MC_control_functions,RJMCMC_control_functions


@hydra.main(config_path=CONFIG_DIR,config_name="config", version_base=None)
def main(cfg: DictConfig):
    root = logging.getLogger()
    err = next(h for h in root.handlers if isinstance(h, ErrorOutputHandler))

    run_num, runs = cfg_list_check(cfg,err)
    run_num=1 
    runs = runs[0]
    err.checkpoint()
    if cfg.mc.mc:
        mc_file = monte_carlo_control_functions(runs,run_num,err)
    if cfg.mc.ac:
        ac_file = analytic_control_functions(runs,run_num,err)

    if 'mc_file' in locals() and 'ac_file' in locals():
        plot_analytic_comp_MC(run_num,ac_file,mc_file,cfg.temp.unit)
    
    RJMCMC_control_functions(runs,run_num,err,ac_file)
    # MC_control_functions(runs,run_num,err,ac_file)
