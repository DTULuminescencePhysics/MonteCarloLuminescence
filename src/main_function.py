from __future__ import annotations
import hydra 
import logging
from omegaconf import DictConfig
from src.errors import ErrorOutputHandler
from src.filesystem import CONFIG_DIR
from src.helper_functions import cfg_list_check, cfg_temperature_check
from src.MC_analytic_control import monte_carlo_control_functions, analytic_control_functions
from src.back_tracing import MC_control_functions,RJMCMC_control_functions
from src.classes.output.graph import MainPlot 


@hydra.main(config_path=CONFIG_DIR,config_name="config", version_base=None)
def main(cfg: DictConfig):
    root = logging.getLogger()
    err = next(h for h in root.handlers if isinstance(h, ErrorOutputHandler))

    cfg_temperature_check(cfg,err)
    err.checkpoint()
    run_num, runs = cfg_list_check(cfg,err)
    # run_num=1 
    # runs = runs[0]
    pl = MainPlot(unit=cfg.temp.unit,celsius=cfg.temp.celsius)
    err.checkpoint()
    if cfg.setup.mc:
        monte_carlo_control_functions(runs,run_num,pl,err)
    if cfg.setup.ac:
        analytic_control_functions(runs,run_num,pl,err)

    pl.plot_forward()
    comp_file = f"{pl.forward_ratio_file[0]}.csv"
    if cfg.setup.TC:
        RJMCMC_control_functions(runs,run_num,err,comp_file)
        # MC_control_functions(runs,run_num,err,comp_file)
