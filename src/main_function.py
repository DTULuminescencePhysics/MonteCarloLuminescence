from __future__ import annotations
import hydra 
import os
import argparse
import subprocess
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager

from src.filesystem import CONFIG_DIR
from src.errors import  get_error_output_handler
from src.helper_functions import cfg_list_check, cfg_temperature_check
from src.MC_analytic_control import monte_carlo_control_functions
from src.back_tracing import back_tracing_selector 
from src.classes.output.results_file import output_file
from src.classes.output.graph import chronologyPlot



def run_program(cfg: DictConfig, run:bool = True) -> None:
    if run:
        err = get_error_output_handler()
    
        cfg_temperature_check(cfg,err)
        err.checkpoint()
        run_num, runs = cfg_list_check(cfg,err)
        err.checkpoint()
        results = output_file("results.hdf5",cfg)
        if cfg.setup.mc:
            monte_carlo_control_functions(runs,run_num,results,err)
      
        if cfg.setup.TC:
            back_tracing_selector(results,runs,run_num,err)
            cp = chronologyPlot(0.5,unit=cfg.temp.unit,celsius=cfg.temp.celsius)
            cp.make_weighting_plot(results)


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def hydra_main(cfg: DictConfig,) -> None:
    
        run_program(cfg.user_config)


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def hydra_save(cfg: DictConfig,) -> None:
    run_program(cfg.user_config, False)
    print("Experiment saved")

@contextmanager
def working_directory(path: str | Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

def run_prepared_experiment(experiment_dir: str | Path) -> None:

    experiment_dir = Path(experiment_dir).resolve()
    config_path = Path(experiment_dir,".config","config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"No prepared config found at {config_path}")

     
    cfg = OmegaConf.load(config_path)
    with working_directory(experiment_dir):
        run_program(cfg.user_config)


def parse_launcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo Luminescence runner",)

    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit configuration UI instead of running the simulation directly.",)
    parser.add_argument("--run", type=str, default=None,help="Run an experiment that was prepared by the UI.",)
    parser.add_argument("--save", action="store_true", help="save an experiment to run at a later point")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    return args