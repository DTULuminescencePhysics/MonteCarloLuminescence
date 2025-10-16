from __future__ import annotations
import hydra 
from omegaconf import DictConfig, OmegaConf
from src.errors import ErrorOutputHandler
from src.filesystem import CONFIG_DIR
from src.monte_carlo import run_monte_carlo_simulation
from src.process_plot import plot_ratio, plot_forward_results
from src.classes.physics.crystal import Box
# from src.input_check import check_inputs
# from src.classes.monte_carlo import run_monte_carlo_simulation
# from src.process_plot import process_data,plot_data
import logging


@hydra.main(config_path=CONFIG_DIR,config_name="config", version_base=None)
def main(cfg: DictConfig):
    root = logging.getLogger()
    err = next(h for h in root.handlers if isinstance(h, ErrorOutputHandler))
    
    err.output("Setting up simulation crystal...")
    crystal = Box.from_config(cfg)
    err.output("Crystal setup complete.")
    err.output(crystal.__repr__())
    

    run_monte_carlo_simulation(crystal, cfg, err)
    file_name = "MC_results.csv"
    plot_forward_results(file_name, crystal.unit, crystal.kind)
    
    exit()

    # phys_in, mc_in = check_inputs(cfg, err)
    
    # mc = run_monte_carlo_simulation(phys_in,mc_in,err)

    # process_data(mc)
    # plot_data()