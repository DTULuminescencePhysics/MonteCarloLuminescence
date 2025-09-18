import hydra 
from omegaconf import DictConfig
from src.errors import ErrorOutputHandler
from src.filesystem import CONFIG_DIR
from src.input_check import check_inputs
import logging


@hydra.main(config_path=CONFIG_DIR,config_name="config", version_base=None)
def main(cfg: DictConfig):
    root = logging.getLogger()
    err = next(h for h in root.handlers if isinstance(h, ErrorOutputHandler))
    # err = ErrorOutputHandler(logger_name="app")
    # err.install_hooks()

    phys_in, mc_in = check_inputs(cfg, err)
    print(phys_in)



