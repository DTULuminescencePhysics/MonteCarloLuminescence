from __future__ import annotations
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from conf.config_app import AppConfig
    from omegaconf import DictConfig, ListConfig, DictKeyType
    
def dump_setup(config: AppConfig) -> DictConfig:
    """Convert the validated Pydantic AppConfig into an OmegaConf DictConfig."""
    data = config.model_dump(exclude_none=False)
    return OmegaConf.create(data)

def make_cfg_from_app_config(config: AppConfig, CONFIG_DIR:Path)-> ListConfig | DictConfig :
    app_cfg = dump_setup(config)
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()),version_base=None,):
        base_cfg = compose(config_name="config")

    OmegaConf.set_struct(base_cfg, False)

    merged_cfg = OmegaConf.merge(base_cfg, app_cfg)

    OmegaConf.set_struct(merged_cfg, False)

    return merged_cfg


def discover_exisitng_profiles(dir: str | Path ) -> list:
    dir = Path(dir)
    return sorted(path.stem for path in dir.glob("*.yaml"))

def load_profile(profile_name: str, dir: str | Path) -> dict[DictKeyType, Any]:
    path = Path(dir) / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Temperature profile not found: {path}")
    
    cfg = OmegaConf.load(path)
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(
            f"Expected YAML profile to load as a dict, got {type(container).__name__}"
        )

    return container

def save_profile(data: dict,  path: Path,) -> Path:

    OmegaConf.save(OmegaConf.create(data), path)
    return path

def list_to_text(values: list[float] | None) -> str:
    if values is None:
        return ""
    return ", ".join(str(v) for v in values)

def text_to_float_list(text: str) -> list[float] | None:
    if not text.strip():
        return None
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def update_main_user_config(directory: str | Path, configFile: str , user_config_name: str,) -> Path:
    config_path = Path(directory,configFile)
    cfg = OmegaConf.load(config_path)
    for item in cfg.defaults:
        if "user_config" in item:
            item["user_config"] = user_config_name
            break
    else:
        raise ValueError("No user_config entry found in defaults.")

    OmegaConf.save(cfg, config_path)

    return config_path