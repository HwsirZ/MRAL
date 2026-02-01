import habitat
from .wrappers import SLAMWrapper
from omegaconf import OmegaConf

def make_env(cfg_path):
    cfg = habitat.get_config(cfg_path)
    print(OmegaConf.to_yaml(cfg.habitat))
    env = habitat.Env(cfg)
    env = SLAMWrapper(env)
    return env
