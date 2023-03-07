from pathlib import Path
import omegaconf
from omegaconf import OmegaConf


def pretty(d: dict, indent: int = 0) -> None:
    """print a nested dict in a readable manner"""

    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + str(key) + ':')
            pretty(value, indent + 1)
        else:
            print('\t' * indent + f'{key:30}: {str(value):20}, {type(value)}')


def read_config(path: str = 'config.yaml', verbose: bool = True) -> dict:
    """read a config file from disk and return the contents as a dictionary"""

    path = Path(path)
    config = OmegaConf.load(path)

    if verbose:
        pretty(config)
        print()

    return config


def save_config(config: dict, path: str) -> None:
    """save the run configuration in the result dir"""

    with open(path, 'w+') as f:
        OmegaConf.save(config, f)


def get_config(filepath: Path or str = "config.yaml", verbose: bool = True):
    """get default parameters from file and overwrite based on parsed arguments."""

    def_config = read_config(filepath, verbose=False)
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(def_config, cli_config)
    config_dict = omegaconf2dict(config)

    if verbose:
        print(OmegaConf.to_yaml(config))

    return config_dict


def omegaconf2dict(conf: omegaconf.dictconfig.DictConfig) -> dict:
    """get from omegaconf format to dict so that values can be logged correctly via wandb"""

    d = dict()
    for k, v in conf.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            v = omegaconf2dict(v)
        if type(v) is str and v.lower() == 'none':
            v = None
        d.update({k: v})

    return d