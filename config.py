from pathlib import Path
from omegaconf import OmegaConf


def pretty(d: dict, indent: int = 0) -> None:
    """print a nested dict in a readable manner"""

    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + str(key) + ':')
            pretty(value, indent + 1)
        else:
            print('\t' * indent + f'{key:30}: {str(value):20}')


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

    if verbose:
        pretty(config)

    return config