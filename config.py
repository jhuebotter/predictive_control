import yaml
from pathlib import Path


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
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if verbose:
        pretty(config)

    return config


def save_config(config: dict, path: str) -> None:
    """save the run configuration in the result dir"""

    with open(path, 'w+') as f:
        yaml.dump(config, f)