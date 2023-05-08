import pandas as pd
from pathlib import Path
import time
import sys

import collections 
if sys.version_info.major == 3 and sys.version_info.minor >= 10:

    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


class PandasLogger(object):
    """Logger class for graphical summary with Weights and Biases"""

    def __init__(self, name=None, dir: str = "results") -> None:

        self.results_dir = dir
        self.result = pd.DataFrame([])
        self.name = name

    def save_summary(self, summary: dict, step: int = 0) -> None:

        result = dict(step=step, name=self.name)
        for k, v in summary.items():
            if type(v) == list:
                s = ''
                for i in v:
                    s += f'{i}, '
                result[k] = s
            elif isinstance(v, dict):
                d = self.flatten(v, k)
                result.update(d)
            elif isinstance(v, (str, int, float, bool)):
                result[k] = v

        df = pd.DataFrame([result])
        self.result = pd.concat([self.result, df])
        self.save_result()

    def save_result(self, file: str = 'results.csv') -> None:

        if self.result.empty:
            print("nothing to save")
        else:
            path = Path(self.results_dir, file)
            print("saving results at", path)
            if path.exists():
                print("updating existing results")
                attempts = 0
                while attempts < 60:
                    try:
                        df = pd.read_csv(path)
                        break
                    except:
                        attempts += 1
                        time.sleep(1.0)
            else:
                print("making new results frame")
                df = pd.DataFrame([])
            df = pd.concat([df, self.result], ignore_index=True)
            attempts = 0
            while attempts < 60:
                try:
                    df.to_csv(path, index=False)
                    self.result = pd.DataFrame([])
                    break
                except:
                    attempts += 1
                    time.sleep(1.0)

    def flatten(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


if __name__ == '__main__':

    logger = PandasLogger(name='test2', dir='../results')

    for i in range(1, 11):
        results = \
            {'loss': 1./i, 'config':
                {'batch_size': 32, 'lr': 0.001, 'options': {
                    'bs': None, 'yes': False, 'list': [1, 2, 3]
                }}
            }
        logger.save_summary(results, step=i)