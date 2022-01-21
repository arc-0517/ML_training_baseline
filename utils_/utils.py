import random

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_rich_pbar(transient: bool = True, auto_refresh: bool = False):
    """A colorful progress bar based on the `rich` python library."""
    console = Console(color_system='256', force_terminal=True, width=160)
    return Progress(
        console=console,
        auto_refresh=auto_refresh,
        transient=transient
    )

def get_scaler(scaler_type: str = 'norm'):
    if scaler_type == 'norm':
        scaler = Normalizer()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError()
    return scaler


def set_seed(seed):
    # torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    random.seed(seed)
