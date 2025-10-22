import os, pickle, random
import numpy as np


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_pickle(obj, path: str):
    ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
