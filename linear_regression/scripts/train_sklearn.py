#!/usr/bin/env python
import argparse
from linreg_sklearn.train import run_train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/sklearn.yaml")
    args = p.parse_args()
    run_train(args.config)
