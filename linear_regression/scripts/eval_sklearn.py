#!/usr/bin/env python
import argparse
from linreg_sklearn.evaluate import run_evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    args = p.parse_args()
    run_evaluate(args.model, args.csv, args.target)
