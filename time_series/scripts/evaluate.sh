#!/usr/bin/env bash
python -m src.evaluate --ckpt runs/best.pt --data_csv data/val.csv --window 24 --horizon 1
