#!/usr/bin/env bash
python -m src.infer --ckpt runs/best.pt --recent_csv data/recent.csv --window 24 --horizon 7
