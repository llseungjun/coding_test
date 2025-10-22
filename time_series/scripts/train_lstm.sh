#!/usr/bin/env bash
python -m src.train --data_csv data/train.csv --model lstm --window 24 --horizon 1 --epochs 10 --lr 1e-3 --batch_size 64
