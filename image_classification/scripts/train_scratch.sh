#!/usr/bin/env bash
python -m src.train --model scratch --train_dir data/train --val_dir data/val --epochs 20 --batch_size 64 --lr 1e-3 --img_size 224
