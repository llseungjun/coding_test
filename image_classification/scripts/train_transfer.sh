#!/usr/bin/env bash
python -m src.train --model resnet18 --train_dir data/train --val_dir data/val --epochs 10 --batch_size 64 --lr 3e-4 --img_size 224 --freeze_backbone true
