#!/bin/bash

python training_scripts/training_pipeline.py ConvNeXtBase --pretrained --augment --learning-rate 1e-4 --batch-size 64 --epochs 50 --root-dir ./data/dataset_split --save-weights-dir ./weights/convnext-base --save-weights-freq 2
