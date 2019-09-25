#!/bin/bash
# For Ant Dataset online submission
# the online judgement system don't have "python3"

# test data preprocessing
python ant_preprocess.py char test $1
# train and predict
python run.py --dataset Ant --model SiameseCNN --mode submit --word-segment char --submit-path $2
