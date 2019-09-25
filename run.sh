#!/bin/bash
# For Ant Dataset online submission

# test data preprocessing
python3 ant_preprocess.py char test $1
# train and predict
python3 run.py --dataset Ant --mode submit --word-segment char --submit-path $2
