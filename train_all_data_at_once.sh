#!/bin/bash
# ./test_all_data_at_once Model

MODEL=${1:-'ERCNN'}
LOGDIR=log/All_`date +"%m_%d-%H_%M_%S"`

# Ant
python3 run.py --dataset Ant --word-segment char --model $MODEL --logdir $LOGDIR &
python3 run.py --dataset Ant --word-segment word --model $MODEL --logdir $LOGDIR &

# CCSK
python3 run.py --dataset CCSK --word-segment char --model $MODEL --logdir $LOGDIR &
python3 run.py --dataset CCSK --word-segment word --model $MODEL --logdir $LOGDIR &

# PiPiDai
python3 run.py --dataset PiPiDai --word-segment char --model $MODEL --logdir $LOGDIR &
python3 run.py --dataset PiPiDai --word-segment word --model $MODEL --logdir $LOGDIR &
