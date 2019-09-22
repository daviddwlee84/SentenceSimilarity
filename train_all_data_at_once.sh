#!/bin/bash
# ./test_all_data_at_once Model

MODEL=${1:-'ERCNN'}
LOGDIR=log/${MODEL}_All_`date +"%m_%d-%H_%M_%S"`

# Ant
nohup python3 run.py --dataset Ant --word-segment char --model $MODEL --logdir $LOGDIR >& /dev/null &
nohup python3 run.py --dataset Ant --word-segment word --model $MODEL --logdir $LOGDIR >& /dev/null &

# CCSK
nohup python3 run.py --dataset CCSK --word-segment char --model $MODEL --logdir $LOGDIR >& /dev/null &
nohup python3 run.py --dataset CCSK --word-segment word --model $MODEL --logdir $LOGDIR >& /dev/null &

# PiPiDai
nohup python3 run.py --dataset PiPiDai --word-segment char --model $MODEL --logdir $LOGDIR >& /dev/null &
nohup python3 run.py --dataset PiPiDai --word-segment word --model $MODEL --logdir $LOGDIR >& /dev/null &

echo "Running scripts"
ps aux | grep run.py | grep $MODEL
echo "To stop all processes execute:"
echo "kill -9" `ps x | grep run.py | grep -v grep | awk '{print $1}' | xargs`
