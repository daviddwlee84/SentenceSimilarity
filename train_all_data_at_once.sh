#!/bin/bash
# ./test_all_data_at_once Model

MODEL=${1:-'ERCNN'}
LOGDIR=log/${MODEL}_All_`date +"%m_%d-%H_%M_%S"`

mkdir -p $LOGDIR

export CUDA_VISIBLE_DEVICES=0
# Ant
nohup python3 run.py --dataset Ant --word-segment char --model $MODEL --lr 0.00001 --logdir $LOGDIR > /dev/null 2> $LOGDIR/AntCharTrainErr.log &
nohup python3 run.py --dataset Ant --word-segment word --model $MODEL --lr 0.00001 --logdir $LOGDIR > /dev/null 2> $LOGDIR/AntWordTrainErr.log &

# CCSK
nohup python3 run.py --dataset CCSK --word-segment char --model $MODEL --lr 0.001 --logdir $LOGDIR > /dev/null 2> $LOGDIR/CCSKCharTrainErr.log &
nohup python3 run.py --dataset CCSK --word-segment word --model $MODEL --lr 0.001 --logdir $LOGDIR > /dev/null 2> $LOGDIR/CCSKWordTrainErr.log &

export CUDA_VISIBLE_DEVICES=1
# PiPiDai
nohup python3 run.py --dataset PiPiDai --word-segment char --model $MODEL --lr 0.0001 --logdir $LOGDIR > /dev/null 2> $LOGDIR/PiPiDaiCharTrainErr.log &
nohup python3 run.py --dataset PiPiDai --word-segment word --model $MODEL --lr 0.0001 --logdir $LOGDIR > /dev/null 2> $LOGDIR/PiPiDaiWordTrainErr.log &

echo "Running scripts"
ps aux | grep run.py | grep $MODEL
echo "To stop all processes execute:"
echo "kill -9" `ps x | grep run.py | grep -v grep | awk '{print $1}' | xargs`
