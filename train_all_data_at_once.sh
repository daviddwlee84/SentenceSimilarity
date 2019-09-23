#!/bin/bash
# ./test_all_data_at_once Model

MODEL=${1:-'ERCNN'}
LOGDIR=log/${MODEL}_All_`date +"%m_%d-%H_%M_%S"`

mkdir -p $LOGDIR

export CUDA_VISIBLE_DEVICES=0
# Ant
nohup python3 run.py --dataset Ant --word-segment char --model $MODEL --logdir $LOGDIR > /dev/null 2> $LOGDIR/AntCharTrainErr.log &
nohup python3 run.py --dataset Ant --word-segment word --model $MODEL --logdir $LOGDIR > /dev/null 2> $LOGDIR/AntWordTrainErr.log &
# nohup python3 run.py --dataset Ant --word-segment char --model $MODEL --not-train-embed --logdir $LOGDIR > /dev/null 2> $LOGDIR/AntCharFixedErr.log &
# nohup python3 run.py --dataset Ant --word-segment word --model $MODEL --not-train-embed --logdir $LOGDIR > /dev/null 2> $LOGDIR/AntWordFixedErr.log &

# CCSK
nohup python3 run.py --dataset CCSK --word-segment char --model $MODEL --logdir $LOGDIR > /dev/null 2> $LOGDIR/CCSKCharTrainErr.log &
nohup python3 run.py --dataset CCSK --word-segment word --model $MODEL --logdir $LOGDIR > /dev/null 2> $LOGDIR/CCSKWordTrainErr.log &
# nohup python3 run.py --dataset CCSK --word-segment char --model $MODEL --not-train-embed --logdir $LOGDIR > /dev/null 2> $LOGDIR/CCSKCharFixedErr.log &
# nohup python3 run.py --dataset CCSK --word-segment word --model $MODEL --not-train-embed --logdir $LOGDIR > /dev/null 2> $LOGDIR/CCSKWordFixedErr.log &

export CUDA_VISIBLE_DEVICES=1
# PiPiDai
nohup python3 run.py --dataset PiPiDai --word-segment char --model $MODEL --logdir $LOGDIR > /dev/null 2> $LOGDIR/PiPiDaiCharTrainErr.log &
nohup python3 run.py --dataset PiPiDai --word-segment word --model $MODEL --logdir $LOGDIR > /dev/null 2> $LOGDIR/PiPiDaiWordTrainErr.log &
# nohup python3 run.py --dataset PiPiDai --word-segment char --model $MODEL --not-train-embed --logdir $LOGDIR > /dev/null 2> $LOGDIR/PiPiDaiCharFixedErr.log &
# nohup python3 run.py --dataset PiPiDai --word-segment word --model $MODEL --not-train-embed --logdir $LOGDIR > /dev/null 2> $LOGDIR/PiPiDaiWordFixedErr.log &

echo "Running scripts"
ps aux | grep run.py | grep $MODEL
echo "To stop all processes execute:"
echo "kill -9" `ps x | grep run.py | grep -v grep | awk '{print $1}' | xargs`
