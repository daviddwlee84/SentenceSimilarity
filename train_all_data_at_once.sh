#!/bin/bash
# ./train_all_data_at_once Model Shared_Args

MODEL=${1:-'ERCNN'}
LOGDIR=log/${MODEL}_All_`date +"%m_%d-%H_%M_%S"`
# for larger model use smaller batch
BATCH_SIZE_CONF='--batch-size 128 --test-batch-size 256'

CUSTOM_ARGS=${2:-''}
SHARED_ARGS="--model $MODEL --logdir $LOGDIR $BATCH_SIZE_CONF $CUSTOM_ARGS"

mkdir -p $LOGDIR

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1

# export CUDA_VISIBLE_DEVICES=0
# Ant
nohup python3 run.py --dataset Ant --word-segment char --lr 0.001 $SHARED_ARGS > /dev/null 2> $LOGDIR/AntCharTrainErr.log &
nohup python3 run.py --dataset Ant --word-segment word --lr 0.001 $SHARED_ARGS > /dev/null 2> $LOGDIR/AntWordTrainErr.log &

# CCKS
nohup python3 run.py --dataset CCKS --word-segment char --lr 0.001 $SHARED_ARGS > /dev/null 2> $LOGDIR/CCKSCharTrainErr.log &
nohup python3 run.py --dataset CCKS --word-segment word --lr 0.001 $SHARED_ARGS > /dev/null 2> $LOGDIR/CCKSWordTrainErr.log &

# export CUDA_VISIBLE_DEVICES=1
# PiPiDai
nohup python3 run.py --dataset PiPiDai --word-segment char --lr 0.001 $SHARED_ARGS > /dev/null 2> $LOGDIR/PiPiDaiCharTrainErr.log &
nohup python3 run.py --dataset PiPiDai --word-segment word --lr 0.001 $SHARED_ARGS > /dev/null 2> $LOGDIR/PiPiDaiWordTrainErr.log &

echo "== Running scripts =="
ps aux | grep run.py | grep $MODEL
echo "To stop all processes execute:"
echo "kill -9" `ps x | grep run.py | grep -v grep | awk '{print $1}' | xargs`

echo "== Tensorboard =="
echo "Tensorboard should be run at port 6006"
echo "To terminate it just press CTRL+C to quit"
tensorboard --logdir $LOGDIR/.
