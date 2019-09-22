#!/bin/bash
# Ant
python3 ant_preprocess.py char train
python3 ant_preprocess.py word train
# CCSK
python3 ccsk_preprocess.py
# PiPiDai
python3 pipidai_preprocess.py
