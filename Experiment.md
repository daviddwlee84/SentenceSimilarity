# Experiment

## Baseline

* 10 fold cross-validation
* 7:3 train:test
* training using balance data, test using original data

> Valid Data: the performance of cross-validation on the last epoch

---

## Deprecated

Fixed embedding

> Complexity: trainable/all parameters

| Model      | Complexity    | Dataset | Embedding       | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ---------- | ------------- | ------- | --------------- | -------- | --------- | -------- | -------- | ------- |
| SiameseCNN | 38601/251501  | Ant     | cw2vec (fixed)  | char     | 67.24%    | -        | 50.54%   | -       |
| SiameseCNN | 38601/1354601 | Ant     | cw2vec (fixed)  | word     | 66.90%    | -        | 49.64%   | -       |
| SiameseCNN | 38601/185701  | CCSK    | cw2vec (fixed)  | char     | 68.55%    | -        | 64.21%   | -       |
| SiameseCNN | 38601/617301  | CCSK    | cw2vec (fixed)  | word     | 67.15%    | -        | 64.59%   | -       |
| SiameseCNN | 38601/717501  | PiPiDai | PiPiDai (fixed) | char     | 62.46%    | -        | 60.69%   | -       |
| SiameseCNN | 38601/3409701 | PiPiDai | PiPiDai (fixed) | word     | 60.79%    | -        | 62.55%   | -       |

2019/9/22,23 Use Balanced Data

| Model      | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ---------- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
| SiameseCNN | 251501     | Ant     | cw2vec    | char     | 72.85%    | 73.57%   | 57.62%   | 50.76%  |
| SiameseCNN | 1354601    | Ant     | cw2vec    | word     | 74.00%    | 74.75%   | 56.67%   | 49.38%  |
| SiameseCNN | 185701     | CCSK    | cw2vec    | char     | 76.96%    | 77.61%   | 75.57%   | 75.50%  |
| SiameseCNN | 617301     | CCSK    | cw2vec    | word     | 77.31%    | 77.94%   | 77.43%   | 77.32%  |
| SiameseCNN | 717501     | PiPiDai | PiPiDai   | char     | 71.06%    | 71.15%   | 73.75%   | 73.67%  |
| SiameseCNN | 3409701    | PiPiDai | PiPiDai   | word     | 72.19%    | 71.85%   | 74.50%   | 74.38%  |

| Model | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ----- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
| ERCNN | 2722885    | Ant     | cw2vec    | char     | 95.12%    | 96.75%   | 61.94%   | 48.74%  |
| ERCNN | 3825985    | Ant     | cw2vec    | word     | 97.58%    | 99.26%   | 41.46%   | 39.42%  |
| ERCNN | 2657085    | CCSK    | cw2vec    | char     | 49.39%    | 33.33%   | 50.02%   | 33.34%  |
| ERCNN | 3088685    | CCSK    | cw2vec    | word     | 95.07%    | 96.25%   | 51.65%   | 48.79%  |
| ERCNN | 3649685    | PiPiDai | PiPiDai   | char     | 95.34%    | 95.77%   | 52.87%   | 49.33%  |
| ERCNN | 6341885    | PiPiDai | PiPiDai   | word     | 83.31%    | 83.25%   | 55.94%   | 55.59%  |

| Model       | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ----------- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
| Transformer | 913493     | Ant     | cw2vec    | char     | 69.20%    | 68.11%   | 75.70%   | 62.77%  |
| Transformer | 2016593    | Ant     | cw2vec    | word     | 74.65%    | 74.64%   | 73.25%   | 62.90%  |
| Transformer | 847693     | CCSK    | cw2vec    | char     | 77.61%    | 78.10%   | 72.78%   | 72.63%  |
| Transformer | 1279293    | CCSK    | cw2vec    | word     | 78.95%    | 79.89%   | 72.32%   | 71.95%  |
| Transformer | 3473093    | PiPiDai | PiPiDai   | char     | 49.77%    | 33.33%   | 51.72%   | 34.09%  |
| Transformer | 6165293    | PiPiDai | PiPiDai   | word     | 49.77%    | 33.33%   | 51.72%   | 34.09%  |

> 1. Basically Transformer predict everything to be positive in PiPiDai
> 2. Basically Transformer predict everything to be positive in PiPiDai

---

### Ant Financial

(during training)

| Model             | Word Segment | Embedding     | Batch Preprocessing | Epoch | Average Loss | Accuracy | Remark                              |
| ----------------- | ------------ | ------------- | ------------------- | ----- | ------------ | -------- | ----------------------------------- |
| Random            | -            | -             | -                   | -     | -            | 82%      | -                                   |
| Paper (rejected)  | char         | cw2vec (100d) | -                   | -     | -            | 76.89%   | not sure how dev set been generated |
| original Keras    | word         | cw2vec (100d) | none                | -     | -            | 83%      | learned nothing???                  |
| ERCN              | word         | cw2vec (100d) | none                | 10    | 0.4611       | 82%      | learned nothing                     |
| ERCN              | char         | cw2vec (100d) | none                | 10    | 0.4611       | 82%      | learned nothing                     |
| ERCNN-Transformer | char         | cw2vec (100d) | none                | 10    | 0.4128       | 83%      | learned nothing                     |

### Quora

(during training)

| Model             | Embedding    | Batch Preprocessing | Epoch | Average Loss | Accuracy | Remark                                             |
| ----------------- | ------------ | ------------------- | ----- | ------------ | -------- | -------------------------------------------------- |
| Random            | GloVe (300d) | -                   | -     | -            | 63%      | -                                                  |
| Paper (rejected)  | GloVe (300d) | -                   | -     | -            | 88.15%   | not sure how dev set been generated                |
| ERCNN             | GloVe (300d) | none                | 10    | 0.4094       | 80%      | using the same model... but this learned something |
| ERCNN-Transformer | GloVe (300d) | none                | 5     | 10.2011      | 63%      | learned nothing (to be improved)                   |
