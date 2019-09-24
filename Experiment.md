# Experiment

## Baseline

* 10 fold cross-validation
* 7:3 train:test
* training and testing use original data
* embedding dimension
  * cw2vec: 100
  * PiPiDai: 300

> Valid Data: the performance of cross-validation on the last epoch

| Model      | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ---------- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
| SiameseCNN | 4533869    | Ant     | cw2vec    | char     | 81.85%    | 45.01%   | 81.57%   | 44.93%  |
| SiameseCNN | 5636969    | Ant     | cw2vec    | word     | 81.85%    | 45.01%   | 81.57%   | 44.93%  |
| SiameseCNN | 4468069    | CCSK    | cw2vec    | char     | 78.03%    | 77.67%   | 73.81%   | 73.38%  |
| SiameseCNN | 4899669    | CCSK    | cw2vec    | word     | 77.87%    | 77.28%   | 72.04%   | 71.24%  |
| SiameseCNN | 7097021    | PiPiDai | PiPiDai   | char     | 51.99%    | 34.21%   | 51.72%   | 34.09%  |
| SiameseCNN | 9789221    | PiPiDai | PiPiDai   | word     | 51.99%    | 34.09%   | 51.72%   | 34.09%  |

* this model predict everything to be positive on PiPiDai dataset...

> 2019/9/23 Old SiameseCNN (using TextCNN structure)
>
> | Model      | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
> | ---------- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
> | SiameseCNN | 251501     | Ant     | cw2vec    | char     | 81.88%    | 45.17%   | 81.58%   | 45.03%  |
> | SiameseCNN | 1354601    | Ant     | cw2vec    | word     | 83.35%    | 53.69%   | 81.45%   | 45.96%  |
> | SiameseCNN | 185701     | CCSK    | cw2vec    | char     | 84.28%    | 84.25%   | 75.07%   | 74.96%  |
> | SiameseCNN | 617301     | CCSK    | cw2vec    | word     | 87.78%    | 87.78%   | 77.43%   | 77.38%  |
> | SiameseCNN | 717501     | PiPiDai | PiPiDai   | char     | 73.30%    | 73.18%   | 68.93%   | 68.81%  |
> | SiameseCNN | 3409701    | PiPiDai | PiPiDai   | word     | 84.17%    | 84.09%   | 74.68%   | 74.55%  |

| Model      | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ---------- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
| SiameseRNN | 288365     | Ant     | cw2vec    | char     | 81.85%    | 45.01%   | 81.57%   | 44.93%  |
| SiameseRNN | 1391465    | Ant     | cw2vec    | word     | 81.85%    | 45.01%   | 81.57%   | 44.93%  |
| SiameseRNN | 222565     | CCSK    | cw2vec    | char     | 64.98%    | 64.96%   | 63.15%   | 63.13%  |
| SiameseRNN | 654165     | CCSK    | cw2vec    | word     | 70.18%    | 70.17%   | 66.23%   | 66.19%  |
| SiameseRNN | 779965     | PiPiDai | PiPiDai   | char     | 69.99%    | 69.87%   | 68.10%   | 68.00%  |
| SiameseRNN | 3472165    | PiPiDai | PiPiDai   | word     | 73.36%    | 73.21%   | 70.76%   | 70.64%  |

| Model       | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ----------- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
| SiameseLSTM | 475757     | Ant     | cw2vec    | char     | 81.85%    | 45.01%   | 81.57%   | 44.93%  |
| SiameseLSTM | 1578857    | Ant     | cw2vec    | word     | 81.85%    | 45.01%   | 81.57%   | 44.93%  |
| SiameseLSTM | 409957     | CCSK    | cw2vec    | char     | 64.49%    | 64.42%   | 62.98%   | 62.92%  |
| SiameseLSTM | 841557     | CCSK    | cw2vec    | word     | 69.45%    | 69.44%   | 65.71%   | 65.70%  |
| SiameseLSTM | 1044157    | PiPiDai | PiPiDai   | char     | 67.58%    | 67.46%   | 66.51%   | 66.40%  |
| SiameseLSTM | 3736357    | PiPiDai | PiPiDai   | word     | 70.90%    | 70.86%   | 68.56%   | 68.53%  |

| Model | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ----- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
| ERCNN | 2722885    | Ant     | cw2vec    | char     | 81.85%    | 45.01%   | 81.57%   | 44.93%  |
| ERCNN | 3825985    | Ant     | cw2vec    | word     | 81.85%    | 45.01%   | 81.57%   | 44.93%  |
| ERCNN | 2657085    | CCSK    | cw2vec    | char     | 70.91%    | 70.90%   | 70.28%   | 70.25%  |
| ERCNN | 3088685    | CCSK    | cw2vec    | word     | 77.18%    | 77.18%   | 74.76%   | 74.74%  |
| ERCNN | 3649685    | PiPiDai | PiPiDai   | char     | 82.43%    | 82.25%   | 81.21%   | 81.04%  |
| ERCNN | 6341885    | PiPiDai | PiPiDai   | word     | 86.16%    | 86.11%   | 83.72%   | 83.67%  |

| Model       | Complexity | Dataset | Embedding | Word Seg | Valid Acc | Valid F1 | Test Acc | Test F1 |
| ----------- | ---------- | ------- | --------- | -------- | --------- | -------- | -------- | ------- |
| Transformer | 913493     | Ant     | cw2vec    | char     | 84.02%    | 60.68%   | 82.91%   | 58.95%  |
| Transformer | 2016593    | Ant     | cw2vec    | word     | 83.66%    | 57.37%   | 82.58%   | 55.36%  |
| Transformer | 847693     | CCSK    | cw2vec    | char     | 75.51%    | 75.46%   | 74.39%   | 74.34%  |
| Transformer | 1279293    | CCSK    | cw2vec    | word     | 79.53%    | 79.50%   | 78.06%   | 78.04%  |
| Transformer | 3473093    | PiPiDai | PiPiDai   | char     | 51.99%    | 34.21%   | 51.72%   | 34.09%  |
| Transformer | 6165293    | PiPiDai | PiPiDai   | word     | 51.99%    | 34.21%   | 51.72%   | 34.09%  |

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

* training using balance data, test using original data

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
> 2. When the model doesn't work, its loss will stick on 13.815511

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
