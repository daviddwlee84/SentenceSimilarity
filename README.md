# Enhanced RCNN

## Getting Started

```sh
# Data preprocessing
python3 preprocess.py [word/char] train

# Training
## Ant
python3 run.py --dataset Ant --model [ERCNN/Transformer] --mode train --word-segment [word/char]
## Quora
python3 run.py --dataset Quora --model [ERCNN/Transformer] --mode train

# Test (using entire training set)
## Ant
python3 run.py --dataset Ant --model [ERCNN/Transformer] --mode test --word-segment [word/char]
## Quora
python3 run.py --dataset Quora --model [ERCNN/Transformer] --mode test

# Predict (input two sentence manually)
## Ant
python3 run.py --dataset Ant --model [ERCNN/Transformer] --mode predict --word-segment [word/char]
## Quora
python3 run.py --dataset Quora --model [ERCNN/Transformer] --mode predict
```

## Data

Original

* `raw_data/competition_train.csv` - Ant Financial
* `word2vec/substoke_char.vec.avg` - Ant Financial
* `word2vec/substoke_word.vec.avg` - Ant Financial
* `word2vec/glove.word2vec.txt` - Quora Question Pairs

  ```sh
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
  unzip glove.840B.300d
  ```

  ```py
  from gensim.scripts.glove2word2vec import glove2word2vec
  _ = glove2word2vec('glove.840B.300d.txt', 'word2vec/glove.word2vec.txt')
  ```

  ```sh
  rm glove.840B*
  ```

Generated

* `data/sentence_char_train.csv` - Ant Financial
* `data/sentence_word_train.csv` - Ant Financial
* `model/*`
* `word2vec/Ant_tokenizer.pickle` - Ant Financial
* `word2vec/Quora_tokenizer.pickle` - Quora Question Pairs

Not Sure

* `data/stopwords.txt` - Ant Financial
* `data/test.csv` - Ant Financial

## Dataset

### ANT Financial Competition

* [Original competition](https://dc.cloud.alipay.com/index#/topic/intro?id=3)
* [Long-term competition with same topic](https://dc.cloud.alipay.com/index#/topic/intro?id=8)

Goal: classify whether two question sentences are asking the same thing => predict true or false

Evaluation: **f1-score**

Data

* Positive data: 18.23%

### Quora Question Pairs

```sh
kaggle competitions download -c quora-question-pairs
unzip test.csv -d raw_data
unzip train.csv -d raw_data
rm *.zip
```

Goal: classify whether question pairs are duplicates or not => predict the probability that the questions are duplicates (a number between 0 and 1)

Evaluation: **log loss** between the predicted values and the ground truth

* [Kaggle - Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
* [Quora - First Quora Dataset Release: Question Pairs](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)

Data

* Positive data: 36.92%
* 400K rows in train set and about 2.35M rows in test set
* 6 columns in train set but only 3 of them are in test set
  * train set
    * id - the id of a training set question pair
    * qid1, qid2 - unique ids of each question (only available in train.csv)
    * question1, question2 - the full text of each question
    * is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise
  * test set
    * test_id
    * question1, question2
* about 63% non-duplicate questions and 37% duplicate questions in the training data set

## Experiment

### Ant Financial

(during training)

| Model             | Word Segment | Batch Preprocessing | Epoch | Average Loss | Accuracy |
| ----------------- | ------------ | ------------------- | ----- | ------------ | -------- |
| Random            | -            | -                   | -     | -            | 82%      |
| ERCN              | word         | none                | 10    | 0.4611       | 82%      |
| ERCN              | char         | none                | 10    | 0.4611       | 82%      |
| ERCNN-Transformer | char         | none                | 10    | 0.4128       | 83%      |

### Quora

(during training)

| Model             | Batch Preprocessing | Epoch | Average Loss | Accuracy |
| ----------------- | ------------------- | ----- | ------------ | -------- |
| Random            | -                   | -     | -            | 63%      |
| ERCNN             | none                | 10    | 0.4094       | 80%      |
| ERCNN-Transformer | none                | 5     | 10.2011      | 63%      |

## TODO

> Test the test during training: `python3 run.py --mode train --word-segment word --log-interval 1 --test-interval 1`

* Pure Test
* More evaluation matrics: recall & f1-score
* Data Generator
  * generate batch with half positive and negative sample

## Notes about Virtualenv

```sh
# this will create a env_name folder in current directory
virtualenv python=/path/to/python3.x env_name

# activate the environment
source ./env_name/bin/activate
```

Add alias in bashrc

* Goto work directory and activate the environment
  * `alias davidlee="cd /home/username/working_dir; source env_name/bin/activate"`
* Use pip source when install packages
  * `alias pipp="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple"`

## Links

### PyTorch

* [torch](https://pytorch.org/docs/stable/torch.html)
  * torch.Tensor vs torch.tensor (the second one can specify the `dtype`)
    * `torch.LongTensor(var) == torch.tensor(var, dtype=torch.long)`
  * [torch.max](https://pytorch.org/docs/stable/torch.html#torch.max)
    * return a tuple `(Tensor, LongTensor)`
* torch.nn
  * Containers
    * [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#module)
      * class to inherit from
      * must implement `__init__()` with `super()` and the `forward()`
    * [torch.nn.Sequential](https://pytorch.org/docs/stable/nn.html#sequential)
  * Sparse Layers
    * [torch.nn.Embedding](https://pytorch.org/docs/stable/nn.html#embedding)
  * Recurrent Layers
    * [torch.nn.GRU](https://pytorch.org/docs/stable/nn.html#gru)
  * Linear Layers
    * [torch.nn.Linear](https://pytorch.org/docs/stable/nn.html#linear)
  * Non-linear activations (weighted sum, nonlinearity)
    * [torch.nn.ReLU](https://pytorch.org/docs/stable/nn.html#relu)
* [torch.optim](https://pytorch.org/docs/stable/optim.html)
  * `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False`
* [How to do product of matrices in PyTorch](https://stackoverflow.com/questions/44524901/how-to-do-product-of-matrices-in-pytorch)

### Gensim

* [scripts.glove2word2vec – Convert glove format to word2vec](https://radimrehurek.com/gensim/scripts/glove2word2vec.html)

### Others

* [Pytorch equivalent of Keras](https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2)
  * keras.layers.Dense => torch.nn.Linear
* [Same implementation different results between Keras and PyTorch - lstm](https://discuss.pytorch.org/t/same-implementation-different-results-between-keras-and-pytorch-lstm/39146)
* [Convert Pandas dataframe to PyTorch tensor?](https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor)
