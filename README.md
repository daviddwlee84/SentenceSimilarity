# Enhanced RCNN

## Getting Started

Clone this project (make sure `git-lfs` is installed)

```sh
git lfs clone --depth=1 https://github.com/daviddwlee84/Enhanced-RCNN.git
```

Quick Execute All

```sh
./all_data_preprocess.sh
./train_all_data_at_once.sh [model name]
```

Usage

```sh
# Data preprocessing
## Ant
python3 ant_preprocess.py [word/char] train
## CCSK
python3 ccsk_preprocess.py
## PiPiDai
python3 pipidai_preprocess.py

# Train & Evaluate
## Chinese
python3 run.py --dataset [Ant/CCSK/PiPiDai] --model [model name] --word-segment [word/char]
# train all the model at once use ./train_all_data_at_once.sh
## English
python3 run.py --dataset Quora --model [model name]
```

Model

* `ERCNN` (default)
* `Transformer`
  * ERCNN + replace the BiRNN with Transformer
* Baseline
  * Siamese Series
    * `SiameseCNN`

Dataset

* `Ant` - Chinese
* `CCSK` - Chinese
* `PiPiDai` - Chinese (encoded)
* `Quora` - English

Mode

* `train`
  * using 70% training data
  * k-fold cross-validation (k == training epochs)
  * will test the performance using valid set when each epoch end and save the model
* `test`
  * using 30% test data
  * will load the latest model with the same settings
* `both` (include train and test)
* `predict`
  * will load the latest model with the same settings

Sampling

* `random` (Original): data is skewed (the ratio is listed below)
* `balance`: positive vs. negative data will be the same
  * `generate-train`
  * `generate-test`

```txt
$ python3 run.py --help
usage: run.py [-h] [--dataset dataset] [--mode mode] [--sampling mode]
              [--generate-train] [--generate-test] [--model model]
              [--word-segment WS] [--batch-size N] [--test-batch-size N]
              [--k-fold N] [--lr N] [--beta1 N] [--beta2 N] [--epsilon N]
              [--no-cuda] [--seed N] [--test-split N] [--log-interval N]
              [--test-interval N] [--not-save-model]

Enhanced RCNN on Sentence Similarity

optional arguments:
  -h, --help           show this help message and exit
  --dataset dataset    [Ant] Finance or [Quora] Question Pairs (default: Ant)
  --mode mode          script mode [train/test/both/predict] (default: both)
  --sampling mode      sampling mode during training (default: balance)
  --generate-train     use generated negative samples when training (used in
                       balance sampling)
  --generate-test      use generated negative samples when testing (used in
                       balance sampling)
  --model model        model to use [ERCNN/Transformer] (default: ERCNN)
  --word-segment WS    chinese word split mode [word/char] (default: char)
  --batch-size N       input batch size for training (default: 256)
  --test-batch-size N  input batch size for testing (default: 1000)
  --k-fold N           k-fold cross validation i.e. number of epochs to train
                       (default: 10)
  --lr N               learning rate (default: 0.001)
  --beta1 N            beta 1 for Adam optimizer (default: 0.9)
  --beta2 N            beta 2 for Adam optimizer (default: 0.999)
  --epsilon N          epsilon for Adam optimizer (default: 1e-08)
  --no-cuda            disables CUDA training
  --seed N             random seed (default: 16)
  --test-split N       test data split (default: 0.3)
  --log-interval N     how many batches to wait before logging training status
  --test-interval N    how many batches to test during training
  --not-save-model     for not saving the current model
```

## Data

Original

* `raw_data/competition_train.csv` - Ant Financial
* `raw_data/train.csv` - Quora Question Pairs
* `word2vec/substoke_char.vec.avg` - Ant Financial
* `word2vec/substoke_word.vec.avg` - Ant Financial
* `data/stopwords.txt` - Ant Financial
* `word2vec/glove.word2vec.txt` - Quora Question Pairs
* `raw_data/task3_train.txt` - CCSK 2018
* `raw_data/task3_dev.txt` - CCSK 2018

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
* `word2vec/Ant_char_tokenizer.pickle` - Ant Financial
* `word2vec/Ant_char_embed_matrix.pickle` - Ant Financial
* `word2vec/Ant_word_tokenizer.pickle` - Ant Financial
* `word2vec/Ant_word_embed_matrix.pickle` - Ant Financial
* `word2vec/Quora_tokenizer.pickle` - Quora Question Pairs
* `word2vec/Quora_embed_matrix.pickle` - Quora Question Pairs
* `model/*`
* `log/*`

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

### CCSK 2018

* [微眾銀行智能客服問句匹配大賽](https://biendata.com/competition/CCKS2018_3/)

Data

* Positive data: 50%
* Data amount: 100000

### CHIP 2018

* [第四屆中國健康信息處理會議](https://biendata.com/competition/chip2018/)

> 須連繫主辦方才能取得數據

### PiPiDai

* [第三屆魔鏡杯大賽](https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=1)

> Link失效
>
> * [LittletreeZou/Question-Pairs-Matching](https://github.com/LittletreeZou/Question-Pairs-Matching)

* Positive data: 52%
* Data amount: 254386

## [Experiment](Experiment.md)

## TODO

* More evaluation matrics: recall & f1-score
* Continue training?!
* Potential multi-class classification
  * num_class input
  * sigmoid => softmax
  * (but how about siamese model??)

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

### Paper

* [Neural Network Models for Paraphrase Identification, Semantic Textual Similarity, Natural Language Inference, and Question Answering](https://arxiv.org/pdf/1806.04330.pdf)

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
* [Check the total number of parameters in a PyTorch model](https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model)

### Gensim

* [scripts.glove2word2vec – Convert glove format to word2vec](https://radimrehurek.com/gensim/scripts/glove2word2vec.html)

### Others

* [Pytorch equivalent of Keras](https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2)
  * keras.layers.Dense => torch.nn.Linear
* [Same implementation different results between Keras and PyTorch - lstm](https://discuss.pytorch.org/t/same-implementation-different-results-between-keras-and-pytorch-lstm/39146)
* [Convert Pandas dataframe to PyTorch tensor?](https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor)

### Related Project

* [**brightmart/nlu_sim**](https://github.com/brightmart/nlu_sim) - all kinds of baseline models for sentence similarity
  * Imbalance Classification for Skew Data
  * Models including ESIM
* [**ESIM**](https://github.com/HsiaoYetGun/ESIM)

### Article

* [【金融大腦-一支優秀的隊伍】比賽經驗分享](https://openclub.alipay.com/club/history/read/9106)
* [問題對語義相似度計算-參賽總結](http://www.zhuzongkui.top/2018/08/10/competition-summary/?nsukey=pP8wE99JdxhVIRBers958wxK101Qowhy%2B7NNjtgQZoEamy6LNx7T4rtO8LzwlMSocnlO5eW4D1NRExTJ61iafIN0fUD6etae3r5dIXkdehi0Mu0wtucwpaQO3iFlYmMPb6BNJZiCa%2FI4R0%2F2u7jAhTqH4yIRUCkHogG6E2wvqncsl4eju4hKOdHO8pS%2FbXEuQJueu4J%2BEk%2Bau2fWUFSvmA%3D%3D)
* [Kaggle文本語義相似度計算Top5解決方案分享](https://www.sohu.com/a/287860625_787107) - Dataset links
  * [從Kaggle賽題: Quora Question Pairs 看文本相似性/相關性](https://zhuanlan.zhihu.com/p/35093355)

### Candidate Set

* [Glyph-vectors](https://arxiv.org/pdf/1901.10125.pdf)
* Transformer
* [Capsule Network](https://en.wikipedia.org/wiki/Capsule_neural_network)
  * [Capsule Networks: The New Deep Learning Network](https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8)
* [cw2vec](http://www.statnlp.org/wp-content/uploads/papers/2018/cw2vec/cw2vec.pdf)
  * [github](https://github.com/bamtercelboo/cw2vec)

### Baseline

#### Siamese Models

> Siamese-CNN, Siamese-RNN, Siamese-LSTM, Siamese-RCNN, Siamese-Attention-RCNN

* [**ShawnyXiao/TextClassification-Keras**](https://github.com/ShawnyXiao/TextClassification-Keras): Text classification models implemented in Keras, including: FastText, TextCNN, TextRNN, TextBiRNN, TextAttBiRNN, HAN, RCNN, RCNNVariant, etc.
* [akshaysharma096/Siamese-Networks: Few Shot Learning by Siamese Networks, using Keras.](https://github.com/akshaysharma096/Siamese-Networks)
* [Siamese Networks: Algorithm, Applications And PyTorch Implementation](https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18)

Contrastive Loss

* [pytorch-siamese/contrastive.py at master · delijati/pytorch-siamese](https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py)
* [**Siamese Neural Network ( With Pytorch Code Example ) - Innovation Incubator Group of Companies**](https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/)
* [siamese-network/ContrastiveLoss.py at master · melody-rain/siamese-network](https://github.com/melody-rain/siamese-network/blob/master/models/ContrastiveLoss.py)

## Trouble Shooting

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

> somehow the `nn.Module` in a list can't be auto connect `to(device)`
