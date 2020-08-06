# Sentence Similarity

(mainly based on Enhanced-RCNN model and other baselines)

## Getting Started

Clone this project (make sure `git-lfs` is installed)

> Sorry for the limitation of the [Git-LFS bandwidth quota](https://docs.github.com/en/github/managing-large-files/about-storage-and-bandwidth-usage), might have some problem to clone this project.

```sh
git lfs clone --depth=1 https://github.com/daviddwlee84/SentenceSimilarity.git
```

Quick Execute All

```sh
# Data preprocessing
./all_data_preprocess.sh
# Train & Evaluate
./train_all_data_at_once.sh [model name]

# Test Ant Submission functionality
bash run.sh raw_data/competition_train.csv ant_test_pred.csv
# pack the Ant Submission files
zip -r AntSubmit.zip . -i \*.py \*.sh -i data/stopwords.txt
```

Usage

```sh
# Data preprocessing
## Ant
python3 ant_preprocess.py [word/char] train
## CCKS
python3 ccks_preprocess.py
## PiPiDai
python3 pipidai_preprocess.py

# Train & Evaluate
## Chinese
python3 run.py --dataset [Ant/CCKS/PiPiDai] --model [model name] --word-segment [word/char]
# train all the model at once use ./train_all_data_at_once.sh
## English
python3 run.py --dataset Quora --model [model name]

# Use Tensorboard
tensorboard --logdir log/same_as_model_log_dir
## remote connection(forward local port to remote port) (execute in local machine)
## then you should be able to access with http://localhost:$LOCAL_PORT
ssh -NfL $LOCAL_PORT:localhost:$REMOTE_PORT $REMOTE_USER@$REMOTE_IP > /dev/null 2>&1
### to close connection (just kill the ssh command which run in background)
ps aux | grep "ssh -NfL" | grep -v grep | awk '{print $2}' | xargs kill
```

Model

* `ERCNN` (default)
* `Transformer`
  * ERCNN + replace the BiRNN with Transformer
* Baseline
  * Siamese Series
    * `SiameseCNN`
      * Convolutional Neural Networks for Sentence Classification
      * Character-level Convolutional Networks for Text Classification
    * `SiameseRNN`
    * `SiameseLSTM`
      * Siamese Recurrent Architectures for Learning Sentence Similarity
    * `SiameseRCNN`
      * Siamese Recurrent Architectures for Learning Sentence Similarity
    * `SiameseAttentionRNN`
      * Text Classification Research with Attention-based Recurrent Neural Networks
  * Multi-Perspective Series
    * `MPCNN`
      * Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks
      * just a "more sentence similarity measurements" version of SiameseCNN (also use Siamese network to encode sentences)
      * TODO: Model too big to run.... (consume too much GPU memory) => Smaller batch size
    * `MPLSTM`: skip
    * `BiMPM`
      * Bilateral Multi-Perspective Matching for Natural Language Sentences
  * `ESIM`

Dataset

* `Ant` - Chinese
* `CCKS` - Chinese
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
  -h, --help             show this help message and exit
  --dataset dataset      Chinese: Ant, CCKS; English: Quora (default: Ant)
  --mode mode            script mode [train/test/both/predict/submit(Ant)]
                         (default: both)
  --sampling mode        sampling mode during training (default: random)
  --generate-train       use generated negative samples when training (used in
                         balance sampling)
  --generate-test        use generated negative samples when testing (used in
                         balance sampling)
  --model model          model to use [ERCNN/Transformer/Siamese(CNN/RNN/LSTM/R
                         CNN/AttentionRNN)] (default: ERCNN)
  --word-segment WS      chinese word split mode [word/char] (default: char)
  --chinese-embed embed  chinese embedding (default: cw2vec)
  --not-train-embed      whether to freeze the embedding parameters
  --batch-size N         input batch size for training (default: 256)
  --test-batch-size N    input batch size for testing (default: 1000)
  --k-fold N             k-fold cross validation i.e. number of epochs to train
                         (default: 10)
  --lr N                 learning rate (default: 0.001)
  --beta1 N              beta 1 for Adam optimizer (default: 0.9)
  --beta2 N              beta 2 for Adam optimizer (default: 0.999)
  --epsilon N            epsilon for Adam optimizer (default: 1e-08)
  --no-cuda              disables CUDA training
  --seed N               random seed (default: 16)
  --test-split N         test data split (default: 0.3)
  --logdir path          set log directory (default: ./log)
  --log-interval N       how many batches to wait before logging training
                         status
  --test-interval N      how many batches to test during training
  --not-save-model       for not saving the current model
  --load-model name      load the specific model checkpoint file
  --submit-path path:    submission file path (currently for Ant dataset)
```

> Related Additional Datasets
>
> * [xiaohai-AI/lcqmc_data](https://github.com/xiaohai-AI/lcqmc_data)
> * [zzy99/epidemic-sentence-pair](https://github.com/zzy99/epidemic-sentence-pair)
> * [IAdmireu/ChineseSTS](https://github.com/IAdmireu/ChineseSTS)

## Data

Original

* `raw_data/competition_train.csv` - Ant Financial
* `raw_data/train.csv` - Quora Question Pairs
* `word2vec/substoke_char.vec.avg` - Ant Financial
* `word2vec/substoke_word.vec.avg` - Ant Financial
* `data/stopwords.txt` - Ant Financial
* `word2vec/glove.word2vec.txt` - Quora Question Pairs
* `raw_data/task3_train.txt` - CCKS 2018
* `raw_data/task3_dev.txt` - CCKS 2018

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

* [Jupyter Notebook - Data Analysis](DataAnalysis.ipynb)
  * `jupyter notebook DataAnalysis.ipynb`

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

### CCKS 2018

> CCKS: China Conference on Knowledge Graph and Semantic Computing

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

## Notes

### Notes for unbalanced data

#### Balance data generator

> In `data_prepare.py`, the `class BalanceDataHelper`

#### Use different loss

* Dice loss
  * [Dice Loss PR · Issue #1249 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/1249)
  * other approach

    ```py
    if weight is None:
            weight = torch.ones(
                y_pred.shape[-1], dtype=torch.float).to(device=y_pred.device)  # (C)
        if not mode:
            return self.simple_cross_entry(y_pred, golden, seq_mask, weight)
        probs = nn.functional.softmax(y_pred, dim=2)  # (B, T, C)
        B, T, C = probs.shape

        golden_index = golden.unsqueeze(dim=2)  # (B, T, 1)
        golden_probs = torch.gather(
            probs, dim=2, index=golden_index)  # (B, T, 1)

        probs_in_package = golden_probs.expand(B, T, T).transpose(1, 2)

        packages = np.array([np.eye(T)] * B)  # (B, T, T)
        probs_in_package = probs_in_package * \
            torch.tensor(packages, dtype=torch.float).to(device=probs.device)
        max_probs_in_package, _ = torch.max(probs_in_package, dim=2)

        golden_probs = golden_probs.squeeze(dim=2)

        golden_weight = golden_probs / (max_probs_in_package)  # (B, T)

        golden_weight = golden_weight.view(-1)
        golden_weight = golden_weight.detach()
        y_pred = y_pred.view(-1, C)
        golden = golden.view(-1)
        seq_mask = seq_mask.view(-1)

        negative_label = torch.tensor(
            [0] * (B * T), dtype=torch.long, device=y_pred.device)
        golden_loss = nn.functional.cross_entropy(
            y_pred, golden, weight=weight, reduction='none')
        negative_loss = nn.functional.cross_entropy(
            y_pred, negative_label, weight=weight, reduction='none')

        loss = golden_weight * golden_loss + \
            (1 - golden_weight) * negative_loss  # (B * T)
        loss = torch.dot(loss, seq_mask) / (torch.sum(seq_mask) + self.epsilon)
    ```

* Triplet-Loss
* N-pair Loss

### Notes about Virtualenv

```sh
# this will create a env_name folder in current directory
virtualenv --python=/path/to/python3.x env_name

# activate the environment
source ./env_name/bin/activate
```

> * [Unable to install pyhton 3.7 version on ubuntu 16.04 error Couldn't find any package by regex 'python3.7 | DigitalOcean](https://www.digitalocean.com/community/questions/unable-to-install-pyhton-3-7-version-on-ubuntu-16-04-error-couldn-t-find-any-package-by-regex-python3-7)

Add alias in bashrc

* Goto work directory and activate the environment
  * `alias davidlee="cd /home/username/working_dir; source env_name/bin/activate"`
* Use pip source when install packages
  * `alias pipp="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple"`

Install Jupyter notebook use the virtualenv kernel

1. make sure you activate the environment
2. `pip3 install jupyterlab`
3. `python3 -m ipykernel install --user --name=python3.6virtualenv`
4. execute jupyter notebook as normal `jupyter notebook`
5. Goto kernel > change kernel > select `python3.6virtualenv`

> * [Virtualenv in IPython Jupyter Notebook](https://zhuanlan.zhihu.com/p/33257881)

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

Summary

* [**brightmart/nlu_sim**](https://github.com/brightmart/nlu_sim) - all kinds of baseline models for sentence similarity
  * Imbalance Classification for Skew Data
  * Models including ESIM
* [tuzhucheng/MP-CNN-Variants: Variants of Multi-Perspective Convolutional Neural Networks](https://github.com/tuzhucheng/MP-CNN-Variants)

Model Source Code

* [ESIM](https://github.com/HsiaoYetGun/ESIM)
* BiMPM
  * [BiMPM](https://github.com/zhiguowang/BiMPM) - TensorFlow
  * [galsang/BIMPM-pytorch](https://github.com/galsang/BIMPM-pytorch) - PyTorch
* [Milti-Perspective-CNN](https://github.com/castorini/MP-CNN-Torch)
  * [pytorch implementation](https://github.com/tuzhucheng/MP-CNN-Variants)

--

* [tlatkowski/multihead-siamese-nets](https://github.com/tlatkowski/multihead-siamese-nets) Implementation of Siamese Neural Networks built upon multihead attention mechanism for text semantic similarity task

### Article

* [【金融大腦-一支優秀的隊伍】比賽經驗分享](https://openclub.alipay.com/club/history/read/9106)
* [問題對語義相似度計算-參賽總結](http://www.zhuzongkui.top/2018/08/10/competition-summary/?nsukey=pP8wE99JdxhVIRBers958wxK101Qowhy%2B7NNjtgQZoEamy6LNx7T4rtO8LzwlMSocnlO5eW4D1NRExTJ61iafIN0fUD6etae3r5dIXkdehi0Mu0wtucwpaQO3iFlYmMPb6BNJZiCa%2FI4R0%2F2u7jAhTqH4yIRUCkHogG6E2wvqncsl4eju4hKOdHO8pS%2FbXEuQJueu4J%2BEk%2Bau2fWUFSvmA%3D%3D)
* [Kaggle文本語義相似度計算Top5解決方案分享](https://www.sohu.com/a/287860625_787107) - Dataset links
  * [從Kaggle賽題: Quora Question Pairs 看文本相似性/相關性](https://zhuanlan.zhihu.com/p/35093355)

### Candidate Set

* [Glyph-vectors](https://arxiv.org/pdf/1901.10125.pdf)
  * [ShannonAI/glyce: Code for NeurIPS 2019 - Glyce: Glyph-vectors for Chinese Character Representations](https://github.com/ShannonAI/glyce)
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

## Appendix

### Attention

```py
# https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(
            1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim=2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
```
