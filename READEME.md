# ANT Financial Competition

* [Original competition](https://dc.cloud.alipay.com/index#/topic/intro?id=3)
* [Long-term competition with same topic](https://dc.cloud.alipay.com/index#/topic/intro?id=8)

## Getting Started

```sh
# Data preprocessing
python3 preprocess.py [word/char] train

# Training
python3 run.py --mode train --word-segment [word/char]
```

## Data

Original

* `raw_data/competition_train.csv`
* `word2vec/substoke_char.vec.avg`
* `word2vec/substoke_word.vec.avg`

Generated

* `data/sentence_char_train.csv`
* `data/sentence_word_train.csv`
* `model/*`
* `word2vec/tokenizer.pickle`

Not Sure

* `data/stopwords.txt`
* `data/test.csv`

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

### Others

* [Pytorch equivalent of Keras](https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2)
  * keras.layers.Dense => torch.nn.Linear
* [Same implementation different results between Keras and PyTorch - lstm](https://discuss.pytorch.org/t/same-implementation-different-results-between-keras-and-pytorch-lstm/39146)
* [Convert Pandas dataframe to PyTorch tensor?](https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor)
