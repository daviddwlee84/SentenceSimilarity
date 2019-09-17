import numpy as np
import pandas as pd
import torch
import keras
import pickle
import os
from gensim.models.keyedvectors import KeyedVectors


def training_data_loader(mode="word", dataset="Ant"):
    if dataset == "Ant":
        data = pd.read_csv(f"data/sentence_{mode}_train.csv",
                           header=None, names=["doc1", "doc2", "label"])

        data["doc1"] = data.apply(lambda x: str(x[0]), axis=1)
        data["doc2"] = data.apply(lambda x: str(x[1]), axis=1)

        X1 = data["doc1"]
        X2 = data["doc2"]
        Y = data["label"]

    elif dataset == "Quora":
        data = pd.read_csv(f"raw_data/train.csv", encoding="utf-8")
        data['id'] = data['id'].apply(str)

        data['question1'].fillna('', inplace=True)
        data['question2'].fillna('', inplace=True)

        X1 = data['question1']
        X2 = data['question2']
        Y = data['is_duplicate']

    return X1, X2, Y


def embedding_loader(embedding_folder="word2vec", mode="word", dataset="Ant"):
    X1, X2, _ = training_data_loader(mode, dataset)

    if dataset == "Ant":
        tokenizer_pickle_file = f'{embedding_folder}/{dataset}_{mode}_tokenizer.pickle'
        embed_pickle_file = f'{embedding_folder}/{dataset}_{mode}_embed_matrix.pickle'
    elif dataset == "Quora":
        tokenizer_pickle_file = f'{embedding_folder}/{dataset}_tokenizer.pickle'
        embed_pickle_file = f'{embedding_folder}/{dataset}_embed_matrix.pickle'

    # Load tokenizer
    if os.path.isfile(tokenizer_pickle_file):
        with open(tokenizer_pickle_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(list(X1.values))
        tokenizer.fit_on_texts(list(X2.values))
        with open(tokenizer_pickle_file, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load embedding matrix
    if os.path.isfile(embed_pickle_file):
        with open(embed_pickle_file, 'rb') as handle:
            embeddings_matrix = pickle.load(handle)
    else:
        word_index = tokenizer.word_index

        if dataset == "Ant":
            embed_model = KeyedVectors.load_word2vec_format(
                f"{embedding_folder}/substoke_{mode}.vec.avg", binary=False, encoding='utf8')
        elif dataset == "Quora":
            embed_model = KeyedVectors.load_word2vec_format(
                f"{embedding_folder}/glove.word2vec.txt", binary=False, encoding='utf8')

        embeddings_matrix = np.zeros(
            (len(word_index) + 1, embed_model.vector_size))
        vocab_list = [(k, embed_model.wv[k])
                      for k, v in embed_model.wv.vocab.items()]

        for word, i in word_index.items():
            if word in embed_model:
                embedding_vector = embed_model[word]
            else:
                embedding_vector = None
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector

        with open(embed_pickle_file, 'wb') as handle:
            pickle.dump(embeddings_matrix, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer, torch.Tensor(embeddings_matrix)


def tokenize_and_padding(X1, X2, max_len, tokenizer=None, debug=False):
    list_tokenized_X1 = tokenizer.texts_to_sequences(X1)
    list_tokenized_X2 = tokenizer.texts_to_sequences(X2)
    if debug:
        print('Tokenized sentences:', list_tokenized_X1, '\t', list_tokenized_X2)

    padded_token_X1 = keras.preprocessing.sequence.pad_sequences(
        list_tokenized_X1, maxlen=max_len)
    padded_token_X2 = keras.preprocessing.sequence.pad_sequences(
        list_tokenized_X2, maxlen=max_len)
    if debug:
        print('Padded sentences:', padded_token_X1, '\t', padded_token_X2)

    return torch.tensor(padded_token_X1, dtype=torch.long), torch.tensor(padded_token_X2, dtype=torch.long)
