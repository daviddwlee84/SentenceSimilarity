import numpy as np
import pandas as pd
import torch
import keras
import pickle
import os
from sklearn.model_selection import train_test_split
from gensim.models.keyedvectors import KeyedVectors
import logging

logger = logging.getLogger('data_prepare')


def data_loader(mode="word", dataset="Ant"):
    """ load entire training (labeled) data """
    logger.info(f'Loading data of {dataset} dataset...')
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

    logger.info(f'Data amount: {len(Y)}')
    logger.info(f"Percentage of positive sample: {sum(Y)/len(Y)}")

    return X1.values, X2.values, Y.values


def train_test_data_loader(random_seed, mode="word", dataset="Ant", test_split=0.3):
    logger.info(f"Percentage of test data split: {test_split}")
    X1, X2, Y = data_loader(mode, dataset)
    X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(
        X1, X2, Y, test_size=test_split, random_state=random_seed)
    return X1_train, X2_train, Y_train, X1_test, X2_test, Y_test


def embedding_loader(embedding_folder="word2vec", mode="word", dataset="Ant"):
    X1, X2, _ = data_loader(mode, dataset)

    if dataset == "Ant":
        tokenizer_pickle_file = f'{embedding_folder}/{dataset}_{mode}_tokenizer.pickle'
        embed_pickle_file = f'{embedding_folder}/{dataset}_{mode}_embed_matrix.pickle'
    elif dataset == "Quora":
        tokenizer_pickle_file = f'{embedding_folder}/{dataset}_tokenizer.pickle'
        embed_pickle_file = f'{embedding_folder}/{dataset}_embed_matrix.pickle'

    # Load tokenizer
    logger.info('Loading tokenizer...')
    if os.path.isfile(tokenizer_pickle_file):
        with open(tokenizer_pickle_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(list(X1))
        tokenizer.fit_on_texts(list(X2))
        with open(tokenizer_pickle_file, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load embedding matrix
    logger.info('Loading embedding matrix...')
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


# For balance train


def balance_data_loader(random_seed, mode="word", dataset="Ant", test_split=0.3):
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = train_test_data_loader(
        random_seed, mode, dataset, test_split)

    trainHelper = BalanceDataHelper(X1_train, X2_train, Y_train, random_seed)
    testHelper = BalanceDataHelper(X1_test, X2_test, Y_test, random_seed)

    return trainHelper, testHelper


class BalanceDataHelper:
    def __init__(self, X1, X2, Y, random_seed):
        np.random.seed(random_seed)
        self._seperate_data(X1, X2, Y)

    def _seperate_data(self, X1, X2, Y):
        positive_index = Y[Y == 1]
        negative_index = Y[Y == 0]

        self.positive_count = len(positive_index)
        self.negative_count = len(negative_index)

        self.POS_SENTENCE_PAIR = list(
            zip(X1[positive_index], X2[positive_index]))
        self.NEG_SENTENCES = X1[negative_index] + X2[negative_index]

    def _get_negative_samples(self, positive_samples):
        """ get the same amount of the positive sample by replacing one of the sentence in positive samples """
        negative_samples = []
        for pos_sent1, pos_sent2 in positive_samples:
            # find a unique and non-repeated negative sample
            while (pos_sent1, pos_sent2) in self.POS_SENTENCE_PAIR or (pos_sent1, pos_sent2) in self.POS_SENTENCE_PAIR:
                neg_sent = np.random.choice(self.NEG_SENTENCES)
                if np.random.randint(0, 2):
                    # replace first sentence
                    pos_sent1 = neg_sent
                else:
                    # replace second sentence
                    pos_sent2 = neg_sent
            negative_samples.append((pos_sent1, pos_sent2))

        assert len(positive_samples) == len(negative_samples)
        return negative_samples

    def batch_iter(self, batch_size, shuffle=True, neg_label=0.0):
        """ generator of a batch of balance data, neg_label usually be 0 or -1 """
        positive_data_order = list(range(self.positive_count))
        if shuffle:
            np.random.shuffle(positive_data_order)

        assert batch_size % 2 == 0
        semi_batch_size = batch_size // 2

        for batch_step in range(self.positive_count // semi_batch_size):
            start_index = batch_step*semi_batch_size
            end_index = batch_step*semi_batch_size + semi_batch_size
            if end_index > self.positive_count:
                end_index = self.positive_count
            positive_data_index = positive_data_order[start_index:end_index]

            positive_sentence_pair = [self.POS_SENTENCE_PAIR[idx]
                                      for idx in positive_data_index]
            negative_sentence_pair = self._get_negative_samples(
                positive_sentence_pair)

            x_pair = positive_sentence_pair + negative_sentence_pair
            x1, x2 = zip(*x_pair)  # unzip zipped pair
            y = [[1.0]] * semi_batch_size + [[neg_label]] * semi_batch_size
            yield x1, x2, y


if __name__ == "__main__":
    # test
    trainHelper, _ = balance_data_loader(87)
    batch_iterator = trainHelper.batch_iter(4)
    print(next(batch_iterator))
    print(next(batch_iterator))
