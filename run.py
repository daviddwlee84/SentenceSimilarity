import argparse
import os
import pickle

import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score # TODO
from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as dataset
from torchvision import transforms

from rcnn import EnhancedRCNN

MODEL_PATH = "model"
EMBEDDING = "word2vec"

PAD_IDX = 0


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


def embedding_loader(X1, X2, embedding_folder=EMBEDDING, mode="word", dataset="Ant"):
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
            pickle.dump(embeddings_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer, torch.Tensor(embeddings_matrix)


def train(args, model, device, optimizer):
    model.train()

    X1, X2, Y = training_data_loader(mode=args.word_segment, dataset=args.dataset)
    tokenizer, embeddings_matrix = embedding_loader(
        X1, X2, mode=args.word_segment, dataset=args.dataset)

    stratified_folder = StratifiedKFold(
        n_splits=args.k_fold, random_state=args.seed, shuffle=True)

    for epoch, (train_index, test_index) in enumerate(stratified_folder.split(X1, Y)):
        X_fold_train1, X_fold_test1 = X1[train_index], X1[test_index]
        X_fold_train2, X_fold_test2 = X2[train_index], X2[test_index]
        Y_fold_train, Y_fold_test = Y[train_index], Y[test_index]

        list_tokenized_train1 = tokenizer.texts_to_sequences(X_fold_train1)
        list_tokenized_train2 = tokenizer.texts_to_sequences(X_fold_train2)
        list_tokenized_test1 = tokenizer.texts_to_sequences(X_fold_test1)
        list_tokenized_test2 = tokenizer.texts_to_sequences(X_fold_test2)

        input_train1 = keras.preprocessing.sequence.pad_sequences(
            list_tokenized_train1, maxlen=args.max_len)
        input_train2 = keras.preprocessing.sequence.pad_sequences(
            list_tokenized_train2, maxlen=args.max_len)
        input_test1 = keras.preprocessing.sequence.pad_sequences(
            list_tokenized_test1, maxlen=args.max_len)
        input_test2 = keras.preprocessing.sequence.pad_sequences(
            list_tokenized_test2, maxlen=args.max_len)

        train_tensor = dataset.TensorDataset(torch.tensor(input_train1, dtype=torch.long), torch.tensor(
            input_train2, dtype=torch.long), torch.tensor(Y_fold_train.values, dtype=torch.float))
        train_dataset = dataset.DataLoader(
            train_tensor, batch_size=args.batch_size)
        test_tensor = dataset.TensorDataset(torch.tensor(input_test1, dtype=torch.long), torch.tensor(
            input_test2, dtype=torch.long), torch.tensor(Y_fold_test.values, dtype=torch.float))
        test_dataset = dataset.DataLoader(
            test_tensor, batch_size=args.test_batch_size)

        for batch_idx, (input_1, input_2, target) in enumerate(train_dataset):
            input_1, input_2, target = input_1.to(
                device), input_2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input_1, input_2)
            loss = F.binary_cross_entropy(output, target.view_as(output))
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch + 1, batch_idx *
                    len(input_1), len(train_dataset.dataset),
                    100. * batch_idx / len(train_dataset), loss.item()))
            if batch_idx % args.test_interval == 0:
                test(args, model, device, test_dataset)
                model.train()  # switch the model mode back to train

        if not args.not_save_model:
            torch.save(model.state_dict(),
                       f"{MODEL_PATH}/{dataset}_epoch_{epoch + 1}_{mode}.pkl")


def test(args, model, device, test_loader):
    model.eval() # Turn on evaluation mode which disables dropout
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_1, input_2, target in test_loader:
            input_1, input_2, target = input_1.to(
                device), input_2.to(device), target.to(device)
            output = model(input_1, input_2)
            # sum up batch loss
            test_loss += F.binary_cross_entropy(output,
                                                target.view_as(output), reduction='sum').item()
            pred = output.round()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='Ant', metavar='D',
                        help='[Ant] Finance or [Quora] Question Pairs (default: Ant)')
    parser.add_argument('--mode', type=str, default='train', metavar='M',
                        help='train or test or both or predict mode (default: train)')
    parser.add_argument('--word-segment', type=str, default='word', metavar='WS',
                        help='chinese word split mode (char/word) (default: word)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                        help='beta 1 for Adam optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='beta 2 for Adam optimizer (default: 0.999)')
    parser.add_argument('--epsilon', type=float, default=1e-08, metavar='E',
                        help='epsilon for Adam optimizer (default: 1e-08)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=16, metavar='S',
                        help='random seed (default: 16)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=100, metavar='N',
                        help='how many batches to test during training')
    parser.add_argument('--not-save-model', action='store_true', default=False,
                        help='for not saving the current model')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device:", device)

    torch.manual_seed(args.seed)

    # additional custom parameter
    args.k_fold = 10
    args.max_len = 48
    args.max_feature = 20000

    X1, X2, _ = training_data_loader(mode=args.word_segment, dataset=args.dataset)
    _, embeddings_matrix = embedding_loader(X1, X2, mode=args.word_segment, dataset=args.dataset)

    model = EnhancedRCNN(embeddings_matrix, args.max_len, PAD_IDX).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(
        args.beta1, args.beta2), eps=args.epsilon)

    if args.mode == "train" or args.mode == "both":
        train(args, model, device, optimizer)

    if args.mode == "test" or args.mode == "both":
        pass
        # test(args, model, device, test_loader)


if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    main()
