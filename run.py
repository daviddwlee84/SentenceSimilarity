import argparse
import os
import glob

# from sklearn.metrics import f1_score, recall_score, precision_score  # TODO

import torch
import torch.optim as optim

from rcnn import EnhancedRCNN, EnhancedRCNN_Transformer
from data_prepare import embedding_loader, tokenize_and_padding

MODEL_PATH = "model"

def load_latest_model(args, model_obj):
    if args.dataset == "Ant":
        list_of_models = glob.glob(
            f"{args.model_path}/{args.dataset}_{args.model}_epoch_*_{args.word_segment}.pkl")
    elif args.dataset == "Quora":
        list_of_models = glob.glob(
            f"{args.model_path}/{args.dataset}_{args.model}_epoch_*.pkl")
    latest_checkpoint = max(list_of_models, key=os.path.getctime)
    model_obj.load_state_dict(torch.load(latest_checkpoint))


def predict(args, model, tokenizer, device):
    model.eval()

    # zh_en = input('Chinese or English:')
    if args.dataset == "Ant":
        zh_en = 'c'
    elif args.dataset == "Quora":
        zh_en = 'e'

    raw_sentence_1 = input('Input test setnetnce 1: ')
    raw_sentence_2 = input('Input test setnetnce 2: ')

    if zh_en[0].lower() == 'c': # Chinese
        from preprocess import stopwordslist
        stopwords = stopwordslist()
        sentence_1 = []; sentence_2 = []
        if args.word_segment == "word":
            import jieba
            for word in ['花呗','借呗','支付宝','余额宝','饿了么','微粒贷','双十一','小蓝车','拼多多','外卖','美团','账单','到账','能不能','应还','会不会','找不到','另一个','微信','网商贷']:
                jieba.add_word(word)
            for word in ["开花", "开了花", "提花", "申花", "天花", "银花", "我花", "借花"]:
                jieba.del_word(word)
            
            for c in jieba.cut(raw_sentence_1):
                if c not in stopwords and c != ' ':
                    sentence_1.append(c)
            for c in jieba.cut(raw_sentence_2):
                if c not in stopwords and c != ' ':
                    sentence_2.append(c)
        elif args.word_segment == "char":
            for c in raw_sentence_1:
                if c not in stopwords and c != ' ':
                    sentence_1.append(c)
            for c in raw_sentence_2:
                if c not in stopwords and c != ' ':
                    sentence_2.append(c)
    elif zh_en[0].lower() == 'e': # English
        sentence_1 = raw_sentence_1.split()
        sentence_2 = raw_sentence_2.split()

    sentence_1 = [sentence_1]
    sentence_2 = [sentence_2]

    print('Processed sentences:', sentence_1, '\n', sentence_2)

    input_tensor_1, input_tensor_2 = tokenize_and_padding(
        sentence_1, sentence_2, args.max_len, tokenizer, debug=True)

    output = model(input_tensor_1.to(device), input_tensor_2.to(device))

    print('Predict similarity:', output)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='Ant', metavar='dataset',
                        choices=['Ant', 'Quora'],
                        help='[Ant] Finance or [Quora] Question Pairs (default: Ant)')
    parser.add_argument('--mode', type=str, default='train', metavar='mode',
                        choices=['train', 'test', 'predict'],
                        help='script mode [train/test/predict] (default: train)')
    parser.add_argument('--sampling', type=str, default='random', metavar='mode',
                        choices=['random', 'balance'],
                        help='sampling mode during training (default: random)')
    parser.add_argument('--model', type=str, default='ERCNN', metavar='model',
                        choices=['ERCNN', 'Transformer'],
                        help='model to use [ERCNN/Transformer] (default: ERCNN)')
    parser.add_argument('--word-segment', type=str, default='word', metavar='WS',
                        choices=['word', 'char'],
                        help='chinese word split mode [word/char] (default: word)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='N',
                        help='beta 1 for Adam optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='N',
                        help='beta 2 for Adam optimizer (default: 0.999)')
    parser.add_argument('--epsilon', type=float, default=1e-08, metavar='N',
                        help='epsilon for Adam optimizer (default: 1e-08)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=16, metavar='N',
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
    args.model_path = MODEL_PATH

    tokenizer, embeddings_matrix = embedding_loader(
        mode=args.word_segment, dataset=args.dataset)

    # model and optimizer
    if args.model == "ERCNN":
        model = EnhancedRCNN(embeddings_matrix, args.max_len).to(device)
    elif args.model == "Transformer":
        model = EnhancedRCNN_Transformer(
            embeddings_matrix, args.max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(
        args.beta1, args.beta2), eps=args.epsilon)

    # sampling mode
    if args.sampling == "random":
        from random_train import train, test

    if args.mode == "train":
        train(args, model, tokenizer, device, optimizer)
    elif args.mode == "test":
        load_latest_model(args, model)
        test(args, model, device)
    elif args.mode == "predict":
        load_latest_model(args, model)
        predict(args, model, tokenizer, device)


if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    main()
