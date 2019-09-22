import argparse
import os
import glob
import logging
import time
from setproctitle import setproctitle

import torch
import torch.optim as optim

from models.rcnn import EnhancedRCNN
from models.rcnn_transformer import EnhancedRCNN_Transformer
from models.siamese_models import SiameseModel
from models.siamese_elements import SingleSiameseCNN
from data_prepare import embedding_loader, tokenize_and_padding

MODEL_PATH = "model"
LOG_PATH = "log"


def load_latest_model(args, model_obj):
    if args.dataset != "Quora":  # Chinese dataset
        possible_model_name = f"{args.model_path}/{args.dataset}_{args.sampling}_{args.model}_epoch_*_{args.chinese_embed}_{args.word_segment}.pkl"
    else:  # English dataset
        possible_model_name = f"{args.model_path}/{args.dataset}_{args.sampling}_{args.model}_epoch_*.pkl"

    list_of_models = glob.glob(possible_model_name)
    if len(list_of_models) == 0:
        logging.warning(
            f'No candidate model name "{possible_model_name}" found')

    latest_checkpoint = max(list_of_models, key=os.path.getctime)
    logging.info(f"Loading the latest model: {latest_checkpoint}")
    model_obj.load_state_dict(torch.load(latest_checkpoint))


def predict(args, model, tokenizer, device):
    model.eval()

    # zh_en = input('Chinese or English:')
    if args.dataset != "Quora":  # Chinese dataset
        zh_en = 'c'
    else:  # English dataset
        zh_en = 'e'

    raw_sentence_1 = input('Input test setnetnce 1: ')
    raw_sentence_2 = input('Input test setnetnce 2: ')

    if zh_en[0].lower() == 'c':  # Chinese
        from ant_preprocess import stopwordslist
        stopwords = stopwordslist()
        sentence_1 = []
        sentence_2 = []
        if args.word_segment == "word":
            import jieba
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
    elif zh_en[0].lower() == 'e':  # English
        sentence_1 = raw_sentence_1.split()
        sentence_2 = raw_sentence_2.split()

    sentence_1 = [sentence_1]
    sentence_2 = [sentence_2]

    print('Processed sentences:', sentence_1, '\n', sentence_2)

    input_tensor_1, input_tensor_2 = tokenize_and_padding(
        sentence_1, sentence_2, args.max_len, tokenizer, debug=True)

    output = model(input_tensor_1.to(device), input_tensor_2.to(device))

    print('Predict similarity:', output)


def get_model_parameters(model, trainable_only=False):
    if trainable_only:
        pytorch_total_params = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params


def print_settings(args):
    logging.info('Configurations:')
    logging.info(f'\tDataset\t\t: {args.dataset}')
    if args.dataset != "Quora":  # All the Chinese dataset
        logging.info(f'\t Word Segment\t: {args.word_segment}')
        logging.info(f'\t Embedding\t: {args.chinese_embed}')
    logging.info(f'\t Train Embedding: {args.train_embed}')
    logging.info(f'\tMode\t\t: {args.mode}')
    logging.info(f'\tSampling Mode\t: {args.sampling}')
    if args.sampling == "balance":
        logging.info(f'\t Generate train\t: {args.generate_train}')
        logging.info(f'\t Generate test\t: {args.generate_test}')
    logging.info(f'\tUsing Model\t: {args.model}')


def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Enhanced RCNN on Sentence Similarity')
    parser.add_argument('--dataset', type=str, default='Ant', metavar='dataset',
                        choices=['Ant', 'CCSK', 'PiPiDai', 'Quora'],
                        help='Chinese: Ant, CCSK; English: Quora (default: Ant)')
    parser.add_argument('--mode', type=str, default='both', metavar='mode',
                        choices=['train', 'test', 'both', 'predict'],
                        help='script mode [train/test/both/predict] (default: both)')
    parser.add_argument('--sampling', type=str, default='balance', metavar='mode',
                        choices=['random', 'balance'],
                        help='sampling mode during training (default: balance)')
    parser.add_argument('--generate-train', action='store_true', default=False,
                        help='use generated negative samples when training (used in balance sampling)')
    parser.add_argument('--generate-test', action='store_true', default=False,
                        help='use generated negative samples when testing (used in balance sampling)')
    parser.add_argument('--model', type=str, default='ERCNN', metavar='model',
                        choices=['ERCNN', 'Transformer', 'SiameseCNN'],
                        help='model to use [ERCNN/Transformer] (default: ERCNN)')
    parser.add_argument('--word-segment', type=str, default='char', metavar='WS',
                        choices=['word', 'char'],
                        help='chinese word split mode [word/char] (default: char)')
    parser.add_argument('--chinese-embed', type=str, default='cw2vec', metavar='embed',
                        choices=['cw2vec', 'glyce'],
                        help='chinese embedding (default: cw2vec)')
    parser.add_argument('--train-embed', action='store_true', default=False,
                        help='whether to freeze the embedding parameters')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--k-fold', type=int, default=10, metavar='N',
                        help='k-fold cross validation i.e. number of epochs to train (default: 10)')
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
    parser.add_argument('--test-split', type=float, default=0.3, metavar='N',
                        help='test data split (default: 0.3)')
    parser.add_argument('--logdir', type=str, default=LOG_PATH, metavar='path',
                        help='set log directory (default: ./log)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=100, metavar='N',
                        help='how many batches to test during training')
    parser.add_argument('--not-save-model', action='store_true', default=False,
                        help='for not saving the current model')

    args = parser.parse_args()

    # Logging
    ctime = time.localtime()
    os.makedirs(args.logdir, exist_ok=True)
    if args.dataset != "Quora":  # Chinese dataset
        logfilename = '{}_{}_{}_{}_{}_{}_{}-{}_{}-{}.log'.format(
            args.mode, args.dataset, args.sampling, args.model, args.chinese_embed, args.word_segment,
            ctime.tm_mon, ctime.tm_mday, ctime.tm_hour, ctime.tm_min
        )
    else:  # English dataset
        logfilename = '{}_{}_{}_{}_{}-{}_{}-{}.log'.format(
            args.mode, args.dataset, args.sampling, args.model,
            ctime.tm_mon, ctime.tm_mday, ctime.tm_hour, ctime.tm_min
        )
    setproctitle('WWW--' + logfilename)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-13s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=f'{args.logdir}/{logfilename}',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-13s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # PyTorch device configure (cuda/GPU or CPU)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Use device: {device}")

    torch.manual_seed(args.seed)

    # additional custom parameter
    args.max_len = 48
    args.max_feature = 20000
    args.model_path = MODEL_PATH

    print_settings(args)

    tokenizer, embeddings_matrix = embedding_loader(
        mode=args.word_segment, embed=args.chinese_embed, dataset=args.dataset)

    # model and optimizer
    logging.info("Building model...")
    if args.model == "ERCNN":
        model = EnhancedRCNN(
            embeddings_matrix, args.max_len, freeze_embed=not args.train_embed).to(device)
    elif args.model == "Transformer":
        model = EnhancedRCNN_Transformer(
            embeddings_matrix, args.max_len, freeze_embed=not args.train_embed).to(device)
    elif args.model[:7] == "Siamese":
        if args.model[7:] == "CNN":
            single_model = SingleSiameseCNN(
                embeddings_matrix, args.max_len, device, freeze_embed=not args.train_embed).to(device)
        model = SiameseModel(single_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(
        args.beta1, args.beta2), eps=args.epsilon)
    logging.info(f'Model Complexity (Parameters):')
    logging.info(f'\tAll\t\t: {get_model_parameters(model)}')
    logging.info(f'\tTrainable\t: {get_model_parameters(model, True)}')

    # sampling mode
    if args.sampling == "random":
        from random_train import train, test
    elif args.sampling == "balance":
        from balance_train import train
        from random_train import test  # use unbalancd data (raw data) to test

    if args.mode == "train" or args.mode == "both":
        logging.info(f"Training using {args.sampling} sampling mode...")
        train(args, model, tokenizer, device, optimizer)
    if args.mode == "test" or args.mode == "both":
        logging.info(f"Testing on {args.test_split*10}% data...")
        if args.mode != "both":
            load_latest_model(args, model)
        test(args, model, tokenizer, device)
    if args.mode == "predict":
        logging.info("Predicting manually...")
        load_latest_model(args, model)
        predict(args, model, tokenizer, device)


if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    main()
