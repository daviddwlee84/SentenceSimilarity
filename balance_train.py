
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from data_prepare import train_test_data_loader, tokenize_and_padding, BalanceDataHelper
from models.functions import contrastive_loss, dice_loss
from utils import save_model

import logging

logger = logging.getLogger('balance_train')


def train(args, model, tokenizer, device, optimizer, tbwriter):
    model.train()

    X1, X2, Y, _, _, _ = train_test_data_loader(
        args.seed, mode=args.word_segment, dataset=args.dataset, test_split=args.test_split)

    stratified_folder = StratifiedKFold(
        n_splits=args.k_fold, random_state=args.seed, shuffle=True)

    for epoch, (train_index, test_index) in enumerate(stratified_folder.split(X1, Y)):
        X_fold_train1, X_fold_test1 = X1[train_index], X1[test_index]
        X_fold_train2, X_fold_test2 = X2[train_index], X2[test_index]
        Y_fold_train, Y_fold_test = Y[train_index], Y[test_index]

        train_data_helper = BalanceDataHelper(
            X_fold_train1, X_fold_train2, Y_fold_train, args.seed, generate_mode=args.generate_train)
        train_dataset = train_data_helper.batch_iter(args.batch_size)
        test_data_helper = BalanceDataHelper(
            X_fold_test1, X_fold_test2, Y_fold_test, args.seed)

        for batch_idx, (X_fold_train1, X_fold_train2, target) in enumerate(train_dataset):
            target = torch.tensor(target, dtype=torch.float)
            X_tensor_train_1, X_tensor_train_2 = tokenize_and_padding(
                X_fold_train1, X_fold_train2, args.max_len, tokenizer)
            input_1, input_2, target = X_tensor_train_1.to(
                device), X_tensor_train_2.to(device), target.to(device)

            optimizer.zero_grad()
            if args.model[:7] == "Siamese" and False:  # currently disable this
                output1, output2 = model(input_1, input_2)
                loss = contrastive_loss(output1, output2, target)
            else:
                output = model(input_1, input_2)
                if args.dataset == "Ant" and False:  # currently disable this
                    # use dice loss on unbalance dataset
                    loss = dice_loss(output, target.view_as(output))
                else:
                    loss = F.binary_cross_entropy(
                        output, target.view_as(output))
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch + 1, batch_idx *
                    len(input_1), train_data_helper.dataset_size,
                    100. * batch_idx / len(train_data_helper), loss.item()))
                tbwriter.add_scalar('data/train/loss', loss.item(),
                                    epoch * len(train_dataset.dataset) + batch_idx * len(input_1))

        valid_loss, valid_acc, valid_f1 = _test_on_dataloader(
            args, model, tokenizer, device, test_data_helper)
        tbwriter.add_scalar('data/valid/loss', valid_loss, epoch + 1)
        tbwriter.add_scalar('data/valid/acc', valid_acc, epoch + 1)
        tbwriter.add_scalar('data/valid/f1', valid_f1, epoch + 1)
        model.train()  # switch the model mode back to train
        if not args.not_save_model:
            save_model(args, model, epoch)


def _test_on_dataloader(args, model, tokenizer, device, test_data_helper, dataset="Valid", final=False):
    model.eval()  # Turn on evaluation mode which disables dropout
    test_loss = 0
    correct = 0
    test_dataset = test_data_helper.batch_iter(args.batch_size)
    with torch.no_grad():
        accumulated_pred = []  # for f1 score
        accumulated_target = []  # for f1 score
        for X_fold_test1, X_fold_test2, target in test_dataset:
            target = torch.tensor(target, dtype=torch.float)
            X_tensor_test_1, X_tensor_test_2 = tokenize_and_padding(
                X_fold_test1, X_fold_test2, args.max_len, tokenizer)
            input_1, input_2, target = X_tensor_test_1.to(
                device), X_tensor_test_2.to(device), target.to(device)

            if args.model[:7] == "Siamese" and False:  # currently disable this
                output1, output2 = model(input_1, input_2)
                # Oneshot Learning
                output = F.pairwise_distance(
                    output1, output2)  # euclidean distance
                test_loss += contrastive_loss(output1, output2, target).item()
            else:
                output = model(input_1, input_2)
                # sum up batch loss
                if args.dataset == "Ant" and False:  # currently disable this
                    # use dice loss on unbalance dataset
                    test_loss += dice_loss(output,
                                           target.view_as(output)).item()
                else:
                    test_loss += F.binary_cross_entropy(
                        output, target.view_as(output), reduction='sum').item()

            pred = output.round()
            correct += pred.eq(target.view_as(pred)).sum().item()
            accumulated_pred.extend(  # for f1 score
                pred.view(len(pred)).tolist())
            accumulated_target.extend(  # for f1 score
                target.view(len(target)).tolist())

    test_loss /= test_data_helper.dataset_size

    logger.info('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1: {:.2f}%'.format(
        dataset, test_loss, correct, test_data_helper.dataset_size,
        100. * correct / test_data_helper.dataset_size,
        100. * f1_score(accumulated_target, accumulated_pred, average='macro')))
    if final:
        logger.info('Confusion Matrix:\n{}'.format(str(confusion_matrix(
            accumulated_target, accumulated_pred))))
        logger.info('Classification Report:\n{}'.format(classification_report(
            accumulated_target, accumulated_pred)))

    return test_loss, correct / test_data_helper.dataset_size, f1_score(accumulated_target, accumulated_pred, average='macro')


def test(args, model, tokenizer, device):
    _, _, _, X1, X2, Y = train_test_data_loader(
        args.seed, mode=args.word_segment, dataset=args.dataset, test_split=args.test_split)
    test_data_helper = BalanceDataHelper(
        X1, X2, Y, args.seed, generate_mode=args.generate_test)
    logger.info(f'Test on {test_data_helper.dataset_size} amount of data')
    _test_on_dataloader(args, model, tokenizer, device,
                        test_data_helper, dataset="Test", final=True)
