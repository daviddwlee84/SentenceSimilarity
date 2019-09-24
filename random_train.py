
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from data_prepare import train_test_data_loader, tokenize_and_padding
from models.functions import contrastive_loss, dice_loss
from utils import save_model

import logging

logger = logging.getLogger('random_train')


def train(args, model, tokenizer, device, optimizer):
    model.train()

    X1, X2, Y, _, _, _ = train_test_data_loader(
        args.seed, mode=args.word_segment, dataset=args.dataset, test_split=args.test_split)

    stratified_folder = StratifiedKFold(
        n_splits=args.k_fold, random_state=args.seed, shuffle=True)

    for epoch, (train_index, test_index) in enumerate(stratified_folder.split(X1, Y)):
        X_fold_train1, X_fold_test1 = X1[train_index], X1[test_index]
        X_fold_train2, X_fold_test2 = X2[train_index], X2[test_index]
        Y_fold_train, Y_fold_test = Y[train_index], Y[test_index]

        X_tensor_train_1, X_tensor_train_2 = tokenize_and_padding(
            X_fold_train1, X_fold_train2, args.max_len, tokenizer)
        X_tensor_test_1, X_tensor_test_2 = tokenize_and_padding(
            X_fold_test1, X_fold_test2, args.max_len, tokenizer)

        train_tensor = torch_data.TensorDataset(X_tensor_train_1, X_tensor_train_2,
                                                torch.tensor(Y_fold_train, dtype=torch.float))
        train_dataset = torch_data.DataLoader(
            train_tensor, batch_size=args.batch_size)
        test_tensor = torch_data.TensorDataset(X_tensor_test_1, X_tensor_test_2,
                                               torch.tensor(Y_fold_test, dtype=torch.float))
        test_dataset = torch_data.DataLoader(
            test_tensor, batch_size=args.test_batch_size)

        for batch_idx, (input_1, input_2, target) in enumerate(train_dataset):
            input_1, input_2, target = input_1.to(
                device), input_2.to(device), target.to(device)
            optimizer.zero_grad()

            if args.model[:7] == "Siamese" and False:  # currently disable this
                output1, output2 = model(input_1, input_2)
                loss = contrastive_loss(output1, output2, target)
            else:
                output = model(input_1, input_2)
                if args.dataset == "Ant":  # use dice loss on unbalance dataset
                    loss = dice_loss(output, target.view_as(output))
                else:
                    loss = F.binary_cross_entropy(
                        output, target.view_as(output))

            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch + 1, batch_idx *
                    len(input_1), len(train_dataset.dataset),
                    100. * batch_idx / len(train_dataset), loss.item()))

        _test_on_dataloader(args, model, device, test_dataset)
        model.train()  # switch the model mode back to train
        if not args.not_save_model:
            save_model(args, model, epoch)


def _test_on_dataloader(args, model, device, test_loader, dataset="Valid", final=False):
    model.eval()  # Turn on evaluation mode which disables dropout
    test_loss = 0
    correct = 0
    with torch.no_grad():
        accumulated_pred = []  # for f1 score
        accumulated_target = []  # for f1 score
        for input_1, input_2, target in test_loader:
            input_1, input_2, target = input_1.to(
                device), input_2.to(device), target.to(device)

            if args.model[:7] == "Siamese" and False:  # currently disable this
                output1, output2 = model(input_1, input_2)
                # Oneshot Learning
                output = F.pairwise_distance(
                    output1, output2)  # euclidean distance
                test_loss += contrastive_loss(output1, output2, target).item()
            else:
                output = model(input_1, input_2)
                # sum up batch loss
                if args.dataset == "Ant":  # use dice loss on unbalance dataset
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

    test_loss /= len(test_loader.dataset)

    logger.info('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1: {:.2f}%'.format(
        dataset, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        100. * f1_score(accumulated_target, accumulated_pred, average='macro')))
    if final:
        logger.info('Confusion Matrix:\n{}'.format(str(confusion_matrix(
            accumulated_target, accumulated_pred))))
        logger.info('Classification Report:\n{}'.format(classification_report(
            accumulated_target, accumulated_pred)))


def test(args, model, tokenizer, device):
    _, _, _, X1, X2, Y = train_test_data_loader(
        args.seed, mode=args.word_segment, dataset=args.dataset, test_split=args.test_split)
    input_X1, input_X2 = tokenize_and_padding(X1, X2, args.max_len, tokenizer)
    input_tensor = torch_data.TensorDataset(input_X1, input_X2,
                                            torch.tensor(Y, dtype=torch.float))
    test_loader = torch_data.DataLoader(
        input_tensor, batch_size=args.test_batch_size)
    logger.info(f'Test on {len(test_loader.dataset)} amount of data')
    _test_on_dataloader(args, model, device, test_loader,
                        dataset="Test", final=True)
