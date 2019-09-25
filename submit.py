import os
import logging
import codecs
import torch
import torch.utils.data as torch_data

from data_prepare import data_loader, tokenize_and_padding

logger = logging.getLogger('submit')


def predict_all(args, model, tokenizer, device):
    X1, X2 = data_loader(mode=args.word_segment,
                         dataset=args.dataset, is_submit=True)
    input_X1, input_X2 = tokenize_and_padding(X1, X2, args.max_len, tokenizer)
    input_tensor = torch_data.TensorDataset(input_X1, input_X2)
    test_loader = torch_data.DataLoader(
        input_tensor, batch_size=args.test_batch_size)
    logger.info(f'Predict on {len(test_loader.dataset)} amount of data')

    model.eval()  # Turn on evaluation mode which disables dropout
    with torch.no_grad():
        accumulated_pred = []
        for input_1, input_2 in test_loader:
            input_1, input_2 = input_1.to(
                device), input_2.to(device)

            output = model(input_1, input_2)

            pred = output.round()
            accumulated_pred.extend(
                pred.view(len(pred)).tolist())

    return accumulated_pred


def ant_submit(args, model, tokenizer, device):
    prediction = predict_all(args, model, tokenizer, device)
    with codecs.open(args.submit_path, 'w', 'utf-8') as f:
        for line_num, ans in enumerate(prediction):
            f.write(f'{line_num}\t{int(ans)}\n')
