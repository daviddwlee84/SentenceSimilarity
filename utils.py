import os
import time
import traceback
import logging
import torch

logger = logging.getLogger('utils')


def save_model(args, model, epoch):
    logger.info(f'Saving model on epoch {epoch + 1}')
    train_embed_txt = '(F)' if args.not_train_embed else '(T)'
    if args.dataset != "Quora":  # Chinese dataset
        torch.save(model.state_dict(),
                   f"{args.model_path}/{args.dataset}_{args.sampling}_{args.model}_epoch_{epoch + 1}_{args.chinese_embed}{train_embed_txt}_{args.word_segment}.pkl")
    else:  # English dataset
        torch.save(model.state_dict(),
                   f"{args.model_path}/{args.dataset}_{args.sampling}_{args.model}_epoch_{epoch + 1}_{train_embed_txt}.pkl")


def get_available_gpu(num_gpu=1, min_memory=500, try_times=3, delay=2, allow_gpus_nums=[0, 1, 2, 3], verbose=False):
    ''' get available GPU list with nvidia-smi commands
    :param num_gpu: number of GPU you want to use
    :param min_memory: minimum memory MiB
    :param try_times: how many attempt
    :param allow_gpus_nums: accessible gpu indices
    :param delay: start next attempt when fail
    :param verbose: verbose mode
    :return:best choices list
    '''
    for i in range(try_times):
        info_text = os.popen(
            'nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv').read()
        try:
            gpu_info = [(gpu_id, int(memory.replace('%', '').replace('MiB', '').split(',')[1].strip()))
                        for gpu_id, memory in enumerate(info_text.split('\n')[1:-1])]
        except:
            if verbose:
                print(traceback.format_exc())
            return "Not found gpu info ..."

        # Sorting the GPUs form the highest RAM
        gpu_info.sort(key=lambda info: info[1], reverse=True)
        available_gpu = [
            gpu_id for gpu_id, memory in gpu_info if gpu_id in allow_gpus_nums and memory >= min_memory]

        if len(available_gpu) == 0:
            print(
                f'No GPU available, attempt ({i}/{try_times})..., will retry in {delay} seconds ...')
            time.sleep(delay)
            delay *= 2
            continue

        selected_nums = available_gpu[:num_gpu]

        # TODO
        # if verbose:
        #     print('Available GPU List')
        #     first_line = [['id', 'utilization.gpu(%)', 'memory.free(MiB)']]
        #     matrix = first_line + [list(map(lambda x: str(x), available_gpu))]
        #     s = [[str(e) for e in row] for row in matrix]
        #     lens = [max(map(len, col)) for col in zip(*s)]
        #     fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        #     table = [fmt.format(*row) for row in s]
        #     print('\n'.join(table))
        #     print('Select GPU id #' + ",".join(map(lambda x: str(x), selected_nums)))

    return selected_nums
