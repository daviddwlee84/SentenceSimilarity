import os
import pandas as pd
from tqdm import tqdm
import threading


def _question_replacement(train_pd, question_pd, mode):
    print('Dealing with', mode)

    for i in tqdm(range(len(train_pd))):
        q1 = train_pd.iloc[i, 1]
        train_pd.iloc[i, 1] = question_pd[question_pd.qid ==
                                          q1][mode].values[0]
        q2 = train_pd.iloc[i, 2]
        train_pd.iloc[i, 2] = question_pd[question_pd.qid ==
                                          q2][mode].values[0]

    # train_pd.q1 = train_pd.apply(
    #     lambda x: question_pd[question_pd.qid == x.q1][mode].values, axis=1)
    # train_pd.q2 = train_pd.apply(
    #     lambda x: question_pd[question_pd.qid == x.q2][mode].values, axis=1)

    train_pd.to_csv(f'data/PiPiDai_{mode}_train.csv')


def process(train_file='raw_data/PiPiDai/train.csv', question_file='raw_data/PiPiDai/question.csv'):
    train_pd = pd.read_csv(train_file)
    question_pd = pd.read_csv(question_file)
    char_thread = threading.Thread(target=_question_replacement, args=(
        train_pd.copy(), question_pd, 'chars'))
    word_thread = threading.Thread(target=_question_replacement, args=(
        train_pd.copy(), question_pd, 'words'))
    char_thread.start()
    word_thread.start()
    # _question_replacement(train_pd.copy(), question_pd, 'chars')
    # _question_replacement(train_pd.copy(), question_pd, 'words')


if __name__ == "__main__":
    process()
