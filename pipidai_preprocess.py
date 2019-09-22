import os
import pandas as pd
import threading


def _question_replacement(train_pd, question_pd, mode):
    print('Dealing with', mode)

    train_pd = pd.merge(train_pd, question_pd, left_on=[
        'q1'], right_on=['qid'], how='left')
    train_pd = pd.merge(train_pd, question_pd, left_on=[
        'q2'], right_on=['qid'], how='left')

    train_pd = train_pd[['label', f'{mode}_x', f'{mode}_y']]
    train_pd.columns = ['label', 'q1', 'q2']

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


if __name__ == "__main__":
    process()
