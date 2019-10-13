import os
import codecs
import jieba

from ant_preprocess import stopwordslist


def _char_process(lines, stopwords):
    sentences = []
    for line in lines:
        sent1, sent2, label = line.split('\t')
        doc1 = [c for c in sent1 if c not in stopwords and c != ' ']
        doc2 = [c for c in sent2 if c not in stopwords and c != ' ']
        sentences.append([' '.join(doc1), ' '.join(doc2), str(int(label))])
    return sentences


def _word_process(lines, stopwords):
    sentences = []
    for line in lines:
        sent1, sent2, label = line.split('\t')
        doc1 = [c for c in jieba.cut(sent1) if c not in stopwords and c != ' ']
        doc2 = [c for c in jieba.cut(sent2) if c not in stopwords and c != ' ']
        sentences.append([' '.join(doc1), ' '.join(doc2), str(int(label))])
    return sentences


def _save_csv(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(','.join(line) + '\n')


def process(datapath="raw_data/task3_train.txt"):
    stopwords = stopwordslist()

    with codecs.open(datapath, 'r', 'utf-8') as f:
        lines = f.readlines()

    # word segment with char
    sentences = _char_process(lines, stopwords)
    _save_csv(sentences, 'data/ccks_char.csv')

    # word segment with word
    sentences = _word_process(lines, stopwords)
    _save_csv(sentences, 'data/ccks_word.csv')


if __name__ == "__main__":
    process()
