# coding=utf-8
import sys
import codecs
import jieba
import os

RAW_DATA_PATH = "raw_data"
DATA_PATH = "data"


def stopwordslist(filepath='data/stopwords.txt'):
    stopwords = [line.strip() for line in codecs.open(filepath, 'r', 'utf-8').readlines()]
    return stopwords

def main():
    mode1 = sys.argv[1] # char or word
    mode2 = sys.argv[2] # train or test
    if mode2 == "train":
        # if train then use competition training data
        input_dir = "data/competition_train.csv"
    else: # mode2 == "test"
        # if test  then input test data manually
        input_dir = sys.argv[3]
    stopwords = stopwordslist('data/stopwords.txt')
    if mode1 == "char":
        with codecs.open(input_dir, 'r', 'utf-8') as f:
            lines = f.readlines()
            sentence_list = []
            for line in lines:
                # each line the two sentences will be seperated by TAB
                lineArr = line.strip().split("\t")
                doc1 = []
                for c in lineArr[1]:
                    if c not in stopwords and c != ' ':
                        doc1.append(c)
                doc2 = []
                for c in lineArr[2]:
                    if c not in stopwords and c != ' ':
                        doc2.append(c)
                if mode2 == "train":
                    sentence_list.append([" ".join(doc1), " ".join(doc2), lineArr[3]])
                else: # mode2 == "test"
                    sentence_list.append([" ".join(doc1), " ".join(doc2)])
        if mode2 == "train":
            with open("data/sentence_char_train.csv", "w") as f:
                for i in sentence_list:
                    line = ",".join(i)
                    f.write(line + "\n")
        else: # mode2 == "test"
            with open("sentence_char_test.csv", "w") as f:
                for i in sentence_list:
                    line = ",".join(i)
                    f.write(line + "\n")
    else: # mode1 == "word"
        with codecs.open(input_dir, 'r', 'utf-8') as f:
            lines = f.readlines()
            sentence_list = []
            for line in lines:
                lineArr = line.strip().split("\t")
                doc1 = []
                for c in list(jieba.cut(lineArr[1])):
                    if c not in stopwords and c != ' ':
                        doc1.append(c)
                doc2 = []
                for c in list(jieba.cut(lineArr[2])):
                    if c not in stopwords and c != ' ':
                        doc2.append(c)
                if mode2 == "train":
                    sentence_list.append([" ".join(doc1), " ".join(doc2), lineArr[3]])
                else: # mode2 == "test"
                    sentence_list.append([" ".join(doc1), " ".join(doc2)])

        if mode2 == "train":
            with open("data/sentence_word_train.csv", "w") as f:
                for i in sentence_list:
                    line = ",".join(i)
                    f.write(line + "\n")
        else: # mode2 == "test"
            with open("sentence_word_test.csv", "w") as f:
                for i in sentence_list:
                    line = ",".join(i)
                    f.write(line + "\n")


if __name__ == "__main__":
    for word in ['花呗','借呗','支付宝','余额宝','饿了么','微粒贷','双十一','小蓝车','拼多多','外卖','美团','账单','到账','能不能','应还','会不会','找不到','另一个','微信','网商贷']:
        jieba.add_word(word)
    for word in ["开花", "开了花", "提花", "申花", "天花", "银花", "我花", "借花"]:
        jieba.del_word(word)

    os.makedirs(DATA_PATH, exist_ok=True)
