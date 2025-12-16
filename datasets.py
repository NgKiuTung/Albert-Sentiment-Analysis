import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def get_word2index_dict(n_common=4000):
    word_count = Counter()
    data_train = pd.read_csv("datasets/data_train.csv", sep="\t")
    data_test = pd.read_csv("datasets/data_test.csv", sep="\t")
    data_total = pd.concat([data_train, data_test], axis=0)
    sentences = data_total["content"].values.tolist()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1

    most_common = word_count.most_common(n_common)
    word2index_dict = {word: index + 2 for index, (word, count) in enumerate(most_common)}
    word2index_dict["PAD"] = 0
    word2index_dict["UNK"] = 1
    return word2index_dict


def get_datasets():
    label_names = ['like', 'sadness', 'fear', 'anger', 'disgust', 'happiness', 'surprise']
    contents, labels = [], []
    with open("datasets/OCEMOTION.csv", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line_split = line.strip().split("\t")
            content, label = line_split[1], line_split[-1]
            contents.append(content)
            labels.append(str(label_names.index(label)))

    contents_train, contents_test, labels_train, labels_test = train_test_split(contents, labels, train_size=0.8,
                                                                                shuffle=True)
    contents_train += contents_test[:6000]
    labels_train += labels_test[:6000]
    with open("datasets/data_train.csv", "w", encoding="utf-8") as f:
        f.write("content\tlabels\n")
        for i in range(len(contents_train)):
            f.write(contents_train[i] + "\t" + str(labels_train[i]) + "\n")
    with open("datasets/data_val.csv", "w", encoding="utf-8") as f:
        f.write("content\tlabels\n")
        for i in range(len(contents_test)):
            f.write(contents_test[i] + "\t" + str(labels_test[i]) + "\n")


class DataGenerator(Dataset):

    def __init__(self, root, tokenizer, max_len):
        super(DataGenerator, self).__init__()
        self.root = root
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sentences, self.labels = self.get_datasets()

    def __getitem__(self, item):
        sentences = self.sentences[item]
        info_dict = self.tokenizer(sentences, max_length=self.max_len, truncation=True, padding="max_length")
        input_ids = info_dict["input_ids"]
        attention_mask = info_dict["attention_mask"]

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.from_numpy(
            np.array(self.labels[item])).long()

    def __len__(self):
        return len(self.labels)

    def get_datasets(self):
        df = pd.read_csv(self.root, sep="\t")
        sentences, labels = df["content"].values.tolist(), df["labels"].values.tolist()

        return sentences, labels


if __name__ == '__main__':
    get_datasets()
