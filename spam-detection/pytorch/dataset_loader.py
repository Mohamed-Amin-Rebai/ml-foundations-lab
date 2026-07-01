import pandas as pd
import torch

from collections import Counter
from torch.utils.data import Dataset

MAX_LENGTH = 30

def tokenize(text):
    return text.lower().split()

def build_vocab(texts, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def text_to_indices(text, vocab):
    tokens = tokenize(text)

    indices = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    indices = indices[:MAX_LENGTH]

    while len(indices) < MAX_LENGTH:
        indices.append(vocab["<PAD>"])

    return indices


class SpamDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.messages = dataframe["message"].values
        self.labels = dataframe["label"].values
        self.vocab = vocab

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        text = self.messages[idx]
        label = self.labels[idx]

        indices = text_to_indices(text, self.vocab)

        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )