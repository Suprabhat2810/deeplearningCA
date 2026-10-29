"""Shared model + text preprocessing used by train.py and predict.py."""
import re

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD, UNK = 0, 1


def tokenize(text):
    """Lowercase; keep words, and map numbers / URLs to placeholders (strong spam signals)."""
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " <url> ", text)
    text = re.sub(r"\d{5,}", " <longnum> ", text)  # phone numbers, short codes
    text = re.sub(r"\d+", " <num> ", text)
    text = re.sub(r"[£$€]", " <money> ", text)
    return re.findall(r"<\w+>|[a-z']+|[!?]", text)


def encode(text, vocab, max_len):
    ids = [vocab.get(t, UNK) for t in tokenize(text)][:max_len]
    length = max(len(ids), 1)
    ids = ids or [UNK]
    return ids + [PAD] * (max_len - len(ids)), length


class SpamClassifier(nn.Module):
    """Embedding -> {RNN | LSTM | GRU} -> Dropout -> Linear (one logit)."""

    def __init__(self, vocab_size, arch="lstm", embed_dim=64, hidden_dim=64):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[arch]
        self.arch = arch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        # pack so the recurrent layer stops at the real end of each message (ignores padding)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        h = hidden[0] if self.arch == "lstm" else hidden  # LSTM returns (h, c)
        return self.fc(self.dropout(h[-1])).squeeze(1)
