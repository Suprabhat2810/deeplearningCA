"""
SMS Spam Detection - trains and compares three recurrent architectures:
SimpleRNN, LSTM and GRU on the SMS Spam Collection dataset.

Usage:
    python train.py                 # default settings
    python train.py --epochs 15     # train longer
Outputs (in ./outputs):
    results.json, comparison.png, confusion_matrices.png, training_curves.png,
    <model>.pt (best weights), vocab.json
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from model import SpamClassifier, encode, tokenize

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "SMSSpamCollection.tsv"
OUT = ROOT / "outputs"
ARCHS = ["rnn", "lstm", "gru"]
NAMES = {"rnn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vocab(texts, min_freq=2, max_size=8000):
    counts = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in counts.most_common(max_size):
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def to_loader(texts, labels, vocab, max_len, batch_size, shuffle):
    pairs = [encode(t, vocab, max_len) for t in texts]
    x = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    lengths = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)
    return DataLoader(TensorDataset(x, lengths, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    probs, ys = [], []
    for x, lengths, y in loader:
        probs.append(torch.sigmoid(model(x, lengths)))
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy().astype(int)


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_one(arch, vocab_size, loaders, pos_weight, args):
    train_dl, val_dl, test_dl = loaders
    model = SpamClassifier(vocab_size, arch, args.embed_dim, args.hidden_dim)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1, best_state, history = -1.0, None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, lengths, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # keeps SimpleRNN stable
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / len(train_dl.dataset)

        vp, vy = predict_loader(model, val_dl)
        val_f1 = f1_score(vy, (vp >= 0.5).astype(int), zero_division=0)
        val_acc = accuracy_score(vy, (vp >= 0.5).astype(int))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1, "val_acc": val_acc})
        print(f"  [{NAMES[arch]:9s}] epoch {epoch:2d}  loss {train_loss:.4f}  val_acc {val_acc:.4f}  val_f1 {val_f1:.4f}")
        if val_f1 > best_f1:  # keep the best epoch, chosen on validation data only
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tp, ty = predict_loader(model, test_dl)
    pred = (tp >= 0.5).astype(int)
    res = metrics(ty, pred)
    res["best_val_f1"] = best_f1
    res["confusion_matrix"] = confusion_matrix(ty, pred).tolist()
    res["params"] = sum(p.numel() for p in model.parameters())
    res["history"] = history
    torch.save({"state_dict": best_state, "arch": arch, "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim, "max_len": args.max_len}, OUT / f"{arch}.pt")
    return res


def make_plots(results):
    labels = [NAMES[a] for a in ARCHS]

    # 1) metric comparison bars
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = ["accuracy", "precision", "recall", "f1"]
    w = 0.2
    for i, k in enumerate(keys):
        vals = [results[a][k] for a in ARCHS]
        bars = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=k.capitalize())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    lo = min(results[a][k] for a in ARCHS for k in keys)
    ax.set_ylim(max(0, lo - 0.1), 1.05)
    ax.set_title("Test-set comparison of recurrent architectures")
    ax.legend(loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=150); plt.close(fig)

    # 2) confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, a in zip(axes, ARCHS):
        cm = np.array(results[a]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Spam", "Spam"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Spam", "Spam"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(NAMES[a])
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=150); plt.close(fig)

    # 3) training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for a in ARCHS:
        h = results[a]["history"]
        a1.plot([e["epoch"] for e in h], [e["train_loss"] for e in h], marker="o", label=NAMES[a])
        a2.plot([e["epoch"] for e in h], [e["val_f1"] for e in h], marker="o", label=NAMES[a])
    a1.set_title("Training loss"); a1.set_xlabel("Epoch"); a1.legend()
    a2.set_title("Validation F1 (spam class)"); a2.set_xlabel("Epoch"); a2.legend()
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--max-len", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8").dropna()
    df["y"] = (df["label"] == "spam").astype(int)
    print(f"Loaded {len(df)} messages  |  spam: {df.y.sum()}  ham: {(df.y == 0).sum()}")

    # stratified 70 / 15 / 15 split
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.y, random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.y, random_state=args.seed)
    print(f"Split -> train {len(tr)}, val {len(va)}, test {len(te)}")

    vocab = build_vocab(tr.text)  # built from training data only (no leakage)
    json.dump(vocab, open(OUT / "vocab.json", "w"))
    print(f"Vocabulary size: {len(vocab)}")

    loaders = (
        to_loader(tr.text, tr.y.tolist(), vocab, args.max_len, args.batch_size, True),
        to_loader(va.text, va.y.tolist(), vocab, args.max_len, 256, False),
        to_loader(te.text, te.y.tolist(), vocab, args.max_len, 256, False),
    )
    pos_weight = float((tr.y == 0).sum() / (tr.y == 1).sum())  # handle class imbalance

    results = {}
    for arch in ARCHS:
        print(f"\nTraining {NAMES[arch]} ...")
        set_seed(args.seed)  # same init seed for a fair comparison
        results[arch] = train_one(arch, len(vocab), loaders, pos_weight, args)

    results["best_model"] = max(ARCHS, key=lambda a: results[a]["best_val_f1"])
    json.dump(results, open(OUT / "results.json", "w"), indent=2)
    make_plots(results)

    print("\n=== Test-set results ===")
    print(f"{'Model':10s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Params':>9s}")
    for a in ARCHS:
        r = results[a]
        print(f"{NAMES[a]:10s} {r['accuracy']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f} {r['params']:9d}")
    print(f"\nBest model (by validation F1): {NAMES[results['best_model']]}")
    print(f"Saved weights, plots and results.json to {OUT}")


if __name__ == "__main__":
    main()
