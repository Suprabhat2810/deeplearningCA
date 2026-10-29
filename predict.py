"""
Classify SMS messages as Spam / Not Spam with a trained model.

Usage:
    python predict.py "WINNER!! You have won a £1000 prize. Call 09061701461 now"
    python predict.py --model gru "Are we still meeting at 6?"
    python predict.py                     # interactive mode (type messages, Ctrl+C to quit)
"""
import argparse
import json
from pathlib import Path

import torch

from model import SpamClassifier, encode

OUT = Path(__file__).parent / "outputs"


def load(arch):
    ckpt = torch.load(OUT / f"{arch}.pt", map_location="cpu")
    vocab = json.load(open(OUT / "vocab.json"))
    model = SpamClassifier(len(vocab), ckpt["arch"], ckpt["embed_dim"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, ckpt["max_len"]


@torch.no_grad()
def classify(text, model, vocab, max_len):
    ids, length = encode(text, vocab, max_len)
    prob = torch.sigmoid(model(torch.tensor([ids]), torch.tensor([length]))).item()
    return ("Spam" if prob >= 0.5 else "Not Spam"), prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("message", nargs="*", help="SMS text (omit for interactive mode)")
    p.add_argument("--model", choices=["rnn", "lstm", "gru", "best"], default="best",
                   help="which trained model to use (default: best on validation F1)")
    args = p.parse_args()

    arch = args.model
    if arch == "best":
        arch = json.load(open(OUT / "results.json"))["best_model"]
    model, vocab, max_len = load(arch)
    print(f"[using {arch.upper()} model]")

    def show(text):
        label, prob = classify(text, model, vocab, max_len)
        print(f"  -> {label}   (spam probability: {prob:.1%})")

    if args.message:
        show(" ".join(args.message))
    else:
        try:
            while True:
                text = input("\nEnter SMS: ").strip()
                if text:
                    show(text)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")


if __name__ == "__main__":
    main()
