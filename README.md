# SMS Spam Detection (SimpleRNN vs LSTM vs GRU)

Enter an SMS message and the model predicts **Spam** or **Not Spam**.
Bonus: three recurrent architectures (SimpleRNN, LSTM, GRU) are trained under identical conditions and compared.

- **Dataset:** SMS Spam Collection (5,574 messages: 4,827 ham, 747 spam)
- **Framework:** PyTorch
- **Difficulty:** Easy

## Project structure

```
sms-spam-detection/
├── data/SMSSpamCollection.tsv   # dataset (label <TAB> message)
├── model.py                     # tokenizer + SpamClassifier (RNN / LSTM / GRU)
├── train.py                     # trains all 3 models, compares them, saves plots
├── predict.py                   # classify your own messages
├── requirements.txt
└── outputs/                     # created by train.py (weights, vocab, results, plots)
```

## How to run

```bash
pip install -r requirements.txt
python train.py                  # ~1-2 min on CPU
python predict.py "WINNER!! You have won a 1000 prize. Call 09061701461 now"
python predict.py --model gru "Are we still meeting at 6?"
python predict.py                # interactive mode
```

If the dataset file is missing, download the SMS Spam Collection from the UCI repository and save it
as `data/SMSSpamCollection.tsv` (two tab-separated columns: `ham`/`spam`, then the message).

## How it works

1. **Preprocessing:** lowercase, replace URLs, phone numbers/short codes, numbers and currency symbols with placeholder tokens (these are strong spam signals), then split into words.
2. **Vocabulary:** built from the training split only (words seen at least twice), so there is no data leakage.
3. **Split:** stratified 70% train / 15% validation / 15% test.
4. **Model:** `Embedding(64) -> RNN | LSTM | GRU (64 hidden) -> Dropout -> Linear -> sigmoid`. Padded sequences are packed so the network ignores padding.
5. **Training:** Adam, 10 epochs, gradient clipping, and a class weight to handle the 87/13 ham/spam imbalance. The best epoch is chosen by validation F1, never by test data.
6. **Same seed and hyperparameters** for all three models, so the comparison is fair.

## Results (test set, 837 messages, 112 spam)

| Model     | Accuracy | Precision | Recall | F1 (spam) | Params  | Missed spam | False alarms |
|-----------|----------|-----------|--------|-----------|---------|-------------|--------------|
| SimpleRNN | 0.9845   | 0.9304    | 0.9554 | 0.9427    | 210,689 | 5           | 8            |
| LSTM      | 0.9809   | 0.9138    | 0.9464 | 0.9298    | 235,649 | 6           | 10           |
| GRU       | 0.9881   | 0.9474    | 0.9643 | 0.9558    | 227,329 | 4           | 6            |

Plots are in `outputs/`: `comparison.png`, `confusion_matrices.png`, `training_curves.png`.

### Reading the comparison honestly

- All three models land around 98-99% accuracy; the task is easy, and messages are short.
- GRU scored highest on the test set here, but the gap is only 2-4 messages out of 837. With a single run and 112 spam examples, that is within random noise, so don't read it as "GRU is better than LSTM".
- The model chosen automatically by `predict.py` (`--model best`) is the one with the best **validation** F1, which was LSTM in this run. Pass `--model gru` to use another one.
- SimpleRNN works well because SMS messages are short, so vanishing gradients barely matter. On long texts, LSTM/GRU would pull ahead.

## Ideas to extend

- Repeat with several random seeds and average, for a statistically meaningful comparison.
- Try a bidirectional layer, pretrained embeddings (GloVe), or a TF-IDF + Logistic Regression baseline.
- Wrap `predict.py` in a small Streamlit or Flask page.
