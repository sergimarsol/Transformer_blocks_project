import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, FeedForwardClassifier, TransformerDecoder
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════
#  Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════
batch_size = 16       # Number of independent sequences processed in parallel
block_size = 32       # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64           # Embedding dimension
n_head = 2            # Number of attention heads
n_layer = 4           # Number of transformer layers

eval_interval = 100   # How often to log during decoder pretraining
max_iters = 500       # Number of decoder pretraining iterations
eval_iters = 200      # Batches used to estimate test perplexity

# Classifier head
n_input = 64          # Must match n_embd
n_hidden = 100        # Hidden size for the classifier / decoder FFN
n_output = 3          # Number of speaker classes
epochs_CLS = 15       # Epochs for classifier training


# ═══════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def load_texts(directory):
    """Load all non-test text files from *directory* (used for tokenizer)."""
    texts = []
    for filename in os.listdir(directory):
        if "test" in filename:
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts


def collate_batch(batch):
    """Collate a batch of (data, label) pairs into padded tensors."""
    data, labels = zip(*batch)
    padded = pad_sequence(data, batch_first=True, padding_value=0)
    padded = padded[:, :block_size]
    padded = torch.nn.functional.pad(
        padded, (0, max(0, block_size - padded.shape[1])), "constant", 0
    )
    return padded, torch.stack(labels)


def compute_classifier_accuracy(encoder, classifier, data_loader):
    """Return accuracy (%) of encoder+classifier on *data_loader*."""
    encoder.eval(); classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            emb, _ = encoder(X)
            preds = classifier(emb, (X == 0)).argmax(dim=1)
            correct += (preds == Y).sum().item()
            total += Y.size(0)
    encoder.train(); classifier.train()
    return 100 * correct / total


def compute_perplexity(decoder, data_loader, eval_iters=100):
    """Return perplexity of *decoder* on *data_loader*."""
    decoder.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        losses.append(decoder(X, Y).item())
        if len(losses) >= eval_iters:
            break
    decoder.train()
    return torch.exp(torch.tensor(losses).mean()).item()


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading (shared across parts)
# ═══════════════════════════════════════════════════════════════════════════

def setup_data():
    """Build tokenizer and return all data loaders needed."""
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Classification loaders
    train_CLS = DataLoader(
        SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv"),
        batch_size=batch_size, collate_fn=collate_batch, shuffle=True,
    )
    test_CLS = DataLoader(
        SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv"),
        batch_size=batch_size, collate_fn=collate_batch, shuffle=False,
    )

    # Language-modelling loaders
    with open("speechesdataset/train_LM.txt", 'r', encoding='utf-8') as f:
        train_LM_text = f.read()
    train_LM = DataLoader(
        LanguageModelingDataset(tokenizer, train_LM_text, block_size),
        batch_size=batch_size, shuffle=True,
    )

    test_LM = {}
    for politician in ['obama', 'wbush', 'hbush']:
        with open(f"speechesdataset/test_LM_{politician}.txt", 'r', encoding='utf-8') as f:
            txt = f.read()
        test_LM[politician] = DataLoader(
            LanguageModelingDataset(tokenizer, txt, block_size),
            batch_size=batch_size, shuffle=False,
        )

    return tokenizer, train_CLS, test_CLS, train_LM, test_LM


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 — Encoder + Classifier
# ═══════════════════════════════════════════════════════════════════════════

def run_part1(tokenizer, train_CLS, test_CLS):
    """Train the Transformer Encoder jointly with a feedforward classifier
    and report per-epoch train/test accuracy."""
    print("\n" + "=" * 72)
    print("  PART 1: Encoder + Classifier")
    print("=" * 72)

    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size, d_model=n_embd, num_heads=n_head,
        num_layers=n_layer, max_seq_len=block_size,
    ).to(device)

    classifier = FeedForwardClassifier(
        d_model=n_input, hidden_dim=n_hidden, num_classes=n_output,
    ).to(device)

    enc_p = sum(p.numel() for p in encoder.parameters())
    cls_p = sum(p.numel() for p in classifier.parameters())
    print(f"Encoder params: {enc_p}  |  Classifier params: {cls_p}  |  Total: {enc_p + cls_p}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate,
    )

    print("\n--- Training ---")
    for epoch in range(epochs_CLS):
        encoder.train(); classifier.train()
        total_loss = correct = total = 0
        for xb, yb in train_CLS:
            xb, yb = xb.to(device), yb.to(device)
            emb, _ = encoder(xb)
            logits = classifier(emb, (xb == 0))
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        tr_acc = 100 * correct / total
        te_acc = compute_classifier_accuracy(encoder, classifier, test_CLS)
        print(f"Epoch {epoch+1:2d}/{epochs_CLS}  |  "
              f"Loss: {total_loss/total:.4f}  |  "
              f"Train Acc: {tr_acc:.2f}%  |  Test Acc: {te_acc:.2f}%")

    # Sanity check
    print("\n--- Sanity Check: Attention Maps ---")
    utils = Utilities(tokenizer, encoder)
    for sent, label in [
        ("This afternoon, I spoke to former President George W. Bush.", "Obama"),
        ("But new threats also require new thinking.", "W. Bush"),
    ]:
        print(f'\n"{sent}"  ({label})')
        utils.sanity_check(sent, block_size, prefix="encoder_")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — Decoder pretraining (Language Modeling)
# ═══════════════════════════════════════════════════════════════════════════

def train_and_eval_decoder(tokenizer, train_LM, test_LM,
                           window_size=None, use_alibi=False,
                           block_sparse_size=None, label="full"):
    """Train a decoder LM and return {politician: perplexity} dict."""
    torch.manual_seed(seed)

    decoder = TransformerDecoder(
        vocab_size=tokenizer.vocab_size, d_model=n_embd, num_heads=n_head,
        num_layers=n_layer, max_seq_len=block_size, d_ff=n_hidden,
        window_size=window_size, use_alibi=use_alibi,
        block_sparse_size=block_sparse_size,
    ).to(device)

    dec_p = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder params: {dec_p}  (attention: {label})")

    opt = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    decoder.train()
    for i, (xb, yb) in enumerate(train_LM):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        loss = decoder(xb, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % eval_interval == 0 or i == max_iters - 1:
            ppl = torch.exp(torch.tensor(loss.item())).item()
            print(f"  Iter {i+1:4d}/{max_iters}  |  Loss: {loss.item():.4f}  |  Train PPL: {ppl:.2f}")

    # Evaluate on test sets
    results = {}
    for politician, loader in test_LM.items():
        results[politician] = compute_perplexity(decoder, loader, eval_iters)

    return results, decoder


def run_part2(tokenizer, train_LM, test_LM):
    """Pre-train the decoder LM with full causal attention and report
    test perplexity for each politician."""
    print("\n" + "=" * 72)
    print("  PART 2: Decoder Pretraining (Language Modeling)")
    print("=" * 72)

    results, decoder = train_and_eval_decoder(
        tokenizer, train_LM, test_LM, label="full"
    )

    print("\n--- Test Perplexity (full attention) ---")
    for pol, ppl in results.items():
        print(f"  {pol:>6s}: {ppl:.2f}")

    # Sanity check
    print("\n--- Sanity Check: Attention Maps ---")
    utils_dec = Utilities(tokenizer, decoder)
    for sent, label in [
        ("This afternoon, I spoke to former President George W. Bush.", "Obama"),
        ("But new threats also require new thinking.", "W. Bush"),
    ]:
        print(f'\n"{sent}"  ({label})')
        utils_dec.sanity_check(sent, block_size, prefix="decoder_", is_causal=True)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — Architectural explorations
# ═══════════════════════════════════════════════════════════════════════════

def run_part3(tokenizer, train_LM, test_LM, methods):
    """Run architectural exploration variants and compare against the
    full-attention baseline.  *methods* is a list that may contain
    'window', 'alibi', and/or 'block_sparse'."""
    print("\n" + "=" * 72)
    print("  PART 3: Architectural Explorations")
    print("=" * 72)

    all_results = {}

    # ── Baseline: full causal attention ───────────────────────────────────
    print("\n>>> Baseline: Full causal attention")
    baseline, _ = train_and_eval_decoder(
        tokenizer, train_LM, test_LM, label="full"
    )
    all_results["Full attention"] = baseline

    # ── Window attention (best: W=5) ──────────────────────────────────────
    if 'window' in methods:
        for w in [3, 5]:
            tag = f"Window (W={w})"
            print(f"\n>>> {tag}")
            res, _ = train_and_eval_decoder(
                tokenizer, train_LM, test_LM,
                window_size=w, label=f"window={w}"
            )
            all_results[tag] = res

    # ── ALiBi ─────────────────────────────────────────────────────────────
    if 'alibi' in methods:
        print("\n>>> ALiBi")
        res, _ = train_and_eval_decoder(
            tokenizer, train_LM, test_LM,
            use_alibi=True, label="alibi"
        )
        all_results["ALiBi"] = res

    # ── Blockwise sparse attention (best: B=8) ───────────────────────────
    if 'block_sparse' in methods:
        for b in [4, 8]:
            tag = f"Block Sparse (B={b})"
            print(f"\n>>> {tag}")
            res, _ = train_and_eval_decoder(
                tokenizer, train_LM, test_LM,
                block_sparse_size=b, label=f"block_sparse={b}"
            )
            all_results[tag] = res

    # ── Summary table ─────────────────────────────────────────────────────
    politicians = ['obama', 'wbush', 'hbush']
    header = f"{'Configuration':<26s} | {'Obama':>10s} | {'W.Bush':>10s} | {'H.Bush':>10s}"
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print("  PART 3 — Comparison of Test Perplexities")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for name, res in all_results.items():
        vals = "  |  ".join(f"{res[p]:>8.2f}" for p in politicians)
        print(f"{name:<26s} |  {vals}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CSE256 PA2 — Transformer Encoder/Decoder experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'part', choices=['part1', 'part2', 'part3'],
        help="Which part of the assignment to run:\n"
             "  part1  — Encoder + Classifier (train & evaluate)\n"
             "  part2  — Decoder LM pretraining (full attention)\n"
             "  part3  — Architectural explorations (window, alibi, block_sparse)",
    )
    parser.add_argument(
        '--methods', nargs='+',
        choices=['window', 'alibi', 'block_sparse'],
        default=['window', 'alibi', 'block_sparse'],
        help="(Part 3 only) Which exploration methods to run.\n"
             "Default: all three.  Example: --methods window alibi",
    )
    args = parser.parse_args()

    torch.manual_seed(seed)

    print("Loading data and creating tokenizer ...")
    tokenizer, train_CLS, test_CLS, train_LM, test_LM = setup_data()

    if args.part == 'part1':
        run_part1(tokenizer, train_CLS, test_CLS)

    elif args.part == 'part2':
        run_part2(tokenizer, train_LM, test_LM)

    elif args.part == 'part3':
        run_part3(tokenizer, train_LM, test_LM, args.methods)


if __name__ == "__main__":
    main()
