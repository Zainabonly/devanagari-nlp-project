# train_seq2seq_no_sandhi.py
import os
import math
import random
import csv
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sacrebleu
import numpy as np

from transformers import PreTrainedTokenizerFast

# ========== CONFIG ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 50

TRAIN_CSV = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_train.csv"
VALID_CSV = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_val.csv"
TEST_CSV  = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_test.csv"

UNIGRAM_TOKENIZER_PATH = "/DATA/rohit/NLP_2025/unigram"
OUTPUT_DIR = "./seq2seq_out_no_sandhi"

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 3
DROPOUT = 0.1

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-4

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ========== TOKENIZER ==========
print("[INFO] Loading unigram tokenizer from:", UNIGRAM_TOKENIZER_PATH)
tokenizer = PreTrainedTokenizerFast.from_pretrained(UNIGRAM_TOKENIZER_PATH)

specials = {}
if tokenizer.pad_token is None:
    specials["pad_token"] = "<pad>"
if tokenizer.bos_token is None:
    specials["bos_token"] = "<s>"
if tokenizer.eos_token is None:
    specials["eos_token"] = "</s>"

if specials:
    tokenizer.add_special_tokens(specials)

pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

print("Tokenizer Special IDs -> pad:", pad_id, "bos:", bos_id, "eos:", eos_id)

# ========== LOAD CSV FILES ==========
def read_csv_pairs(path):
    src, tgt = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} seems empty or malformed.")
        if "mix" not in reader.fieldnames or "english" not in reader.fieldnames:
            raise ValueError(f"CSV {path} must have columns: mix,english")
        for row in reader:
            s = row["mix"].strip()
            t = row["english"].strip()
            if s and t:
                src.append(s)
                tgt.append(t)
    return src, tgt

print("[INFO] Reading CSV files...")
train_src, train_tgt = read_csv_pairs(TRAIN_CSV)
valid_src, valid_tgt = read_csv_pairs(VALID_CSV)
test_src, test_tgt   = read_csv_pairs(TEST_CSV)

print(f"Loaded CSV sizes: Train={len(train_src)}, Valid={len(valid_src)}, Test={len(test_src)}")

# ========== DATASET ==========
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=MAX_LEN):
        self.src = src_texts
        self.tgt = tgt_texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        s = self.src[idx]
        t = self.tgt[idx]

        src_ids = self.tok.encode(s, add_special_tokens=False)
        tgt_ids = self.tok.encode(t, add_special_tokens=False)

        src_ids = [bos_id] + src_ids[: self.max_len - 2] + [eos_id]
        tgt_ids = [bos_id] + tgt_ids[: self.max_len - 2] + [eos_id]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgts = pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    return srcs, tgts

train_ds = TranslationDataset(train_src, train_tgt, tokenizer)
valid_ds = TranslationDataset(valid_src, valid_tgt, tokenizer)
test_ds  = TranslationDataset(test_src,  test_tgt,  tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========== MODEL ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model)

        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        tgt_mask = self.tf.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        src_pad = (src == pad_id)
        tgt_pad = (tgt == pad_id)

        src_emb = self.pe(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pe(self.tgt_emb(tgt) * math.sqrt(self.d_model))

        out = self.tf(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad
        )
        return self.fc(out)

vocab_size = len(tokenizer)
print("Vocab size:", vocab_size)
model = Seq2SeqTransformer(vocab_size, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT).to(DEVICE)

# ========== TRAIN SETUP ==========
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

# ========== TRAIN / VALID FUNCTIONS ==========
def train_epoch():
    model.train()
    total_loss = 0
    for i, (src, tgt) in enumerate(tqdm(train_loader)):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if (i + 1) % PRINT_EVERY == 0:
            print(f"  iter {i+1} loss = {loss.item():.4f}")

    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in valid_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(valid_loader)

# ========== DECODING + BLEU ==========
def greedy_decode(src_tensor, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        memory = model.tf.encoder(model.pe(model.src_emb(src_tensor) * math.sqrt(D_MODEL)))
        ys = torch.tensor([[bos_id]], device=DEVICE)

        for _ in range(max_len):
            dec_in = model.pe(model.tgt_emb(ys) * math.sqrt(D_MODEL))
            out = model.tf.decoder(dec_in, memory)
            logits = model.fc(out[:, -1])
            next_id = logits.argmax(-1).item()
            ys = torch.cat([ys, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == eos_id:
                break

        return ys.squeeze(0).tolist()

def compute_bleu(ds):
    hyps, refs = [], []
    for i in range(len(ds)):
        src_ids, tgt_ids = ds[i]
        src_tensor = src_ids.unsqueeze(0).to(DEVICE)

        pred_ids = greedy_decode(src_tensor)
        pred_toks = [
            tokenizer.convert_ids_to_tokens(x)
            for x in pred_ids if x not in (pad_id, bos_id, eos_id)
        ]
        tgt_toks = [
            tokenizer.convert_ids_to_tokens(int(x))
            for x in tgt_ids if int(x) not in (pad_id, bos_id, eos_id)
        ]

        hyps.append(" ".join(pred_toks))
        refs.append([" ".join(tgt_toks)])

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score, hyps, refs

# ========== TRAIN LOOP ==========
best_val = 9999

for ep in range(1, EPOCHS + 1):
    print(f"\n=== EPOCH {ep}/{EPOCHS} ===")

    train_loss = train_epoch()
    val_loss = evaluate()

    print(f"Train loss={train_loss:.4f}  Val loss={val_loss:.4f}")

    if len(valid_ds) > 0:
        val_bleu_score, _, _ = compute_bleu(valid_ds)
    else:
        val_bleu_score = 0.0

    print(f"Validation BLEU (approx): {val_bleu_score:.2f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Saved best model.")

# ========== FINAL TEST EVAL ==========
print("\n[INFO] Evaluating on test set...")

if len(test_ds) > 0:
    test_bleu_score, hyp_texts, refs = compute_bleu(test_ds)
else:
    test_bleu_score, hyp_texts, refs = 0.0, [], []

print("TEST BLEU =", test_bleu_score)

# ==========================================================
# ========== SAVE 1. ORIGINAL → UNIGRAM TOKENIZER ==========
# ==========================================================

orig_out_path = os.path.join(OUTPUT_DIR, "test_original_inputs.txt")
print("[INFO] Saving ORIGINAL → UNIGRAM tokenizer outputs to:", orig_out_path)

with open(orig_out_path, "w", encoding="utf-8") as f:
    for i in range(len(test_ds)):
        src_raw = test_src[i]

        ids = tokenizer.encode(src_raw, add_special_tokens=False)
        toks = tokenizer.convert_ids_to_tokens(ids)

        f.write("=====================================\n")
        f.write(f"EXAMPLE {i+1}\n")
        f.write(f"SRC RAW: {src_raw}\n")
        f.write(f"TOKEN IDS: {ids}\n")
        f.write(f"TOKENS: {' '.join(toks)}\n\n")


# ==========================================================
# ========== 2. TOKENIZED INPUTS (same as original here) ===
# ==========================================================

tok_out_path = os.path.join(OUTPUT_DIR, "test_tokenized_inputs.txt")
print("[INFO] Saving tokenized inputs to:", tok_out_path)

with open(tok_out_path, "w", encoding="utf-8") as f:
    for i in range(len(test_ds)):
        src_text = test_src[i]

        ids = tokenizer.encode(src_text, add_special_tokens=False)
        toks = tokenizer.convert_ids_to_tokens(ids)

        f.write("=====================================\n")
        f.write(f"EXAMPLE {i+1}\n")
        f.write(f"SRC RAW: {src_text}\n")
        f.write(f"TOKEN IDS: {ids}\n")
        f.write(f"TOKENS: {' '.join(toks)}\n\n")


# ==========================================================
# ========== 3. SEQ2SEQ MODEL PREDICTIONS ==================
# ==========================================================

seq2seq_out_path = os.path.join(OUTPUT_DIR, "test_seq2seq_outputs.txt")
print("[INFO] Saving seq2seq outputs to:", seq2seq_out_path)

with open(seq2seq_out_path, "w", encoding="utf-8") as f:
    for i in range(len(test_ds)):
        src_ids, tgt_ids = test_ds[i]
        src_tensor = src_ids.unsqueeze(0).to(DEVICE)

        # Predict
        pred_ids = greedy_decode(src_tensor)
        pred_tokens = [
            tokenizer.convert_ids_to_tokens(x)
            for x in pred_ids if x not in (pad_id, bos_id, eos_id)
        ]
        pred_text = " ".join(pred_tokens)

        f.write("=====================================\n")
        f.write(f"EXAMPLE {i+1}\n")
        f.write(f"SRC RAW: {test_src[i]}\n")
        f.write(f"PRED TOKEN IDS: {pred_ids}\n")
        f.write(f"PRED TOKENS: {pred_text}\n")
        f.write(f"PRED TEXT: {pred_text}\n\n")

print("[INFO] All prediction types saved successfully!")
print("All outputs at:", OUTPUT_DIR)
