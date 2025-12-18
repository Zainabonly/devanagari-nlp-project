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
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# ========== CONFIG ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 50

TRAIN_CSV = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_train.csv"
VALID_CSV = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_val.csv"
TEST_CSV  = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_test.csv"

UNIGRAM_TOKENIZER_PATH = "/DATA/rohit/NLP_2025/unigram"

OUTPUT_DIR = "./seq2seq_out_norm"

# model hyperparams
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
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ========== NORMALIZER ==========
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("sa")

# ========== TOKENIZER ==========
print("[INFO] Loading tokenizer:", UNIGRAM_TOKENIZER_PATH)
tokenizer = PreTrainedTokenizerFast.from_pretrained(UNIGRAM_TOKENIZER_PATH)

specials = {}
if tokenizer.pad_token is None: specials["pad_token"] = "<pad>"
if tokenizer.bos_token is None: specials["bos_token"] = "<s>"
if tokenizer.eos_token is None: specials["eos_token"] = "</s>"

if specials:
    tokenizer.add_special_tokens(specials)

pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

print("Special IDs -> pad:", pad_id, " bos:", bos_id, " eos:", eos_id)

# ========== CSV READER ==========
def read_csv_pairs(path):
    src, tgt = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "mix" not in reader.fieldnames or "english" not in reader.fieldnames:
            raise ValueError(f"{path} must contain columns: mix, english")

        for row in reader:
            s = row["mix"].strip()
            t = row["english"].strip()

            if not s or not t:
                continue

            # normalize
            s = normalizer.normalize(s)
            t = normalizer.normalize(t)

            src.append(s)
            tgt.append(t)

    return src, tgt


print("[INFO] Loading CSV datasets...")
train_src, train_tgt = read_csv_pairs(TRAIN_CSV)
valid_src, valid_tgt = read_csv_pairs(VALID_CSV)
test_src,  test_tgt  = read_csv_pairs(TEST_CSV)

print(f"Train={len(train_src)}  Valid={len(valid_src)}  Test={len(test_src)}")

# ========== DATASET ==========
class TranslationDataset(Dataset):
    def __init__(self, src, tgt, tokenizer, max_len=MAX_LEN):
        self.src = src
        self.tgt = tgt
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
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=pad_id)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_id)
    return src, tgt


train_ds = TranslationDataset(train_src, train_tgt, tokenizer)
valid_ds = TranslationDataset(valid_src, valid_tgt, tokenizer)
test_ds  = TranslationDataset(test_src,  test_tgt, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========== MODEL ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model)

        self.tf = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=layers, num_decoder_layers=layers,
            dim_feedforward=2048, dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, vocab)

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
model = Seq2SeqTransformer(vocab_size, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT).to(DEVICE)

# ========== TRAINING SETUP ==========
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.1, total_steps)

# ========== EPOCH FUNCTIONS ==========
def train_epoch():
    model.train()
    total_loss = 0

    for i, (src, tgt) in enumerate(tqdm(train_loader)):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if i % PRINT_EVERY == 0:
            print("Step:", i, "Loss:", loss.item())

    return total_loss / len(train_loader)


def evaluate():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in valid_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(valid_loader)


# ========== GREEDY DECODE ==========
def greedy_decode(src_tensor):
    model.eval()
    with torch.no_grad():
        memory = model.tf.encoder(
            model.pe(model.src_emb(src_tensor) * math.sqrt(D_MODEL))
        )

        ys = torch.tensor([[bos_id]], device=DEVICE)

        for _ in range(MAX_LEN):
            tgt_emb = model.pe(model.tgt_emb(ys) * math.sqrt(D_MODEL))
            out = model.tf.decoder(tgt_emb, memory)
            logits = model.fc(out[:, -1, :])

            next_id = logits.argmax(-1).item()

            if next_id == eos_id:
                break

            ys = torch.cat([ys, torch.tensor([[next_id]], device=DEVICE)], dim=1)

        return ys.squeeze(0).tolist()


# ========== BLEU COMPUTATION ==========
def compute_bleu(dataset):
    hyps, refs = [], []

    for src_ids, tgt_ids in dataset:
        src_tensor = src_ids.unsqueeze(0).to(DEVICE)
        pred_ids = greedy_decode(src_tensor)

        hyp_tokens = [
            tokenizer.convert_ids_to_tokens(int(i))
            for i in pred_ids
            if int(i) not in (pad_id, bos_id, eos_id)
        ]

        tgt_tokens = [
            tokenizer.convert_ids_to_tokens(int(i))
            for i in tgt_ids
            if int(i) not in (pad_id, bos_id, eos_id)
        ]

        hyp_str = " ".join(hyp_tokens)
        tgt_str = " ".join(tgt_tokens)

        hyps.append(hyp_str)
        refs.append([tgt_str])

    bleu_score = sacrebleu.corpus_bleu(hyps, refs).score

    return bleu_score, hyps, refs



# ========== TRAIN LOOP ==========
best_val = float("inf")

for ep in range(1, EPOCHS + 1):
    print(f"\n=== EPOCH {ep}/{EPOCHS} ===")
    tr = train_epoch()
    va = evaluate()
    print(f"Train Loss={tr:.4f} | Val Loss={va:.4f}")

    if va < best_val:
        best_val = va
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best.pt")
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Saved BEST model")

# ========== FINAL TEST BLEU ==========
print("\n[INFO] Evaluating on TEST set...")
test_bleu_score, hyp_texts, refs = compute_bleu(test_ds)
print("TEST BLEU =", test_bleu_score)

# ==========================================================
# ========== 1. ORIGINAL → UNIGRAM TOKENIZER OUTPUT =========
# ==========================================================

orig_out_path = os.path.join(OUTPUT_DIR, "test_original_inputs_normal.txt")
print("[INFO] Saving ORIGINAL → UNIGRAM tokenizer outputs to:", orig_out_path)

with open(orig_out_path, "w", encoding="utf-8") as f:
    for i in range(len(test_ds)):
        src_raw = test_src[i]   # already normalized

        ids = tokenizer.encode(src_raw, add_special_tokens=False)
        toks = tokenizer.convert_ids_to_tokens(ids)

        f.write("=====================================\n")
        f.write(f"EXAMPLE {i+1}\n")
        f.write(f"SRC RAW: {src_raw}\n")
        f.write(f"TOKEN IDS: {ids}\n")
        f.write(f"TOKENS: {' '.join(toks)}\n\n")


# ==========================================================
# ========== 2. TOKENIZED INPUTS (same here) ================
# ==========================================================

tok_out_path = os.path.join(OUTPUT_DIR, "test_tokenized_inputs_normal.txt")
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

seq_out_path = os.path.join(OUTPUT_DIR, "test_seq2seq_outputs_normal.txt")
print("[INFO] Saving seq2seq predictions to:", seq_out_path)

with open(seq_out_path, "w", encoding="utf-8") as f:
    for i in range(len(test_ds)):
        src_ids, tgt_ids = test_ds[i]
        src_tensor = src_ids.unsqueeze(0).to(DEVICE)

        # run seq2seq model
        pred_ids = greedy_decode(src_tensor)
        pred_tokens = [
            tokenizer.convert_ids_to_tokens(t)
            for t in pred_ids if t not in (pad_id, bos_id, eos_id)
        ]
        pred_text = " ".join(pred_tokens)

        f.write("=====================================\n")
        f.write(f"EXAMPLE {i+1}\n")
        f.write(f"SRC RAW: {test_src[i]}\n")
        f.write(f"PRED TOKEN IDS: {pred_ids}\n")
        f.write(f"PRED TOKENS: {pred_text}\n")
        f.write(f"PRED TEXT: {pred_text}\n\n")

print("[INFO] All prediction files saved successfully!")
print("All outputs at:", OUTPUT_DIR)
