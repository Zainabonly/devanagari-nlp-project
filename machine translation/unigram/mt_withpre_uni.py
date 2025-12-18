# train_seq2seq_with_sandhi.py
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

from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer

# ========== CONFIG ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 50

TRAIN_CSV = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_train.csv"
VALID_CSV = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_val.csv"
TEST_CSV  = "/DATA/rohit/NLP_2025/dataset/machinetranslation/mix_test.csv"

UNIGRAM_TOKENIZER_PATH = "/DATA/rohit/NLP_2025/unigram"   # folder with tokenizer.json
SANDHI_MODEL_PATH = "/DATA/rohit/NLP_2025/byt5-small_new/mix/byt5_best_model"

OUTPUT_DIR = "./seq2seq_out"

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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ========== SANdhI SPLITTER (uses your fine-tuned byT5/byt5 model) ==========
class SandhiSplitter:
    def __init__(self, model_path, tokenizer_file=None):
        """
        Expects model_path to contain a seq2seq model (byt5) that takes raw mix
        and outputs sandhi-split text. We will try to load a tokenizer for it.
        """
        print(f"[SandhiSplitter] loading model from {model_path} ...")
        try:
            # try to load tokenizer saved with the model (fallback to AutoTokenizer)
            if tokenizer_file:
                self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_file)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
            self.model.eval()
        except Exception as e:
            print("[SandhiSplitter] WARNING: couldn't load splitter model/tokenizer:", e)
            self.model = None
            self.tokenizer = None

    def split_batch(self, texts, batch_size=16):
        """
        Accepts list of raw strings and returns list of split strings.
        If model not available, returns original texts.
        """
        if self.model is None:
            return texts

        outputs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            toks = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
            with torch.no_grad():
                gen = self.model.generate(toks.input_ids, attention_mask=toks.attention_mask, max_length=MAX_LEN)

            # print(f"GEN : {gen}")
            dec = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            # print(f"DE : {gen}")
            outputs.extend(dec)
        return outputs

# ========== TOKENIZER ==========
print("[INFO] Loading unigram tokenizer from:", UNIGRAM_TOKENIZER_PATH)
tokenizer = PreTrainedTokenizerFast.from_pretrained(UNIGRAM_TOKENIZER_PATH)

# Ensure special tokens exist (PAD=0, BOS=1, EOS=2) — we will map them.
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

print("Tokenizer special ids -> pad:", pad_id, "bos:", bos_id, "eos:", eos_id)

# ========== DATA LOADING FROM CSV (headers: 'mix','english') ==========
def read_csv_pairs(path):
    src, tgt = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "mix" not in reader.fieldnames or "english" not in reader.fieldnames:
            raise ValueError(f"CSV {path} must have columns: mix,english")
        for row in reader:
            s = row["mix"].strip()
            t = row["english"].strip()
            if s and t:
                src.append(s)
                tgt.append(t)
    return src, tgt

print("[INFO] Reading train/valid/test CSV files...")
train_src, train_tgt = read_csv_pairs(TRAIN_CSV)
valid_src, valid_tgt = read_csv_pairs(VALID_CSV)
test_src,  test_tgt  = read_csv_pairs(TEST_CSV)

print(f"Loaded CSV sizes: Train={len(train_src)}, Valid={len(valid_src)}, Test={len(test_src)}")

# ========== SANdHI SPLIT ON SRC SENTENCES ==========
print("[INFO] Loading Sandhi Splitter model and splitting source sentences...")
sandhi = SandhiSplitter(SANDHI_MODEL_PATH, tokenizer_file=None)  # if your sandhi model has its own tokenizer, set tokenizer_file
train_src_split = sandhi.split_batch(train_src, batch_size=32)
valid_src_split = sandhi.split_batch(valid_src, batch_size=32)
test_src_split  = sandhi.split_batch(test_src,  batch_size=32)

# Optional: inspect some pairs
for i in range(min(3, len(train_src))):
    print("TRAIN SRC RAW:", train_src[i])
    print("TRAIN SRC SPLIT:", train_src_split[i])
    print("TRAIN TGT    :", train_tgt[i])
    print("-"*50)

# ========== TranslationDataset (uses tokenizer.encode) ==========
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=MAX_LEN):
        assert len(src_texts) == len(tgt_texts)
        self.src = src_texts
        self.tgt = tgt_texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        s = self.src[idx]
        t = self.tgt[idx]

        # encode (returns list of ids). We want to use BOS/EOS and not include pad here.
        src_ids = self.tok.encode(s, add_special_tokens=False)
        tgt_ids = self.tok.encode(t, add_special_tokens=False)

        # add bos/eos
        src_ids = [bos_id] + src_ids[: self.max_len - 2] + [eos_id]
        tgt_ids = [bos_id] + tgt_ids[: self.max_len - 2] + [eos_id]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded

# ========== SEQ2SEQ TRANSFORMER MODEL ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # src: (B, S), tgt: (B, T)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_pad_mask = (src == pad_id).to(src.device)
        tgt_pad_mask = (tgt == pad_id).to(tgt.device)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.fc_out(out)

# ========== Prepare datasets (use CSV splits) ==========
train_dataset = TranslationDataset(train_src_split, train_tgt, tokenizer)
valid_dataset = TranslationDataset(valid_src_split, valid_tgt, tokenizer)
test_dataset  = TranslationDataset(test_src_split,  test_tgt,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========== Model initialization ==========
src_vocab_size = len(tokenizer)
tgt_vocab_size = len(tokenizer)
print("Vocab sizes -> src:", src_vocab_size, "tgt:", tgt_vocab_size)

model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT).to(DEVICE)

# Optionally copy pretrained token embeddings from a pre-trained model if available:
# (commented out; only if you have a pretrained mt5/byt5 you want to reuse)
# try:
#     pretrained = AutoModelForSeq2SeqLM.from_pretrained(MT5_MODEL_PATH)
#     with torch.no_grad():
#         emb = pretrained.get_input_embeddings().weight[:src_vocab_size]
#         model.src_embedding.weight.data[:emb.shape[0]] = emb
# except Exception as e:
#     print("No pretrained embedding loaded:", e)

# ========== Training utilities ==========
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS if len(train_loader) > 0 else 1
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

# ========== Training loop ==========
def train_one_epoch():
    model.train()
    total_loss = 0.0
    it = 0
    for src_batch, tgt_batch in tqdm(train_loader, desc="Training"):
        it += 1
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        optimizer.zero_grad()
        logits = model(src_batch, tgt_input)  # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if total_steps > 0:
            scheduler.step()
        total_loss += loss.item()
        if it % PRINT_EVERY == 0:
            print(f"  iter {it} loss = {loss.item():.4f}")

    return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

def validate():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in valid_loader:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            logits = model(src_batch, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0

# ========== Greedy decode for BLEU ==========
def greedy_decode(src_tensor, max_len=80):
    """
    src_tensor: (1, S) tensor on DEVICE
    returns: list of token ids (including BOS and maybe EOS)
    """
    model.eval()
    with torch.no_grad():
        memory = model.transformer.encoder(model.pos_encoder(model.src_embedding(src_tensor) * math.sqrt(model.d_model)))
        ys = torch.tensor([[bos_id]], dtype=torch.long).to(DEVICE)  # (1,1)
        for _ in range(max_len):
            # decoder needs (B, T, D)
            dec_in = model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.d_model))
            out = model.transformer.decoder(dec_in, memory)
            prob = model.fc_out(out[:, -1, :])  # (B, V)
            next_word = prob.argmax(dim=-1)     # (B,)
            next_word = next_word.unsqueeze(1)  # (B,1)
            ys = torch.cat([ys, next_word], dim=1)  # (B, T+1)
            if next_word.item() == eos_id:
                break
        return ys.squeeze(0).tolist()

def compute_bleu_for_dataset(dataset):
    refs, hyps = [], []
    for i in range(len(dataset)):
        src_ids, tgt_ids = dataset[i]
        src_tensor = src_ids.unsqueeze(0).to(DEVICE)
        hyp_ids = greedy_decode(src_tensor, max_len=MAX_LEN)
        # remove special tokens
        hyp_tokens = [tokenizer.convert_ids_to_tokens(i) for i in hyp_ids if i not in (pad_id, bos_id, eos_id)]
        tgt_tokens = [tokenizer.convert_ids_to_tokens(int(i)) for i in tgt_ids if int(i) not in (pad_id, bos_id, eos_id)]
        # convert tokens to strings (PreTrainedTokenizerFast may return subword tokens; join with space)
        hyps.append(" ".join(hyp_tokens).strip())
        refs.append([ " ".join(tgt_tokens).strip() ])
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score, hyps, refs

# ========== Run training ==========
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    print(f"\n==== EPOCH {epoch}/{EPOCHS} ====")
    train_loss = train_one_epoch()
    val_loss = validate()
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # compute BLEU on validation small subset if you want (optional)
    val_bleu, _, _ = compute_bleu_for_dataset(valid_dataset) if len(valid_dataset)>0 else (0.0, [], [])
    print(f"Validation BLEU (approx): {val_bleu:.2f}")

    # save best
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Saved best model.")

# ========== Final evaluate on test ==========
print("[INFO] Running final BLEU evaluation on test set...")
test_bleu, hyp_texts, refs = compute_bleu_for_dataset(test_dataset)
print(f"Test BLEU = {test_bleu:.2f}")

# ==========================================================
# ========== SAVE 3-STAGE PIPELINE PREDICTIONS =============
# ==========================================================

# ========== 1. PRETOKENIZER (UNIGRAM TOKENIZER) OUTPUT ==========
pretok_path = os.path.join(OUTPUT_DIR, "test_pretok_outputs.txt")
print("[INFO] Saving PRETOKENIZER outputs to:", pretok_path)

with open(pretok_path, "w", encoding="utf-8") as f:
    for i in range(len(test_dataset)):
        src_raw = test_src[i]

        # Unigram tokenizer raw tokenization
        ids = tokenizer.encode(src_raw, add_special_tokens=False)
        toks = tokenizer.convert_ids_to_tokens(ids)

        f.write("=====================================\n")
        f.write(f"EXAMPLE {i+1}\n")
        f.write(f"SRC RAW: {src_raw}\n")
        f.write(f"PRETOKEN IDS: {ids}\n")
        f.write(f"PRETOKEN TOKENS: {' '.join(toks)}\n\n")


# ========== 2. TOKENIZER OUTPUT AFTER SANDHI SPLIT ==========
tok_path = os.path.join(OUTPUT_DIR, "test_tok_outputs.txt")
print("[INFO] Saving TOKENIZED (post-sandhi) outputs to:", tok_path)

with open(tok_path, "w", encoding="utf-8") as f:
    for i in range(len(test_dataset)):
        src_split = test_src_split[i]

        ids = tokenizer.encode(src_split, add_special_tokens=False)
        toks = tokenizer.convert_ids_to_tokens(ids)

        f.write("=====================================\n")
        f.write(f"EXAMPLE {i+1}\n")
        f.write(f"SRC SPLIT: {src_split}\n")
        f.write(f"SPLIT TOKEN IDS: {ids}\n")
        f.write(f"SPLIT TOKEN TOKENS: {' '.join(toks)}\n\n")


# ========== 3. SEQ2SEQ MODEL PREDICTIONS ==========
seq2seq_path = os.path.join(OUTPUT_DIR, "test_seq2seq_outputs.txt")
print("[INFO] Saving SEQ2SEQ predicted outputs to:", seq2seq_path)

with open(seq2seq_path, "w", encoding="utf-8") as f:
    for i in range(len(test_dataset)):
        src_ids, tgt_ids = test_dataset[i]
        src_tensor = src_ids.unsqueeze(0).to(DEVICE)

        # seq2seq prediction
        pred_ids = greedy_decode(src_tensor, max_len=MAX_LEN)
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


print("[INFO] All 3-stage predictions saved successfully!")
print("All outputs stored in:", OUTPUT_DIR)
