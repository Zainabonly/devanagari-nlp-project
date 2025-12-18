import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import json
import math
import os
import sys
from tqdm import tqdm  # For the "Rectangle" progress bar
from nltk.translate.bleu_score import corpus_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PATHS
# Update this to your actual ByT5 model path
BYT5_MODEL_PATH = "/home/teaching/Cohesive_BPE/byt5_best_model" 
INDIC_TOKENIZER_PATH = "output/tokenizer.model" # Your custom trained model

FILES = {
    'train': 'mix_train.csv',
    'val':   'mix_val.csv',
    'test':  'mix_test.csv'
}

OUTPUT_LOG_FILE = "translation_results.txt"

# Hyperparameters
BATCH_SIZE = 16
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 3
EPOCHS = 1
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_LEN = 128

# ==========================================
# 1. SANDHI SPLITTER (ByT5 Pre-processing)
# ==========================================
class SandhiSplitter:
    def __init__(self, model_path):
        print(f"[SandhiSplitter] Loading model from {model_path} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
            self.model.eval()
            self.loaded = True
        except Exception as e:
            print(f"[SandhiSplitter] ⚠️ WARNING: Could not load model: {e}")
            print("Using raw text without splitting.")
            self.loaded = False

    def split_batch(self, texts, batch_size=64):
        """
        Runs inference on a list of strings to split Sandhi.
        """
        if not self.loaded:
            return texts
            
        print(f"[SandhiSplitter] Processing {len(texts)} lines...")
        results = []
        
        # Process in chunks to prevent OOM
        for i in tqdm(range(0, len(texts), batch_size), desc="Pre-tokenizing with ByT5"):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize inputs
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
            
            with torch.no_grad():
                # Generate
                outputs = self.model.generate(inputs.input_ids, max_length=128)
            
            # Decode
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)
            
        return results

# ==========================================
# 2. CUSTOM SOURCE TOKENIZER (INDIC)
# ==========================================
class CustomIndicBPE:
    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}
        self.merges = []
        # Default IDs
        self.unk_id = 0
        self.pad_id = 1
        self.sos_id = 2
        self.eos_id = 3

    def load(self, filename):
        print(f"[CustomBPE] Loading from {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        if "merges" in data:
            self.merges = [tuple(pair) for pair in data["merges"]]
            
        # Sync IDs from loaded vocab
        self.unk_id = self.vocab.get("[UNK]", 0)
        self.pad_id = self.vocab.get("[PAD]", 1)
        self.sos_id = self.vocab.get("[BOS]", 2)
        self.eos_id = self.vocab.get("[EOS]", 3)

    def _bpe(self, token):
        if len(token) <= 1: return list(token)
        word = list(token)
        for pair in self.merges:
            if len(word) < 2: break
            p1, p2 = pair
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                    new_word.append(p1 + p2)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text):
        if not isinstance(text, str): text = str(text)
        words = text.split()
        ids = []
        for word in words:
            for token in self._bpe(word):
                ids.append(self.vocab.get(token, self.unk_id))
        return ids
    
    def decode(self, ids):
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, "")
            if token not in ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]:
                tokens.append(token)
        return " ".join(tokens)

# ==========================================
# 3. DATASET
# ==========================================
class PreTokenizedDataset(Dataset):
    def __init__(self, csv_file, src_bpe, tgt_tokenizer, precomputed_splits=None):
        self.df = pd.read_csv(csv_file)
        self.df.columns = [c.lower() for c in self.df.columns]
        self.df = self.df.dropna()
        
        self.src_bpe = src_bpe
        self.tgt_tokenizer = tgt_tokenizer
        
        # Use precomputed splits if provided, else raw text
        if precomputed_splits:
            self.src_texts = precomputed_splits
        else:
            self.src_texts = self.df['mix'].astype(str).tolist()
            
        self.tgt_texts = self.df['english'].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Source: Custom BPE (Add SOS/EOS manually)
        src_txt = self.src_texts[idx]
        src_ids = [self.src_bpe.sos_id] + self.src_bpe.encode(src_txt) + [self.src_bpe.eos_id]
        
        # 2. Target: Standard (Auto adds specials)
        tgt_txt = self.tgt_texts[idx]
        tgt_ids = self.tgt_tokenizer.encode(tgt_txt, add_special_tokens=True)
        
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    # Pad Source (Custom Pad ID usually 1)
    src_batch = pad_sequence(src_batch, padding_value=1, batch_first=True)
    # Pad Target (BERT Pad ID usually 0)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_batch, tgt_batch

# ==========================================
# 4. TRANSFORMER
# ==========================================
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
    def __init__(self, src_vocab, tgt_vocab, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, dim_feedforward=1024, 
                                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_pad = (src == 1).to(src.device) # Custom Pad ID
        tgt_pad = (tgt == 0).to(tgt.device) # BERT Pad ID
        
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, 
                               src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
        return self.fc_out(out)

# ==========================================
# 5. LOGGING & EVALUATION
# ==========================================
def generate_and_log_outputs(model, dataset, src_bpe, tgt_tokenizer, filename):
    """
    Runs inference, calculates BLEU, and saves detailed output to a file.
    """
    model.eval()
    print(f"\nGenerations will be saved to: {filename}")
    
    refs = []
    hyps = []
    
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("SOURCE_TEXT | REFERENCE_TEXT | PREDICTED_TEXT\n")
        f.write("-" * 80 + "\n")

        with torch.no_grad():
            # TQDM for inference progress
            for src, tgt in tqdm(loader, desc="Generating Predictions"):
                src = src.to(DEVICE)
                
                # --- 1. Greedy Decode ---
                # Start with [CLS] (BERT ID 101)
                ys = torch.ones(1, 1).fill_(101).type(torch.long).to(DEVICE)
                
                for _ in range(MAX_LEN):
                    out = model(src, ys)
                    prob = out[:, -1]
                    _, next_word = torch.max(prob, dim=1)
                    next_word = next_word.item()
                    
                    ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
                    if next_word == 102: # [SEP]
                        break
                
                # --- 2. Decode IDs to Strings ---
                # Remove special tokens for clean text
                hyp_ids = [t for t in ys[0].tolist() if t not in [0, 101, 102]]
                hyp_text = tgt_tokenizer.decode(hyp_ids)
                
                tgt_ids = [t for t in tgt[0].tolist() if t not in [0, 101, 102]]
                ref_text = tgt_tokenizer.decode(tgt_ids)
                
                # Source text (Reconstruct from IDs for verification)
                src_clean_ids = [t for t in src[0].tolist() if t not in [src_bpe.pad_id, src_bpe.sos_id, src_bpe.eos_id]]
                src_text_recon = src_bpe.decode(src_clean_ids)

                # --- 3. Save to File ---
                f.write(f"{src_text_recon} | {ref_text} | {hyp_text}\n")
                
                # Collect for BLEU
                refs.append([ref_text.split()])
                hyps.append(hyp_text.split())

    # Calculate BLEU
    bleu_score = corpus_bleu(refs, hyps) * 100
    print(f" Final BLEU Score: {bleu_score:.2f}")
    with open(filename, "a", encoding="utf-8") as f:
        f.write("-" * 80 + "\n")
        f.write(f"FINAL BLEU SCORE: {bleu_score:.2f}\n")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    print(f"Running on Device: {DEVICE}")
    
    # 1. Load Tokenizers
    src_bpe = CustomIndicBPE()
    if os.path.exists(INDIC_TOKENIZER_PATH):
        src_bpe.load(INDIC_TOKENIZER_PATH)
    else:
        print(f"ERROR: {INDIC_TOKENIZER_PATH} not found.")
        return

    print("Loading Target Tokenizer (bert-base-uncased)...")
    tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 2. Load ByT5 Splitter
    sandhi_splitter = SandhiSplitter(BYT5_MODEL_PATH)

    # 3. Process Data (Pre-calculate Splits)
    # We read the CSV once, run ByT5 on 'mix' column, then pass to Dataset
    def prepare_data(csv_path):
        if not os.path.exists(csv_path): return None
        print(f"\nProcessing {csv_path}...")
        df = pd.read_csv(csv_path)
        raw_texts = df['mix'].astype(str).tolist()
        # Run ByT5 Batch Inference
        split_texts = sandhi_splitter.split_batch(raw_texts, batch_size=32)
        return PreTokenizedDataset(csv_path, src_bpe, tgt_tokenizer, precomputed_splits=split_texts)

    train_dataset = prepare_data(FILES['train'])
    val_dataset   = prepare_data(FILES['val'])
    test_dataset  = prepare_data(FILES['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 4. Initialize Model
    print(f"\nSource Vocab: {len(src_bpe.vocab)} | Target Vocab: {tgt_tokenizer.vocab_size}")
    model = Seq2SeqTransformer(
        src_vocab=len(src_bpe.vocab), 
        tgt_vocab=tgt_tokenizer.vocab_size, 
        d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore Padding

    # 5. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        
        # Rectangle Progress Bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
        
        epoch_loss = 0
        for src, tgt in loop:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            
            optimizer.zero_grad()
            out = model(src, tgt_in)
            loss = criterion(out.reshape(-1, out.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                out = model(src, tgt_in)
                loss = criterion(out.reshape(-1, out.shape[-1]), tgt_out.reshape(-1))
                val_loss += loss.item()
        
        print(f"   Val Loss: {val_loss/len(val_loader):.4f}")

    # 6. Final Test & Logging
    print("\n--- Generating Predictions on Test Set ---")
    generate_and_log_outputs(model, test_dataset, src_bpe, tgt_tokenizer, OUTPUT_LOG_FILE)

if __name__ == "__main__":
    # Create dummy files for testing if needed
    if not os.path.exists("mix_train.csv"):
        print("Creating dummy data...")
        pd.DataFrame({'mix':['नमस्ते भारत'], 'english':['Hello India']}).to_csv("mix_train.csv", index=False)
        pd.DataFrame({'mix':['मेरा नाम'], 'english':['My name']}).to_csv("mix_val.csv", index=False)
        pd.DataFrame({'mix':['नमस्ते'], 'english':['Hello']}).to_csv("mix_test.csv", index=False)
        
    main()