import re
import collections
import json
import os
import csv
import sys

# --- 1. Configuration & Constants ---
CONSONANTS = set(range(0x0915, 0x0939))
MATRAS = set(range(0x093E, 0x094C))
VIRAMA = 0x094D

# Special Tokens
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

# --- 2. Helper Functions ---

def read_csv_corpus(file_path, column_name, sample_limit=None):
    """
    Reads text from a specific column in a CSV file.
    """
    print(f"Reading data from {file_path} (Column: {column_name})...")
    text_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if column_name not in reader.fieldnames:
                raise ValueError(f"Column '{column_name}' not found in CSV headers: {reader.fieldnames}")
            
            for i, row in enumerate(reader):
                if sample_limit and i >= sample_limit:
                    break
                content = row[column_name]
                if content and isinstance(content, str):
                    text_data.append(content.strip())
                    
        print(f"Loaded {len(text_data)} lines.")
        return " ".join(text_data) # Join all lines into one big text blob
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def get_weighted_stats(vocab, boost_factor=1.3):
    """
    Calculates pair frequencies with your custom Indic Boosting logic.
    """
    pairs = collections.defaultdict(float)

    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            weight = 1.0
            
            # Analyze last char of first token and first char of second token
            # symbols[i] might be a merged token like "ka", so we look at the last char
            char_a = ord(symbols[i][-1]) 
            char_b = ord(symbols[i+1][0])

            # RULE A: Consonant + (Virama or Matra)
            if (char_a in CONSONANTS) and (char_b == VIRAMA or char_b in MATRAS):
                weight = boost_factor

            # RULE B: Virama + Consonant (Conjuncts)
            elif (char_a == VIRAMA) and (char_b in CONSONANTS):
                weight = boost_factor

            pairs[pair] += freq * weight

    return pairs

def merge_vocab(pair, vocab):
    """
    Merges the best pair in the current vocabulary.
    """
    bigram = re.escape(' '.join(pair))
    # Regex to find the pair only when separated by space
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    out = {}
    for word in vocab:
        # Replace "a b" with "ab"
        w_out = p.sub(''.join(pair), word)
        out[w_out] = vocab[word]
    return out

# --- 3. Main Training Loop ---

def train_indic_tokenizer(csv_file, column_name, num_merges=1000, output_dir="output"):
    # A. Load Data
    raw_text = read_csv_corpus(csv_file, column_name)
    
    print("Preprocessing vocabulary...")
    # Simple whitespace tokenization
    words = raw_text.split()
    vocab = collections.Counter(words)
    
    # Split into chars: "hello" -> "h e l l o"
    vocab = {" ".join(list(word)): count for word, count in vocab.items()}

    # Store merge history
    merges = []

    print(f"Starting training for {num_merges} merges...")
    
    for i in range(num_merges):
        # 1. Calculate Weighted Stats
        pairs = get_weighted_stats(vocab)

        if not pairs:
            break

        # 2. Pick best pair (highest weighted score)
        best = max(pairs, key=pairs.get)
        
        # 3. Update Vocab
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        
        if (i + 1) % 50 == 0:
            print(f"Merge {i+1}/{num_merges}: {best} | Score: {pairs[best]:.2f}")

    # --- 4. Build Final Vocabulary ---
    print("Building final vocabulary...")
    
    # Define special tokens IDs
    token_map = {UNK_TOKEN: 0, PAD_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    
    # Collect all unique tokens remaining in the vocab
    unique_tokens = set()
    for word in vocab.keys():
        for token in word.split():
            unique_tokens.add(token)
            
    # Also ensure all individual characters from original text are included 
    # (In case they were never merged)
    # Note: In a strict BPE, we usually initialize with all chars. 
    # Here we just add what's found in the final state + history.
    for pair in merges:
        unique_tokens.add("".join(pair))
        unique_tokens.add(pair[0])
        unique_tokens.add(pair[1])

    # Assign IDs
    current_id = len(token_map)
    for token in sorted(list(unique_tokens)):
        if token not in token_map:
            token_map[token] = current_id
            current_id += 1

    # --- 5. Save Files ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # File 1: tokenizer.vocab (JSON format)
    vocab_path = os.path.join(output_dir, "tokenizer.vocab")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(token_map, f, ensure_ascii=False, indent=2)

    # File 2: merges.txt (Standard BPE format)
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        # Usually writes a version line first, but simple list is fine
        f.write("#version: 0.2\n") 
        for p1, p2 in merges:
            f.write(f"{p1} {p2}\n")

    # File 3: tokenizer.model
    # We will save this as a JSON that contains EVERYTHING needed to load it back
    # cleanly without needing separate files if preferred.
    model_data = {
        "type": "WeightedIndicBPE",
        "version": "1.0",
        "vocab": token_map,
        "merges": merges
    }
    model_path = os.path.join(output_dir, "tokenizer.model")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False)

    print("-" * 30)
    print("Training Complete!")
    print(f"Vocab Size: {len(token_map)}")
    print(f"1. Vocab saved to: {vocab_path}")
    print(f"2. Merges saved to: {merges_path}")
    print(f"3. Model saved to: {model_path}")
    print("-" * 30)

# --- 6. Execution ---
if __name__ == "__main__":
    # !!! CHANGE THESE VARIABLES !!!
    CSV_FILE = "mix_train.csv"         # Your CSV file path
    COLUMN_NAME = "mix"          # The header of the column containing text
    NUM_MERGES = 3000            # How many BPE merges to perform

    # Create a dummy csv for demonstration if it doesn't exist
    if not os.path.exists(CSV_FILE):
        print("Creating dummy CSV for testing...")
        with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text"])
            writer.writerow([1, "नमस्ते भारत।"])
            writer.writerow([2, "संयुक्त अक्षर जैसे क्ष त्र ज्ञ महत्वपूर्ण हैं।"])
            writer.writerow([3, "मैं हिंदी सीख रहा हूँ।"])

    train_indic_tokenizer(CSV_FILE, COLUMN_NAME, NUM_MERGES)