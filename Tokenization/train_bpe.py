import sentencepiece as spm

input_file = "/DATA/rohit/NLP_2025/dataset/tok/token_train.csv"

spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix="bpe_mixed",
    model_type="bpe",     # ← IMPORTANT CHANGE
    vocab_size=3200,
    character_coverage=1.0,
    num_threads=16,       # optional, speeds up training
    train_extremely_large_corpus=False
)

print("BPE tokenizer trained successfully!")