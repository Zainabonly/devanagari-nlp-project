import sentencepiece as spm

input_file = "/DATA/rohit/NLP_2025/dataset/tok/token_train.csv"

spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix="unigram_mixed",
    model_type="unigram",
    vocab_size=3200,     # FIXED
    character_coverage=1.0,
    
)

print("Unigram tokenizer trained successfully!")
