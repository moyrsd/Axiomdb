import os
import itertools
from datasets import load_dataset


try:
    from numba_bpe import NumbaBPETokenizer as BPETokenizer
    print("[INFO] Using Numba-accelerated Tokenizer 🚀")
    IS_NUMBA = True
except ImportError as e:
    print(f"[WARNING] Numba import failed: {e}")
    from axiomdb.tokenizers.custom_bpe import CustomBPETokenizer as BPETokenizer
    print("[INFO] Using standard Tokenizer (Slower).")
    IS_NUMBA = False

def train():
    corpus_file = "wiki_corpus.txt"
    
    if not os.path.exists(corpus_file):
        print("Corpus not found. Downloading Wikipedia data...")
        wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train', streaming=True)
        num_articles_to_use = 10000

        print(f"Creating training corpus file from {num_articles_to_use} Wikipedia articles...")
        with open(corpus_file, "w", encoding="utf-8") as f:
            for i, article in enumerate(itertools.islice(wiki_dataset, num_articles_to_use)):
                text_content = article['text']
                f.write(text_content + "\n\n")
                if (i + 1) % 100 == 0:
                    print(f"  Downloaded {i + 1}/{num_articles_to_use} articles...", end="\r")
        print(f"\nFinished creating {corpus_file}!")
    else:
        print(f"Found existing {corpus_file}, skipping download.")

    print(f"Reading {corpus_file}...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()

 
    MAX_TRAIN_SIZE = 50_000_000 if IS_NUMBA else 5_000_000
    
    if len(text) > MAX_TRAIN_SIZE:
        print(f"\n[INFO] Dataset is {len(text)} characters.")
        print(f"Truncating to {MAX_TRAIN_SIZE} characters for optimal training time...")
        text = text[:MAX_TRAIN_SIZE]
    
    tokenizer = BPETokenizer()
    TARGET_VOCAB_SIZE = 30000
    
    print(f"Starting training (Target Vocab: {TARGET_VOCAB_SIZE})...")
    tokenizer.train(text, vocab_size=TARGET_VOCAB_SIZE, verbose=True)

    save_path = "axiom_tokenizer.json"
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")

    from axiomdb.tokenizers.custom_bpe import CustomBPETokenizer
    print("\nVerifying load compatibility...")
    new_tok = CustomBPETokenizer()
    new_tok.load(save_path)
    print(f"Loaded Tokenizer Vocab Size: {new_tok.vocab_size()}")

if __name__ == "__main__":
    train()