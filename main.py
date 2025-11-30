from axiomdb.tokenizers.custom_bpe import CustomBPETokenizer
import os

def main():
    # Path to the tokenizer file you just trained
    tokenizer_path = "axiomdb/tokenizers/axiom_tokenizer.json"
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: {tokenizer_path} not found. Please run train_bpe.py first.")
        return

    print("Loading Custom BPE Tokenizer...")
    tokenizer = CustomBPETokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"Tokenizer loaded successfully! Vocab Size: {tokenizer.vocab_size()}")

    # Test sentence
    text = "AxiomDB uses a custom BPE tokenizer trained on Wikipedia."
    print(f"\nOriginal Text: '{text}'")

    # Encode
    token_ids = tokenizer.tokenize(text)
    print(f"Token IDs: {token_ids}")

    # Decode (Reconstruct text from IDs)
    # Since we implemented a byte-level BPE, we reconstruct by joining bytes
    decoded_bytes = b""
    for idx in token_ids:
        decoded_bytes += tokenizer.vocab[idx]
    
    decoded_text = decoded_bytes.decode("utf-8", errors="replace")
    print(f"Decoded Text: '{decoded_text}'")

    # Verification
    if text == decoded_text:
        print("\nSUCCESS: Encoding and Decoding are perfectly reversible! ✅")
    else:
        print("\nWARNING: Decoded text does not match original. Check special characters or boundaries. ⚠️")

if __name__ == "__main__":
    main()