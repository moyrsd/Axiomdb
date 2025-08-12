from tokenizer import Tokenizer

new_tokenizer = Tokenizer()
new_tokenizer.load("my_tokenizer.bpe")

text_to_test = "Testing if tokenizer works after loading from file."
print(f"length of text : {len(text_to_test)}")
encoded = new_tokenizer.encode(text_to_test)
print(f"length of Encoded tokens : {len(encoded)}")
decoded = new_tokenizer.decode(encoded)

print(f"\nOriginal: '{text_to_test}'")
print(f"Encoded: {encoded}")
print(f"Decoded: '{decoded}'")