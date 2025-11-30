import regex as re
import json
from typing import List, Dict, Tuple
from collections import defaultdict
from .base import BaseTokenizer


GPT4_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class CustomBPETokenizer(BaseTokenizer):
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.special_tokens: Dict[str, int] = {}
        self.vocab_size_val = 256
        
        # Initialize base vocabulary (0-255)
        for i in range(256):
            self.vocab[i] = bytes([i])

    def train(self, text: str, vocab_size: int = 30000, verbose: bool = True):
        """
        Trains the BPE tokenizer on the provided text corpus.
        This is the pure Python implementation (slow).
        """
        print(f"Training BPE Tokenizer on {len(text)} characters...")
        
        compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
        text_chunks = re.findall(compiled_pattern, text)
        
        ids_list = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = defaultdict(int)
            
            for chunk_ids in ids_list:
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    stats[pair] += 1
            
            if not stats:
                break 

            top_pair = max(stats, key=stats.get)
            
            idx = 256 + i
            self.merges[top_pair] = idx
            
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            ids_list = [self._merge_chunk(chunk, top_pair, idx) for chunk in ids_list]
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {idx} ({self.vocab[idx]})")

        self.vocab_size_val = 256 + len(self.merges)
        print(f"Training complete. Final Vocab Size: {self.vocab_size_val}")

    def _merge_chunk(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """
        Replaces all occurrences of 'pair' with 'idx' in a single list of integers.
        """
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def tokenize(self, text: str) -> List[int]:
        """
        Encodes a string into a list of token IDs using the trained merges.
        """
        compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
        text_chunks = re.findall(compiled_pattern, text)
        
        final_ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            
            # Apply merges in order
            while len(chunk_ids) >= 2:
                stats = defaultdict(int)
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    stats[pair] += 1
                
                pair_to_merge = None
                min_idx = float("inf")
                
                for pair in stats:
                    if pair in self.merges:
                        if self.merges[pair] < min_idx:
                            min_idx = self.merges[pair]
                            pair_to_merge = pair
                
                if pair_to_merge is None:
                    break 
                
                chunk_ids = self._merge_chunk(chunk_ids, pair_to_merge, min_idx)
            
            final_ids.extend(chunk_ids)
            
        return final_ids

    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.tokenize(t) for t in texts]

    def vocab_size(self) -> int:
        return self.vocab_size_val

    def save(self, path: str):
        """Save the tokenizer merges to a JSON file."""
        export_merges = [[p[0], p[1], idx] for p, idx in self.merges.items()]
        data = {
            "vocab_size": self.vocab_size_val,
            "merges": export_merges
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load the tokenizer merges from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.vocab_size_val = data["vocab_size"]
        self.merges = {tuple(item[:2]): item[2] for item in data["merges"]}
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (p0, p1), idx in sorted_merges:
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        print(f"Tokenizer loaded from {path}")