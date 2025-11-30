import numpy as np
import regex as re
from numba import njit, types
from numba.typed import Dict
from custom_bpe import CustomBPETokenizer, GPT4_SPLIT_PATTERN

PAIR_TYPE = types.UniTuple(types.int64, 2)

@njit
def count_pairs_numba(ids_array):
    """
    JIT-compiled function to count pairs in a flat integer array.
    Skips pairs involving -1 (the boundary marker).
    """
   
    counts = Dict.empty(
        key_type=PAIR_TYPE,
        value_type=types.int64
    )
    
    n = len(ids_array)
    for i in range(n - 1):
        a = ids_array[i]
        b = ids_array[i+1]
        
        # Don't merge across boundaries (-1)
        if a == -1 or b == -1:
            continue
            
        pair = (a, b)
        if pair in counts:
            counts[pair] += 1
        else:
            counts[pair] = 1
            
    return counts

@njit
def merge_numba(ids_array, p0, p1, new_id):
    """
    JIT-compiled function to replace (p0, p1) -> new_id.
    Returns a new, slightly shorter numpy array.
    """
    n = len(ids_array)
    # Pass 1: Calculate the size of the new array
    new_len = 0
    i = 0
    while i < n:
        if i < n - 1 and ids_array[i] == p0 and ids_array[i+1] == p1:
            new_len += 1
            i += 2
        else:
            new_len += 1
            i += 1
            
    # Pass 2: Fill the new array
    new_ids = np.empty(new_len, dtype=np.int64)
    idx = 0
    i = 0
    while i < n:
        if i < n - 1 and ids_array[i] == p0 and ids_array[i+1] == p1:
            new_ids[idx] = new_id
            i += 2
        else:
            new_ids[idx] = ids_array[i]
            i += 1
        idx += 1
        
    return new_ids

class NumbaBPETokenizer(CustomBPETokenizer):
    def train(self, text: str, vocab_size: int = 30000, verbose: bool = True):
        print("Pre-tokenizing text...")
        compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
        text_chunks = re.findall(compiled_pattern, text)
        
        # 1. Convert text to a single flat numpy array with -1 separators
        # This is much friendlier to CPU cache and Numba than lists of lists
        print("Converting to numpy array for Numba acceleration...")
        all_ids = []
        for chunk in text_chunks:
            all_ids.extend(list(chunk.encode("utf-8")))
            all_ids.append(-1) # Boundary marker
            
        # Convert to numpy int64 to match Numba types
        ids_array = np.array(all_ids, dtype=np.int64)
        print(f"Training on {len(ids_array)} tokens using Numba JIT...")

        num_merges = vocab_size - 256
        
        for i in range(num_merges):
            # 2. Fast Count (JIT)
            stats = count_pairs_numba(ids_array)
            
            if not stats:
                break

            # Find most frequent pair
            # We have to iterate the typed Dict in python to find max
            # This is fast enough because vocab is small compared to text
            top_pair = None
            max_count = -1
            for pair, count in stats.items():
                if count > max_count:
                    max_count = count
                    top_pair = pair
            
            # 3. Update vocab
            idx = 256 + i
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            # 4. Fast Merge (JIT)
            ids_array = merge_numba(ids_array, top_pair[0], top_pair[1], idx)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {idx} (count: {max_count})")

        self.vocab_size_val = 256 + len(self.merges)
        print(f"Training complete. Final Vocab Size: {self.vocab_size_val}")