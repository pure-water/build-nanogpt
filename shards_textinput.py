import os
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import tiktoken

# Tokenization function for input.txt
def tokenize(doc_text):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc_text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# Function to write the shard data
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Main function to process input.txt
def process_data():
    local_dir = "inputtxt_dataset"
    shard_size = int(5e4)  # Set shard size to 50,000 tokens per shard
    input_file = os.path.join(local_dir, "input.txt")  # Input file to process
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir+"/shards")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Load the content of the input.txt file
    with open(input_file, 'r', encoding='utf-8') as f:
        doc_text = f.read()

    # Split text into paragraphs or sentences to simulate multiple documents
    docs = doc_text.split("\n\n")

    nprocs = max(1, os.cpu_count() // 2)
    nprocs = 1

    total_tokens_processed = 0  # To track the total number of tokens

    # Initialize multiprocessing pool
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        # Tokenize each document and write it into shards
        for tokens in pool.imap(tokenize, docs, chunksize=16):
            total_tokens_processed += len(tokens)  # Accumulate total tokens
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Assign first shard to 'val', the rest to 'train'
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{split}_shard_{shard_index:06d}.npy")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Write remaining tokens to the final shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{split}_shard_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])

    # Print total number of tokens processed
    print(f"Sharding complete. {shard_index + 1} shards created.")
    print(f"Total number of tokens processed: {total_tokens_processed}")

# Use the if __name__ == '__main__': guard
if __name__ == '__main__':
    mp.freeze_support()  # Necessary on Windows when using multiprocessing
    process_data()
