import os
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import tiktoken
import argparse

# Tokenization function for text input
def tokenize(doc_text):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc_text))
    tokens_np = np.array(tokens)
    
    # Check if tokens are within valid range
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# Function to write the shard data
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Function to check token array for issues (NaN, Inf) and print token preview
def inspect_tokens(tokens, shard_index):
    print(f"\nInspecting tokens in shard {shard_index}...")
    # Check for NaN and Inf
    if np.isnan(tokens).any():
        print("Warning: NaN detected in tokens!")
    if np.isinf(tokens).any():
        print("Warning: Inf detected in tokens!")
    
    # Print statistics
    print(f"Token preview (first 10 tokens): {tokens[:10]}")
    print(f"Token preview (last 10 tokens): {tokens[-10:]}")
    print(f"Max token value: {tokens.max()}, Min token value: {tokens.min()}")
    print(f"Total number of tokens in this shard: {len(tokens)}\n")

# Main function to process any input file
def process_data(input_file, dataset_name):
    shard_size = int(1e5)  # Set shard size to 100,000 tokens per shard
    local_dir = f"{dataset_name}_dataset"
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir + "/shards")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Load the content of the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        doc_text = f.read()

    # Split text into paragraphs or sections to simulate multiple documents
    docs = doc_text.split("\n\n")

    nprocs = max(1, os.cpu_count() // 2)  # Adjust process count based on available CPU cores
    nprocs = 1  # You can increase this for multiprocessing if needed

    total_tokens_processed = 0  # To track the total number of tokens

    # Initialize multiprocessing pool
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}", position=0, leave=False)

        # Tokenize each document and write it into shards
        for tokens in pool.imap(tokenize, docs, chunksize=16):
            total_tokens_processed += len(tokens)  # Accumulate total tokens
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                progress_bar.update(len(tokens))
            else:
                # Assign first shard to 'val', the rest to 'train'
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{split}_shard_{dataset_name}_{shard_index:06d}.npy")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]

                # Write the shard
                write_datafile(filename, all_tokens_np)

                # Inspect the shard for issues
                inspect_tokens(all_tokens_np, shard_index)

                # Move to the next shard
                shard_index += 1
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

                # Reset progress bar for the next shard
                progress_bar.close()
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}", position=0, leave=False)

        # Write remaining tokens to the final shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{split}_shard_{dataset_name}_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])

            # Inspect the final shard for issues
            inspect_tokens(all_tokens_np[:token_count], shard_index)

        # Close the final progress bar
        progress_bar.close()

    # Print total number of tokens processed
    print(f"\nSharding complete. {shard_index + 1} shards created.")
    print(f"Total number of tokens processed: {total_tokens_processed}")

# Use the if __name__ == '__main__': guard
if __name__ == '__main__':
    mp.freeze_support()  # Necessary on Windows when using multiprocessing

    # Argument parser to accept dynamic inputs
    parser = argparse.ArgumentParser(description="Process different input datasets and tokenize them.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., inputtxt, fineweb, vulkan)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file (e.g., vulkan_spec.txt)")

    args = parser.parse_args()
    
    # Call process_data function with appropriate input file and dataset name
    process_data(args.input_file, args.dataset)
