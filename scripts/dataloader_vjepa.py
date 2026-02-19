#!/usr/bin/env python
"""
Benchmark VJEPA embedding dataloader throughput.

Usage:
    python benchmark_embedding_dataloader.py \
        --embedding_dir /mnt/filesystem-m3 \
        --layer 8 \
        --batch_size 256 \
        --num_workers 8 \
        --num_batches 1000
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import argparse
import time
from tqdm import tqdm


class VJEPAEmbeddingDataset(Dataset):
    """
    Simple dataset for loading saved VJEPA embeddings.
    
    Args:
        embedding_dir: Path to directory containing layer_X folders
        layer_idx: Which layer to load (e.g., 8, 16, 24)
        rank: Which rank's data to load (None = all ranks)
    """
    def __init__(self, embedding_dir, layer_idx, rank=None):
        self.embedding_dir = Path(embedding_dir)
        self.layer_dir = self.embedding_dir / f"layer_{layer_idx}"
        
        if not self.layer_dir.exists():
            raise ValueError(f"Layer directory not found: {self.layer_dir}")
        
        # Load metadata from all ranks or specific rank
        self.metadata = []
        if rank is not None:
            # Load specific rank
            metadata_path = self.layer_dir / f"rank{rank}_metadata.csv"
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                self.metadata.append(df)
        else:
            # Load all ranks
            for metadata_path in sorted(self.layer_dir.glob("rank*_metadata.csv")):
                df = pd.read_csv(metadata_path)
                self.metadata.append(df)
        
        if not self.metadata:
            raise ValueError(f"No metadata files found in {self.layer_dir}")
        
        # Concatenate all metadata
        self.metadata = pd.concat(self.metadata, ignore_index=True)
        
        print(f"Loaded {len(self.metadata)} embeddings from layer {layer_idx}")
        print(f"Dataset keys: {self.metadata['dataset_key'].unique()}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load embedding
        embedding_path = self.layer_dir / row['embedding_filename']
        embedding = torch.load(embedding_path)
        
        # Return embedding and metadata
        return {
            'embedding': embedding,
            'batch_idx': row['batch_idx'],
            'sample_idx': row['sample_idx'],
            'token_idx': row['token_idx'],
            'dataset_key': row['dataset_key'],
        }


def benchmark_dataloader(dataloader, num_batches=None, log_every=10):
    """
    Benchmark dataloader throughput with detailed statistics.
    
    Args:
        dataloader: PyTorch DataLoader to benchmark
        num_batches: Maximum number of batches to process (None = all)
        log_every: Report throughput every N seconds
    
    Returns:
        dict: Statistics about the benchmark run
    """
    print("\n" + "="*70)
    print("Starting dataloader benchmark...")
    print("="*70)
    
    # Performance tracking
    start_time = time.time()
    last_report_time = start_time
    batches_since_last_report = 0
    total_batches_processed = 0
    total_embeddings_loaded = 0
    
    # Track embedding dimensions from first batch
    embedding_shape = None
    
    max_iter = num_batches if num_batches else len(dataloader)
    
    with tqdm(total=max_iter, desc="Processing batches") as pbar:
        for i, batch in enumerate(dataloader):
            if num_batches and i >= num_batches:
                break
            
            # Get embedding shape from first batch
            if embedding_shape is None:
                embedding_shape = batch['embedding'].shape
                print(f"\nEmbedding shape per batch: {embedding_shape}")
                print(f"  Batch size: {embedding_shape[0]}")
                print(f"  Embedding dim: {embedding_shape[1]}")
            
            batch_size = batch['embedding'].shape[0]
            total_embeddings_loaded += batch_size
            total_batches_processed += 1
            batches_since_last_report += 1
            
            # Report throughput periodically
            current_time = time.time()
            time_since_last_report = current_time - last_report_time
            
            if time_since_last_report >= log_every:
                throughput = batches_since_last_report / time_since_last_report
                embeddings_per_sec = (batches_since_last_report * batch_size) / time_since_last_report
                
                tqdm.write(
                    f"Throughput: {throughput:.2f} batches/sec | "
                    f"{embeddings_per_sec:.1f} embeddings/sec | "
                    f"(over last {time_since_last_report:.2f}s)"
                )
                
                batches_since_last_report = 0
                last_report_time = current_time
            
            pbar.update(1)
    
    # Final statistics
    total_time = time.time() - start_time
    avg_batch_throughput = total_batches_processed / total_time
    avg_embedding_throughput = total_embeddings_loaded / total_time
    
    print("\n" + "="*70)
    print("Benchmark Complete!")
    print("="*70)
    print(f"Total batches processed: {total_batches_processed}")
    print(f"Total embeddings loaded: {total_embeddings_loaded:,}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average throughput: {avg_batch_throughput:.2f} batches/sec")
    print(f"Average throughput: {avg_embedding_throughput:.1f} embeddings/sec")
    print(f"Time per batch: {(total_time/total_batches_processed)*1000:.2f} ms")
    print(f"Time per embedding: {(total_time/total_embeddings_loaded)*1000:.3f} ms")
    
    # Calculate data transfer stats
    if embedding_shape is not None:
        embedding_size_bytes = embedding_shape[1] * 4  # float32 = 4 bytes
        total_data_mb = (total_embeddings_loaded * embedding_size_bytes) / (1024**2)
        throughput_mbps = total_data_mb / total_time
        
        print(f"\nData transfer:")
        print(f"  Per embedding: {embedding_size_bytes/1024:.2f} KB")
        print(f"  Total data: {total_data_mb:.2f} MB")
        print(f"  Throughput: {throughput_mbps:.2f} MB/sec")
    
    print("="*70 + "\n")
    
    return {
        'total_batches': total_batches_processed,
        'total_embeddings': total_embeddings_loaded,
        'total_time': total_time,
        'batch_throughput': avg_batch_throughput,
        'embedding_throughput': avg_embedding_throughput,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark VJEPA embedding dataloader')
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help='Directory containing saved embeddings')
    parser.add_argument('--layer', type=int, required=True,
                        help='Layer index to load (e.g., 8, 16, 24)')
    parser.add_argument('--rank', type=int, default=None,
                        help='Specific rank to load (None = all ranks)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for dataloader')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Maximum number of batches to process (None = all)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the dataset')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pinned memory for faster GPU transfer')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Report throughput every N seconds')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch per worker')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VJEPA Embedding Dataloader Benchmark")
    print("="*70)
    print(f"Configuration:")
    print(f"  Embedding dir: {args.embedding_dir}")
    print(f"  Layer: {args.layer}")
    print(f"  Rank filter: {args.rank if args.rank is not None else 'all'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Shuffle: {args.shuffle}")
    print(f"  Pin memory: {args.pin_memory}")
    print(f"  Prefetch factor: {args.prefetch_factor}")
    print(f"  Max batches: {args.num_batches if args.num_batches else 'all'}")
    print("="*70)
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = VJEPAEmbeddingDataset(
        embedding_dir=args.embedding_dir,
        layer_idx=args.layer,
        rank=args.rank
    )
    
    # Create dataloader
    print(f"\nCreating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    
    print(f"Dataset size: {len(dataset):,} embeddings")
    print(f"Number of batches: {len(dataloader):,}")
    
    # Run benchmark
    stats = benchmark_dataloader(
        dataloader,
        num_batches=args.num_batches,
        log_every=args.log_every
    )
    
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()