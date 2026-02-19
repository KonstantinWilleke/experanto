#!/usr/bin/env python
"""
Distributed VJEPA processing script for PyTorch DDP.
This script processes dataloader batches through VJEPA and optionally saves embeddings.

Usage:
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node=$NUM_PROCESSES \
      run_distributed_vjepa.py \
      --save_embeddings \
      --output_dir ./vjepa_embeddings
"""
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import logging
import time
import datetime
import os
import csv
import os.path as path
from pathlib import Path
import numpy as np
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoVideoProcessor, AutoModel

from experanto.dataloaders import get_multisession_concat_dataloader, LongCycler
from experanto.utils import handle_responses_field, move_data_to_device

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed(enable_cuda=False):
    dist.init_process_group(
        backend='nccl' if enable_cuda else "gloo",
        timeout=datetime.timedelta(minutes=1)
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if enable_cuda:
        torch.cuda.set_device(local_rank)
        logger.info(f"Initialized process group: rank {rank}/{world_size}, local_rank: {local_rank}")

        # NCCL test
        tensor = torch.tensor([rank], dtype=torch.int64, device=f"cuda:{local_rank}")
        gathered = [torch.zeros(1, dtype=torch.int64, device=f"cuda:{local_rank}")
                    for _ in range(world_size)]

        logger.info(f"Rank {rank}: Before barrier")
        dist.barrier()
        logger.info(f"Rank {rank}: After barrier, before all_gather")

        dist.all_gather(gathered, tensor)
        gathered = [t.item() for t in gathered]

        logger.info(f"Rank {rank}: Can see processes {gathered}")
    return rank, local_rank, world_size


class SyncedLongCycler(LongCycler):
    """
    Extends LongCycler to support distributed training by syncing the number of batches
    across all ranks while preserving the cycling behavior for loaders of unequal size.
    """

    def __init__(self, loaders, max_batches):
        super().__init__(loaders)
        self.max_batches = max_batches
        self.iterations_per_epoch = len(self.loaders) * self.max_batches

    def __len__(self):
        return self.iterations_per_epoch


def prepare_synced_dataloaders(dataloaders, rank, world_size):
    """
    Prepares dataloaders for distributed training by synchronizing lengths.
    """
    local_max_length = max([len(loader) for loader in dataloaders.values()])

    all_max_lengths = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(all_max_lengths, torch.tensor([local_max_length], device="cuda"))
    all_max_lengths = [length.item() for length in all_max_lengths]

    global_max_length = max(all_max_lengths)

    if rank == 0:
        logger.info(f"Minimum dataloader lengths across ranks: {all_max_lengths}")
        logger.info(f"Using synchronized length of {global_max_length} batches per key per epoch")

    return SyncedLongCycler(dataloaders, global_max_length)


def process_batch_through_vjepa(batch, model, processor, cfg, device):
    """
    Process a single batch through VJEPA model.
    
    Args:
        batch: Batch from dataloader containing 'inputs' with shape (B, T, C, H, W)
        model: VJEPA model
        processor: VJEPA processor
        cfg: Configuration
        device: Device to run on
    
    Returns:
        dict: Dictionary containing embeddings from requested layers, or None if no layers specified
    """
    # Get vjepa config from correct location
    vjepa_cfg = cfg.distributed.vjepa if 'vjepa' in cfg.distributed else cfg.vjepa
    
    # Extract video data from batch
    video_data = batch.get('inputs', batch.get('screen', None))
    
    if video_data is None:
        raise ValueError("Batch must contain 'inputs' or 'screen' key with video data")
    
    if video_data.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got shape {video_data.shape}")
    
    # Dataloader returns (B, T, C, H, W), VJEPA expects (B, C, T, H, W)
    # print("Before subsampling:", video_data.shape)
    video_data = video_data[:, :, ::3, :, :]
    # print("After subsampling:", video_data.shape)
    video_data = video_data.permute(0, 2, 1, 3, 4)
    
    # Move to device
    video_data = video_data.to(device)
    
    # Prepare inputs for VJEPA2
    # VJEPA2 expects 'pixel_values_videos' not 'pixel_values'
    inputs = {"pixel_values_videos": video_data}
    
    # Check if layers are specified
    layers = vjepa_cfg.get('layers', [])
    
    # Forward pass
    with torch.no_grad():
        if not layers:
            # Normal forward pass without extracting hidden states
            outputs = model(**inputs)
            #Print shape 
            # print("VJEPA output shape:", outputs.last_hidden_state.shape)
            return None  # No embeddings to return
        else:
            # Forward pass with hidden states extraction
            outputs = model(**inputs, output_hidden_states=True)

    # Extract layer embeddings only if layers were specified
    all_hidden_states = outputs.hidden_states
    layer_embeddings = {}
    
    for layer_idx in layers:
        if layer_idx < len(all_hidden_states):
            # Keep raw embeddings: (B, num_tokens, hidden_dim)
            embeddings = all_hidden_states[layer_idx]
            layer_embeddings[f"layer_{layer_idx}"] = embeddings
    
    return layer_embeddings


def save_embeddings_bxt(embeddings, batch_info, output_dir, rank, batch_idx, vjepa_cfg):
    """
    Save embeddings as B×num_tokens individual files per forward pass.
    Model outputs (B, num_tokens, hidden_dim), we save B×num_tokens files each containing (hidden_dim,).
    
    Args:
        embeddings: Dictionary of layer embeddings {layer_X: tensor}
        batch_info: Additional information about the batch (keys, indices, etc.)
        output_dir: Directory to save embeddings
        rank: Process rank
        batch_idx: Global batch index
        vjepa_cfg: VJEPA configuration
    """
    output_dir = Path(output_dir)
    
    # Create layer directories and metadata files if they don't exist
    for layer_name, layer_embeddings in embeddings.items():
        layer_idx = int(layer_name.split('_')[1])
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Open or append to metadata file
        metadata_path = layer_dir / f"rank{rank}_metadata.csv"
        file_exists = metadata_path.exists()
        
        with open(metadata_path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['embedding_filename', 'batch_idx', 'sample_idx', 'token_idx', 'dataset_key', 'shape']
            )
            if not file_exists:
                writer.writeheader()
            
            # layer_embeddings shape: (B, num_tokens, hidden_dim)
            # We want to save B×num_tokens files, each containing (hidden_dim,)
            B, num_tokens, hidden_dim = layer_embeddings.shape
            
            # Save B×num_tokens files
            for b in range(B):
                for token in range(num_tokens):
                    # Extract single token embedding: (hidden_dim,)
                    token_embedding = layer_embeddings[b, token]
                    
                    embedding_filename = f"rank{rank}_batch{batch_idx}_b{b}_token{token}.pt"
                    filepath = layer_dir / embedding_filename
                    torch.save(token_embedding.cpu(), filepath)
                    
                    writer.writerow({
                        'embedding_filename': embedding_filename,
                        'batch_idx': batch_idx,
                        'sample_idx': b,
                        'token_idx': token,
                        'dataset_key': batch_info['key'],
                        'shape': str(tuple(token_embedding.shape))
                    })


def process_dataloader_with_vjepa(
    dataloader, 
    model, 
    processor, 
    cfg, 
    max_batches=None,
    save_embeddings_flag=False,
    output_dir=None,
    log_every=10
):
    """
    Process dataloader batches through VJEPA with optional saving.
    
    Args:
        dataloader: The dataloader to process
        model: VJEPA model
        processor: VJEPA processor
        cfg: Configuration object
        max_batches: Maximum number of batches to process
        save_embeddings_flag: Whether to save embeddings to disk
        output_dir: Directory to save embeddings (required if save_embeddings_flag=True)
        log_every: Interval (in seconds) to log throughput
    
    Returns:
        dict: Statistics about processing
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}" if cfg.distributed.enable_cuda else "cpu"
    
    if save_embeddings_flag and output_dir is None:
        raise ValueError("output_dir must be specified when save_embeddings_flag=True")
    
    # Create output directory if saving (no rank-specific subdirs, all ranks write to same directory structure)
    if save_embeddings_flag:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Performance tracking
    start_time = time.time()
    last_report_time = start_time
    batches_since_last_report = 0
    total_batches_processed = 0
    
    frames_per_batch = cfg.dataloader.batch_size * cfg.dataset.modality_config.screen.chunk_size
    logger.info(f"Rank {rank}: Processing dataloader with {frames_per_batch} frames per batch")
    
    for i, (key, batch) in tqdm(
        enumerate(dataloader),
        total=max_batches if max_batches else len(dataloader),
        disable=rank != 0,
        mininterval=1.0,
        desc=f"Rank {rank}"
    ):
        if max_batches and i >= max_batches:
            break
        
        # Move batch to device
        batch = move_data_to_device(batch, device, dtype=None)
        
        # Process through VJEPA
        try:
            embeddings = process_batch_through_vjepa(
                batch, model, processor, cfg, device
            )
            
            # Save if requested (only if embeddings were extracted)
            if save_embeddings_flag and embeddings is not None:
                batch_info = {
                    'key': key,
                    'batch_idx': i,
                    'shapes': {k: v.shape for k, v in embeddings.items()}
                }
                save_embeddings_bxt(
                    embeddings, 
                    batch_info, 
                    output_dir, 
                    rank, 
                    i,
                    cfg.distributed.vjepa if 'vjepa' in cfg.distributed else cfg.vjepa
                )
            
            total_batches_processed += 1
            batches_since_last_report += 1     
        except Exception as e:
            logger.error(f"Rank {rank}: Error processing batch {i} with key {key}: {str(e)}")
            continue
        
        # Report throughput periodically
        current_time = time.time()
        time_since_last_report = current_time - last_report_time
        
        if time_since_last_report >= log_every:
            # throughput = (batches_since_last_report / time_since_last_report) * frames_per_batch
            throughput = (batches_since_last_report / time_since_last_report) * 8736 #HACK
            logger.info(
                f"Rank {rank}: Throughput: {throughput/1000:.1f}k frames/second "
                f"(over last {time_since_last_report:.2f}s)"
            )
            
            batches_since_last_report = 0
            last_report_time = current_time
    
    # Final statistics
    total_time = time.time() - start_time
    # overall_throughput = (total_batches_processed * frames_per_batch) / total_time
    overall_throughput = (total_batches_processed * 8736) / total_time #HACK
    logger.info(
        f"Rank {rank}: Processed {total_batches_processed} batches in {total_time:.2f}s "
        f"({overall_throughput/1000:.1f}k frames/second)"
    )
    
    # Gather statistics from all ranks
    throughputs = [torch.tensor([0.0], device=device) for _ in range(world_size)]
    dist.all_gather(throughputs, torch.tensor([overall_throughput], device=device))
    
    if rank == 0:
        throughputs = [t.item() for t in throughputs]
        avg_throughput = sum(throughputs) / len(throughputs)
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)
        
        logger.info(f"===== Distributed VJEPA Processing Summary =====")
        logger.info(f"Average throughput: {avg_throughput/1000:.2f}k frames/second")
        logger.info(f"Min throughput: {min_throughput/1000:.2f}k, Max: {max_throughput/1000:.2f}k")
        logger.info(f"Throughput imbalance: {(max_throughput - min_throughput) / avg_throughput * 100:.2f}%")
        
        if save_embeddings_flag:
            logger.info(f"Embeddings saved to: {output_dir}")
    
    return {
        'total_batches': total_batches_processed,
        'total_time': total_time,
        'throughput': overall_throughput
    }


@hydra.main(config_path=f"{root}/configs", config_name="throughput_f16_prenorm", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for distributed VJEPA processing."""
    
    cfg = handle_responses_field(cfg)
    
    # Set up distributed environment
    rank, local_rank, world_size = setup_distributed(enable_cuda=cfg.distributed.enable_cuda)
    
    # Print configuration on rank 0
    if rank == 0:
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    device = f"cuda:{local_rank}" if cfg.distributed.enable_cuda else "cpu"
    
    # Load VJEPA model (on each rank)
    vjepa_cfg = cfg.distributed.vjepa if 'vjepa' in cfg.distributed else cfg.vjepa
    logger.info(f"Rank {rank}: Loading VJEPA model from {vjepa_cfg.hf_repo}")
    model = AutoModel.from_pretrained(vjepa_cfg.hf_repo).to(device)
    processor = AutoVideoProcessor.from_pretrained(vjepa_cfg.hf_repo)
    model.eval()
    logger.info(f"Rank {rank}: VJEPA model loaded successfully")
    
    # Get dataset paths
    full_paths = np.array([os.path.join(cfg.datapath.root, f) for f in cfg.datapath.files])
    
    # Create dataloader
    logger.info(f"Rank {rank}: Creating dataloader")
    train_dl = get_multisession_concat_dataloader(
        full_paths,
        cfg,
    )
    logger.info(f"Rank {rank}: Dataloader created")
    
    # Wait for all processes to sync
    if rank == 0:
        logger.info("Waiting for all processes to sync dataloaders...")
    dist.barrier()
    
    # Process dataloader through VJEPA
    vjepa_cfg = cfg.distributed.vjepa if 'vjepa' in cfg.distributed else cfg.vjepa
    output_dir = vjepa_cfg.get('output_dir', './vjepa_embeddings') if vjepa_cfg.save_embeddings else None
    
    stats = process_dataloader_with_vjepa(
        dataloader=train_dl,
        model=model,
        processor=processor,
        cfg=cfg,
        max_batches=cfg.distributed.get('max_batches', None),
        save_embeddings_flag=vjepa_cfg.save_embeddings,
        output_dir=output_dir,
        log_every=cfg.distributed.get('log_every', 10)
    )
    
    # Clean up
    logger.info(f"Rank {rank}: Waiting for all processes to complete before cleanup...")
    dist.barrier()
    
    time.sleep(1)
    
    if rank == 0:
        logger.info("All processes completed. Starting cleanup...")
    
    try:
        if cfg.distributed.enable_cuda:
            torch.cuda.synchronize()
        
        dist.destroy_process_group()
        logger.info(f"Rank {rank}: Successfully cleaned up process group")
    except Exception as e:
        logger.error(f"Rank {rank}: Error during cleanup: {str(e)}")
    
    logger.info(f"Rank {rank}: Finished processing and cleanup complete")


if __name__ == "__main__":
    main()