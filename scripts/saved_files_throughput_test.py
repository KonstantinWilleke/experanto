#!/usr/bin/env python
"""
Benchmark VJEPA embedding dataloader throughput across different configurations.
Plots tokens/sec (where 1 token = 1 embedding file) vs number of workers for different batch sizes.
"""

import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Benchmark parameters
# BATCH_SIZE_LIST = [2048]
# NUM_WORKERS_LIST = [16]
BATCH_SIZE_LIST = [64, 128, 256, 512, 1024, 2048, 4096]
NUM_WORKERS_LIST = [2, 4, 8, 12, 16]
NUM_RUNS = 1  # Number of times to run each configuration

# Paths and settings
EMBEDDING_DIR = "/mnt/filesystem-m3"
LAYER = 24
NUM_BATCHES = 1000

# System settings
ENABLE_CUDA = True  # Set to True to use GPU with pin_memory
PIN_MEMORY = True   # Use pinned memory for faster GPU transfer
SHUFFLE = False
PREFETCH_FACTOR = 2

# Script paths
WORKING_DIR = "/home/vedang/work/experanto/scripts"
DATALOADER_SCRIPT = "dataloader_vjepa.py"
OUTPUT_DIR = "/home/vedang/work/experanto/benchmarks"
OUTPUT_PLOT = "vjepa_dataloader_tokens_per_sec.png"

TIMEOUT_SECONDS = 1200  # 20 minutes

# Plot styling
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# ============================================================================
# BENCHMARK FUNCTION
# ============================================================================

def run_benchmark(batch_size: int, num_workers: int) -> dict | None:
    """Run a single benchmark and return results or None if failed."""
    cmd = [
        "python",
        DATALOADER_SCRIPT,
        f"--embedding_dir={EMBEDDING_DIR}",
        f"--layer={LAYER}",
        f"--batch_size={batch_size}",
        f"--num_workers={num_workers}",
        f"--num_batches={NUM_BATCHES}",
        f"--prefetch_factor={PREFETCH_FACTOR}",
    ]
    
    if SHUFFLE:
        cmd.append("--shuffle")
    if PIN_MEMORY:
        cmd.append("--pin_memory")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=WORKING_DIR
        )
        
        output = result.stdout + result.stderr
        
        # Parse throughput - looking for "Average throughput: X.X embeddings/sec"
        match = re.search(r'Average throughput:\s+([\d.]+)\s+embeddings/sec', output)
        
        if match:
            tokens_per_sec = float(match.group(1))
            
            return {
                'batch_size': batch_size,
                'num_workers': num_workers,
                'tokens_per_sec': tokens_per_sec,
            }
        
        print(f"    ✗ Could not parse throughput from output")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout after {TIMEOUT_SECONDS // 60} minutes")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    results = []
    
    print("="*70)
    print("VJEPA EMBEDDING DATALOADER BENCHMARK")
    print("="*70)
    print(f"Configuration:")
    print(f"  Embedding dir: {EMBEDDING_DIR}")
    print(f"  Layer: {LAYER}")
    print(f"  Batch sizes: {BATCH_SIZE_LIST}")
    print(f"  Num workers: {NUM_WORKERS_LIST}")
    print(f"  CUDA enabled: {ENABLE_CUDA}")
    print(f"  Pin memory: {PIN_MEMORY}")
    print(f"  Num batches: {NUM_BATCHES}")
    print(f"  Runs per config: {NUM_RUNS}")
    print("="*70)
    
    for batch_size in BATCH_SIZE_LIST:
        for num_workers in NUM_WORKERS_LIST:
            print(f"\nTesting batch_size={batch_size}, num_workers={num_workers}")
            
            throughputs = []
            for run in range(1, NUM_RUNS + 1):
                print(f"  Run {run}/{NUM_RUNS}...", end=" ")
                
                result = run_benchmark(batch_size, num_workers)
                
                if result is not None:
                    throughputs.append(result['tokens_per_sec'])
                    print(f"✓ {result['tokens_per_sec']:.1f} tok/sec")
                else:
                    print("✗ Failed")
            
            if throughputs:
                mean_throughput = sum(throughputs) / len(throughputs)
                # Calculate standard error (std / sqrt(n))
                if len(throughputs) > 1:
                    std = (sum((x - mean_throughput) ** 2 for x in throughputs) / (len(throughputs) - 1)) ** 0.5
                    stderr = std / (len(throughputs) ** 0.5)
                else:
                    stderr = 0
                
                results.append({
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                    'tokens_per_sec': mean_throughput,
                    'stderr': stderr,
                    'num_successful_runs': len(throughputs)
                })
                print(f"  → Mean: {mean_throughput:.1f} ± {stderr:.1f} tok/sec (n={len(throughputs)})")
    
    print("\n" + "="*70)
    print("Results Summary:")
    print("="*70)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    if df.empty:
        raise RuntimeError("No results collected; check logs above.")
    
    # Save results to CSV
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vjepa_dataloader_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    # ========================================================================
    # CREATE PLOT
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors for different batch sizes
    colors = sns.color_palette("husl", len(BATCH_SIZE_LIST))
    
    # Plot with error bars for each batch size
    for i, batch_size in enumerate(BATCH_SIZE_LIST):
        df_subset = df[df['batch_size'] == batch_size]
        ax.errorbar(
            df_subset['num_workers'],
            df_subset['tokens_per_sec'],
            yerr=df_subset['stderr'],
            marker='o',
            markersize=10,
            linewidth=2.5,
            capsize=6,
            capthick=2.5,
            label=f'BS={batch_size}',
            color=colors[i]
        )
    
    # Add value labels on points
    for _, row in df.iterrows():
        ax.text(
            row['num_workers'],
            row['tokens_per_sec'],
            f"{row['tokens_per_sec']:.0f}",
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )
    
    # Find and highlight best configuration
    best_config = df.loc[df['tokens_per_sec'].idxmax()]
    ax.plot(
        best_config['num_workers'],
        best_config['tokens_per_sec'],
        marker='*',
        markersize=20,
        color='red',
        markeredgecolor='black',
        markeredgewidth=2,
        linestyle='none',
        label=f'Best: BS={best_config["batch_size"]}, '
              f'{best_config["num_workers"]}w, '
              f'{best_config["tokens_per_sec"]:.0f} tok/s',
        zorder=10
    )
    
    # Styling
    ax.set_xlabel('Number of Workers', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/second)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'VJEPA Embedding Dataloader: Throughput vs Workers\n'
        f'(Layer {LAYER}, {NUM_BATCHES} batches, '
        f'CUDA={ENABLE_CUDA}, pin_memory={PIN_MEMORY}, '
        f'{NUM_RUNS} runs per config)',
        fontsize=13,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(NUM_WORKERS_LIST)
    ax.legend(title='Batch Size', fontsize=10, loc='best')
    
    # Add efficiency annotation
    if not df.empty:
        # Calculate efficiency (tokens/sec per worker) for best config
        best_efficiency = best_config['tokens_per_sec'] / best_config['num_workers']
        textstr = f'Best efficiency: {best_efficiency:.1f} tok/sec/worker'
        ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / OUTPUT_PLOT
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_path}")
    
    # Show plot
    plt.show()
    
    # ========================================================================
    # PRINT BEST CONFIGURATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70)
    print(f"Batch size: {best_config['batch_size']}")
    print(f"Num workers: {best_config['num_workers']}")
    print(f"Throughput: {best_config['tokens_per_sec']:.1f} tokens/sec")
    print(f"Efficiency: {best_efficiency:.1f} tokens/sec/worker")
    print("="*70)


if __name__ == "__main__":
    main()