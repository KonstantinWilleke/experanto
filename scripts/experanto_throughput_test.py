import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
NUM_WORKERS_LIST = [1, 2, 4, 8, 16]
DATAPATH_ROOTS = [
    "/home/vedang/filesystem-g2-local/",
    "/mnt/filesystem-k0/",
]
NUM_PROCESSES = 1
NUM_RUNS = 1  # Number of times to run each configuration
WORKING_DIR = "/home/vedang/work/experanto/scripts"
OUTPUT_PATH = "/home/vedang/work/experanto/benchmarks/throughput_benchmark.png"
TIMEOUT_SECONDS = 1200  # 20 minutes

def run_benchmark(root: str, num_workers: int) -> float | None:
    """Run a single benchmark and return throughput or None if failed."""
    cmd = [
        "uv", "run", "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={NUM_PROCESSES}",
        "--master_port=29400",
        "distributed_throughput.py",
        f"datapath.root={root}",
        f"dataloader.num_workers={num_workers}",
        "distributed.max_batches=500",
        "distributed.enable_cuda=false",
        "distributed.move_to_device=true",
        "datapath.files=[37_3843837605846_0_V3A_V4]"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=WORKING_DIR
        )
        
        output = result.stdout + result.stderr
        match = re.search(r'Average throughput across all ranks: ([\d.]+)', output)
        
        if match:
            return float(match.group(1))
        
        print(f"    ✗ Could not parse throughput from output")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout after {TIMEOUT_SECONDS // 60} minutes")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None

def main():
    results = []
    
    print("Starting throughput benchmarks...")
    print(f"Running each configuration {NUM_RUNS} times")
    print("=" * 60)
    
    for root in DATAPATH_ROOTS:
        for num_workers in NUM_WORKERS_LIST:
            print(f"\nTesting root={root}, num_workers={num_workers}")
            
            throughputs = []
            for run in range(1, NUM_RUNS + 1):
                print(f"  Run {run}/{NUM_RUNS}...", end=" ")
                
                throughput = run_benchmark(root, num_workers)
                
                if throughput is not None:
                    throughputs.append(throughput)
                    print(f"✓ {throughput:.2f} fps")
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
                    'datapath_root': root,
                    'num_workers': num_workers,
                    'throughput_fps': mean_throughput,
                    'stderr': stderr,
                    'num_successful_runs': len(throughputs)
                })
                print(f"  → Mean: {mean_throughput:.2f} ± {stderr:.2f} fps (n={len(throughputs)})")
    
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    print(df)
    
    if df.empty:
        raise RuntimeError("No results collected; check logs above.")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Plot with error bars
    for root in DATAPATH_ROOTS:
        df_subset = df[df['datapath_root'] == root]
        ax.errorbar(
            df_subset['num_workers'],
            df_subset['throughput_fps'],
            yerr=df_subset['stderr'],
            marker='o',
            markersize=9,
            linewidth=2.2,
            capsize=5,
            capthick=2,
            label=root
        )
    
    # Add value labels on points
    for _, row in df.iterrows():
        ax.text(
            row['num_workers'],
            row['throughput_fps'],
            f"{row['throughput_fps']:.1f}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (frames/second)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Dataloader Throughput vs Number of Workers\n'
        f'(N={NUM_PROCESSES} processes, batch_size=8, {NUM_RUNS} runs per config)',
        fontsize=13,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(NUM_WORKERS_LIST)
    ax.legend(title='Data Path', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    main()