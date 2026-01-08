import itertools
import multiprocessing as mp

import yaml
from spiral import Spiral
import pyarrow as pa
import numpy as np

from pathlib import Path
import json

sp = Spiral()
project = sp.project("external-805943")
screens_table = project.table("screens")

# Loads experiment data from a directory.
experiment_data_path = Path("35_3832003544063_1_V1")

with open(experiment_data_path / "meta.json", "r") as f:
    global_meta = json.load(f)

timestamps = np.load(experiment_data_path / "screen" / "timestamps.npy")
screen_data_meta = []
for i, frame_meta_path in enumerate(sorted((experiment_data_path / "screen" / "meta").glob("*.yml"))):
    with open(frame_meta_path, "r") as f:
        screen_data_meta.append(yaml.safe_load(f))

def constant_array(value, length, pa_type):
    """
    Create a constant array in PyArrow with optimal memory efficiency.

    Parameters:
    -----------
    value : any
        The constant value to repeat
    length : int
        The length of the array
    pa_type : pa.DataType
        The PyArrow data type

    Returns:
    --------
    pa.Array
        A constant array with the specified value repeated
    """
    # Create a scalar with the specified type
    scalar = pa.scalar(value, type=pa_type)

    # Use repeat to create a constant array efficiently
    # This uses run-length encoding internally for maximum efficiency
    return pa.repeat(scalar, length)

def process_batch(batch_data):
    """Process a single batch and return transaction ops."""
    batch, data_key = batch_data

    # Create a new transaction for this worker
    worker_tx = screens_table.txn()

    screen_time = []
    screen_data = []
    for i, frame_path in batch:
        # NOTE(marko): This is weird, but some frame data is missing...
        #   so we can't just take screen_data_meta[i].
        index = int(str(frame_path).split("/")[-1].strip(".npy"))
        meta = screen_data_meta[index]
        timestamps_offset = meta["first_frame_idx"]
        data = np.load(frame_path)
        assert data.shape[0] == meta["num_frames"], f"Data shape {data.shape} does not match metadata {meta}"
        for j in range(meta["num_frames"]):
            screen_time.append(timestamps[timestamps_offset + j])
            screen_data.append(data[0, :].tobytes(order='C'))

    worker_tx.write({
        "data_key": constant_array(data_key, len(screen_time), pa.string()),
        "time": pa.array(screen_time, type=pa.float32()),
        "frame": pa.array(screen_data, type=pa.large_binary()),
    })

    # Take operations aborting the transaction. We want all workers to atomically commit.
    return worker_tx.take()

def main():
    # Create root transaction
    tx = screens_table.txn()

    # Prepare batches
    batch_size = 100
    batches = list(itertools.batched(enumerate(sorted((experiment_data_path / "screen" / "data").glob("*.npy"))), batch_size))

    # Prepare batch data for workers
    batch_data_list = [
        (batch, global_meta["data_key"])
        for batch in batches
    ]

    # Process batches with multiprocessing
    with mp.Pool(processes=4) as pool:
        all_ops = pool.map(process_batch, batch_data_list)

    # Add all ops to the root transaction
    for ops in all_ops:
        tx.include(ops)

    # Commit the root transaction
    tx.commit(compact=True)

if __name__ == "__main__":
    main()