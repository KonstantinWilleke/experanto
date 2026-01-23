# experanto

A Python package to interpolate recordings and stimuli of neuroscience experiments and build model dataloaders.

## Quickstart

We recommend installing the Python environment using `uv`. This will create a local virtual environment (`.venv`) with the necessary dependencies.

Within this directory, just run `uv sync`.

See the [inspect_data.ipynb](inspect_data.ipynb) notebook for a demonstration of how experanto builds dataloaders from experimental session data, including key features and current limitations.

## Project structure

```
experanto/
├── experanto/
│   ├── __init__.py
│   ├── configs.py
│   ├── dataloaders.py
│   ├── datasets.py
│   ├── experiment.py
│   ├── interpolators.py
│   ├── intervals.py
│   ├── utils.py
├── configs/
│   ├── default.yaml
│   ├── throughput_f16_prenorm.yaml
├── scripts/
│   ├── run_distributed_throughput.py
├── tests/
│   ├── ...
├── inspect_data.ipynb
├── pyproject.toml
├── README.md
└── LICENSE
```

## Run distributed dataloader benchmark

- Simply run the run_distributed_throughput.py script in ./scripts
- Refer to the benchmarking.yaml to override default arguments

`torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=29400 distributed_throughput.py datapath.root="/path/to/data/"`

- Make sure to specify the directory where the example datasets sit in, i.e.:
  - `datapath.root="/path/to/data/"` 
- Override other arguments as needed, e.g.:
  - `dataloader.batch_size=1`
  - `dataloader.pin_memory=True`

### Example output
```
[2025-12-17 21:37:46,130][__main__][INFO] - ===== Distributed Performance Summary =====
[2025-12-17 21:37:46,130][__main__][INFO] - Average throughput across all ranks: 2009.02 frames/second
[2025-12-17 21:37:46,130][__main__][INFO] - Min throughput: 1969.14, Max throughput: 2092.32
[2025-12-17 21:37:46,130][__main__][INFO] - Throughput imbalance: 6.13%
[2025-12-17 21:37:46,153][__main__][INFO] - Rank 0: Waiting for all processes to complete before cleanup...
[2025-12-17 21:37:47,154][__main__][INFO] - All processes completed. Starting cleanup...
```
