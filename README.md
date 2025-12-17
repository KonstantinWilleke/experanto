#### fast-forward pytorch dataloaders for neuroscience 
requirements
- required python >= 3.9
- use with latest nvidia NGC docker image (see dockerfile)
- pytorch >= 2.5.1


[demo notebook](./examples/demo.ipynb)

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
│   ├── utils.py
├── configs/
│   ├── __init__.py
│   ├── default.yaml
├── scripts/
│   ├── run_distributed_dataloaders.py
├── tests/
│   ├── __init__.py
│   ├── create_mock_data.py
│   ├── test_sequence_interpolators.py
├── examples/                        
├── logs/                             
├── Dockerfile
├── docker-compose.yml
├── setup.py                    
├── requirements.txt                  
├── pytest.ini                  
└── README.md    
```

### Run distributed dataloader benchmark
- simply run the run_distributed_dataloaders.py script in ./scripts
- refer to the benchmarking.yaml to override default arguments

`torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=29400 run_distributed_dataloaders.py datapath.root="/path/to/data/"`
- make sure to specify the directory where the example datasets sit in, i.e.:
  - `datapath.root="/path/to/data/"` 
- override other arguments as needed, e.g.:
  - `dataloader.batch_size=1`
  - `dataloader.pin_memory=True`




Example output
```

[2025-12-17 21:37:46,130][__main__][INFO] - ===== Distributed Performance Summary =====
[2025-12-17 21:37:46,130][__main__][INFO] - Average throughput across all ranks: 2009.02 frames/second
[2025-12-17 21:37:46,130][__main__][INFO] - Min throughput: 1969.14, Max throughput: 2092.32
[2025-12-17 21:37:46,130][__main__][INFO] - Throughput imbalance: 6.13%
[2025-12-17 21:37:46,153][__main__][INFO] - Rank 0: Waiting for all processes to complete before cleanup...
[2025-12-17 21:37:47,154][__main__][INFO] - All processes completed. Starting cleanup...```