# block_assign (block assignment MIP instance)

This module generates a block-assignment MIP instance (pairwise co-enrollment driven).

## Prerequisites

* Python 3.7+
* `pandas`
* `numpy`
* `gurobipy` (with a valid license)
* Place required CSVs in the repository root `inputs/` directory

## Installation

```bash
# (Optionally) create a virtual env
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install pandas numpy gurobipy
```

## Usage

```bash
# Basic: schedule 20 courses
python block_assign/block_assign_generator.py 20

# With custom RNG seed
python block_assign/block_assign_generator.py 20 --seed 42
```

This will:

1. Read `inputs/anon_coenrol.csv` & `inputs/hist_totals.csv`.
2. Sample pairwise co-enrollments for the top-20 (or random extra) course IDs.
3. Build and write the LP model to `outputs/blockassign_n20_seed42.lp` (with your chosen seed).
4. Print a stats summary, e.g.:

```
Done. LP ⇒ outputs/blockassign_n20_seed42.lp, stats: {
  "samples": 80,
  "weighted": 56,
  "random": 24,
  "unique": 78,
  "max": 5
}
```

---

## Runtime Benchmark Table

Use the table below to record how long the block assignment generator takes for different `size` values. You can measure runtime in bash, e.g.:

```bash
time python block_assign/block_assign_generator.py 50
```

| Size  | Runtime (s) | 
| :--:  | :---------: | 
|  200  |    438      |     
|  300  |    523      |  
|  400  |    2207         |  
|  500  |    10829       |   
|  600  |    32256         |   




## Files

Structure:

block_assign/
- outputs/
- block_assign_generator.py

Note: `setup.py` and its data-generation pipeline were removed for open-sourcing. Bring your own anonymized CSVs listed above.

# block_seq (block sequencing MIP instance)

Run the simulation & optimization in one command:

```bash
python block_seq/block_seq_generator.py <size> [--seed SEED] [--slots SLOTS] [--slow BOOL]
```

* `<size>`: Number of (anonymous) courses to simulate (e.g. `300`). Increasing this increases the model size polynomially, so don't go above 1000 unless you have a lot of RAM.
* `--seed SEED`: Random seed for reproducibility (default: `3`).
* `--slots SLOTS`: Number of time slots (blocks) to use (default: `24`). This must be more than 24.
* `--slow BOOL`: Whether to include 4-block sequence variables (much slower). When True, filenames include `_slow1`; otherwise `_slow0`. Timelimit is 10 hours.

**Example:**

```bash
python block_seq/block_seq_generator.py 300 --seed 42 --slots 24 --slow True
```

Output LP is named `outputs/blockseq_n<size}_seed{seed}_slow{0|1}.lp`.

Structure:

block_seq/
- outputs/
- block_seq_generator.py

Root-level inputs/ directory (shared by both modules):

inputs/
- anon_coenrol.csv
- hist_totals.csv
- anon24.csv
- anon_t_co.csv

## Citation
If you use this repository, please cite the following paper:

Cornell University Uses Integer Programming to Optimize Final Exam Scheduling. Authors: Tinghan Ye, Adam Jovine, Willem van Osselaer, Qihan Zhu, David Shmoys. arXiv: [arXiv:2409.04959](https://arxiv.org/abs/2409.04959).

## Benchmark Table
These are runs on my computer using gurobipy and python 3.11. 

| Size | Slots |   Runtime (s)  |
| ---- | ----- | ----|
| 200  | 10    |  25  |
| 400  | 10    |  54 |
| 200  | 15    | 10651  |
| 400  | 15    |  5+ hours  |
| 200  | 20    | 5+ hours  |
| 400  | 20    |  5+ hours |




