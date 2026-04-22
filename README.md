# Handover Optimization in 5G Networks Using Q-Learning

This repository explores handover management in a simplified 5G setting using Q-learning. It includes:

- a small hand-crafted gridworld for quick experiments
- a dataset-backed environment built from real 5G trace CSV files
- a separate MountainCar benchmark in `qlearning.py`
- a static HTML dashboard for presenting the generated plots

The codebase was originally prepared as part of the EC431 5G Communication and Network course project.

## Project Summary

The main training loop lives in `main.py`. It builds an environment, trains a Q-learning agent, and writes performance plots to disk.

Two environment modes are available:

1. Demo mode: a fixed 3x6 grid with manually defined antenna coverage.
2. Dataset mode: a grid generated from the included 5G production dataset or another compatible dataset root.

## What The Current Implementation Learns

This is the most important detail to understand before running the project:

- The agent now learns two linked policies: movement direction and antenna preference.
- The movement policy uses a Q-table over grid state and action (`up`, `down`, `left`, `right`).
- The antenna policy still learns which serving cell is best for each state.
- The reward function combines goal completion, progress toward the goal, handover cost, invalid-move penalties, and a small bonus for link stability.

So the repository now behaves more like a small joint mobility-and-handover optimizer, while still remaining a simplified educational simulator rather than a radio-accurate 5G system.

## Repository Layout

```text
.
|-- agent.py                  # Q-table setup and antenna-value updates
|-- dataset_loader.py         # Builds a grid environment from 5G CSV traces
|-- gridworld.py              # Grid environment and reward logic
|-- main.py                   # Main training entry point
|-- plotter.py                # Saves training metrics as PNG plots
|-- qlearning.py              # Separate MountainCar Q-learning benchmark
|-- requirements.txt          # Python dependencies
|-- 5G-production-dataset/    # Included dataset used for dataset-backed runs
|-- metrics_plots/            # Default output folder for training plots
|-- dashboard_plots/          # Example plot set used by the dashboard
|-- views/index.html          # Static dashboard page
|-- style/styles.css          # Dashboard styling
`-- qlearning_plots/          # Output folder for the benchmark plot
```

## Environment And Dataset Logic

When `--dataset-root` is provided, `dataset_loader.py` does the following:

- discovers CSV files recursively under the dataset root
- filters files by mobility mode and optional app names
- filters rows by `NetworkMode` when `--network-mode` is supplied
- reads `Latitude`, `Longitude`, and `RAWCELLID` from each row
- bins latitude/longitude into a configurable grid
- keeps the top raw cells observed in each spatial bin
- fills uncovered bins by copying coverage from the nearest observed bin
- derives a start bin from common first positions and a goal bin from common last positions

The loader expects CSVs with columns such as:

- `Latitude`
- `Longitude`
- `RAWCELLID`
- `NetworkMode`

The included dataset already matches this format.

## Requirements

- Python 3.10 or newer
- `numpy`
- `matplotlib`
- `gym`
- `pyglet`

Python 3.10+ is required because the code uses modern type-union syntax such as `str | Path`. The repository was verified locally with Python 3.11.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running The Main Training Script

### Demo Environment

This uses the built-in 3x6 environment and writes plots to `metrics_plots/` by default.

```bash
python main.py
```

A shorter smoke run:

```bash
python main.py --episodes 200 --max-steps 40 --plot-dir metrics_plots/demo_run
```

### Dataset-Backed Environment

This builds the grid from the included dataset:

```bash
python main.py --dataset-root 5G-production-dataset
```

`--dataset-root` can point either to the extracted dataset folder itself or to a parent folder that contains the nested dataset directory.

A more explicit example:

```bash
python main.py --dataset-root 5G-production-dataset --rows 10 --cols 10 --mobility-mode Driving --network-mode 5G --limit-files 10 --episodes 1000 --plot-dir metrics_plots/dataset_run
```

You can also restrict the dataset to specific app folders:

```bash
python main.py --dataset-root 5G-production-dataset --app Netflix --app Download
```

## Main CLI Options

| Option | Purpose |
| --- | --- |
| `--dataset-root` | Use dataset-backed training instead of the built-in demo grid. |
| `--rows`, `--cols` | Grid size used for dataset binning. |
| `--mobility-mode` | File-level filter such as `Driving` or `Static`. |
| `--network-mode` | Repeatable radio-mode filter, for example `--network-mode 5G`. |
| `--app` | Repeatable app-folder filter, for example `--app Netflix`. |
| `--limit-files` | Only use the first `N` matching CSV files. |
| `--top-k-cells` | Maximum number of raw cells kept per spatial bin. |
| `--min-cell-observations` | Minimum observations before a raw cell is kept directly. |
| `--episodes` | Number of training episodes. Default: `6000`. |
| `--lr` | Learning rate. Default: `0.1`. |
| `--gamma` | Discount factor. Default: `0.95`. |
| `--epsilon` | Initial exploration rate. Default: `0.82`. |
| `--min-epsilon` | Floor for exploration after decay. Default: `0.05`. |
| `--epsilon-decay-fraction` | Fraction of training used to decay epsilon toward the minimum. Default: `0.9`. |
| `--max-steps` | Per-episode step cap. |
| `--plot-dir` | Output folder for generated plots. |

## Generated Outputs

`plotter.py` saves these figures into the selected plot directory:

- `Rewards_vs_Episodes_Sampled.png`
- `Handovers_vs_Episodes_Sampled.png`
- `Cumulative_Success.png` (running success ratio, not a raw count)
- `Steps_vs_Episodes_Sampled.png`
- `Cumulative_Rewards_Sampled.png`
- `Histogram_of_Handovers.png`
- `Reward_vs_Handovers_Tradeoff.png`
- `Epsilon_Decay_Sampled.png`

The default output folder is `metrics_plots/`.

## Benchmark Script

`qlearning.py` is a separate Q-learning baseline using `MountainCar-v0`. It is independent from the 5G handover environment and writes its figure to `qlearning_plots/Plot.png`.

Run it with:

```bash
python qlearning.py
```

To override the number of episodes:

PowerShell:

```powershell
$env:QL_EPISODES="1000"
python qlearning.py
```

Bash:

```bash
QL_EPISODES=1000 python qlearning.py
```

Note: the benchmark aggregates reward in 100-episode chunks, so very small runs produce only a sparse plot.

## Dashboard

The repository includes a static dashboard at `views/index.html`. It does not require a web server.

To regenerate the images used by the dashboard:

```bash
python main.py --episodes 1000 --plot-dir dashboard_plots
```

Then open `views/index.html` in a browser.

## Verified Commands

The following commands were smoke-tested locally in this repository:

- `python main.py --episodes 3 --max-steps 8 --plot-dir .codex_tmp_smoke_default`
- `python main.py --dataset-root 5G-production-dataset --episodes 2 --limit-files 1 --max-steps 5 --plot-dir .codex_tmp_smoke_dataset`
- `python qlearning.py` with `QL_EPISODES=10`

## Current Limitations

- The reward signal is still a simplified proxy and does not directly model throughput, latency, RSRP, SINR, or packet loss.
- Movement and handover are learned with tabular Q-learning, so the approach will not scale as smoothly as deep RL methods on larger state spaces.
- Dataset mode uses coarse spatial binning plus nearest-bin fallback, which is useful for experimentation but not a radio-accurate simulator.
- Training logs are very verbose because every step is printed to the console.

## Authors

- Rajat Dhoot
- Vaibhav Joshi
- Aditiya Hingwasiya
- Nitin Somani
