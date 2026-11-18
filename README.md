[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2405.01155)
[![Python versions](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# SynFlowNet with CYP450 Inhibition Rewards

A fork of [SynFlowNet](https://github.com/mirunacrt/synflownet) that generates molecules binding
soluble epoxide hydrolase (sEH) while avoiding cytochrome P450 inhibition.

**Primer**

SynFlowNet is a GFlowNet model that generates molecules from chemical reactions and available
building blocks. This repo contains instructions for how to train SynFlowNet and sample
synthesisable molecules with probability proportional to a reward specified by the user. The code
builds upon the [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) codebase,
available under the [MIT](https://github.com/recursionpharma/gflownet/blob/trunk/LICENSE) license.
For a primer and repo overview visit
[recursionpharma/gflownet](https://github.com/recursionpharma/gflownet).

## What this fork adds

Optimising sEH binding alone produces molecules that are reliably liable: about 69% of them inhibit
CYP1A2 and 34% inhibit CYP3A4

This fork adds a multi-objective reward that penalises predicted CYP inhibition alongside sEH
affinity, and the machinery to train, sample and evaluate it:

- **Five CYP inhibition proxies** (`CYP/`) — one GCN classifier per isoform (1A2, 2C19, 2C9, 2D6,
  3A4), trained on the TDC Veith benchmarks with a scaffold split. The datasets and checkpoints are
  not committed, so building them is the first step after installing (see
  [Training the CYP models](#training-the-cyp-models)).
- **Four new reward functions** in `src/synflownet/tasks/reactions_task_cyp.py` — a
  smoothed sEH × CYP-safety product with a tunable penalty strength, a single-isoform variant, and
  a preference-conditioned form that spans the whole trade-off with one trained model.
- **A scoring and reporting pipeline** (`score_all.py`, `report.py`) that turns a generation run
  into one CSV carrying the sEH score, all five isoform probabilities, and standard drug-likeness
  descriptors.

The headline result: at matched diversity, the smoothed reward raises the fraction of molecules
clean on all five isoforms from **22% to 60–91%** while keeping mean sEH within 0.04 of the
unconstrained baseline.

## Installation

### PIP

This package is installable as a PIP package, but since it depends on some torch-geometric package
wheels, the `--find-links` arguments must be specified as well:

```bash
conda create --name sfn python=3.10.14
conda activate sfn
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

Or for CPU use:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html
```

The CYP code adds no dependencies of its own — `scikit-learn`, `rdkit`, `scipy` and `PyTDC` all
arrive with the base install.

One extra step is required, because `CYP/` is a directory of scripts rather than a package:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/CYP
```

Without it, training fails with `ModuleNotFoundError: predict`.

## Getting started

### Data

The training relies on two data sources: reaction templates and building blocks. Filenames are
specified in the `ReactionTaskConfig`. The model uses pre-computed masks to ensure compatibility
between the building blocks and the reaction templates. Instructions for preprocessing building
blocks and for computing masks can be found in
[synflownet/data/building_blocks](src/synflownet/data/building_blocks).

## Training the CYP models

The proxies are trained once and then reused as a frozen reward term.

```bash
cd CYP
python data.py              # downloads the five TDC Veith benchmarks -> data/*.tab
python process_data.py      # cleans them                             -> csv/*_veith.csv
```

Then train one model per isoform:

```bash
python train_cyp1a2.py
python train_cyp2c19.py
python train_cyp2c9.py
python train_cyp2d6.py
python train_cyp3a4.py
```

Each writes `models/<isoform>_gcn.pt`. The architecture is a 3-layer GCN
(`GCNConv → BatchNorm → ReLU → Dropout`) with hidden size 128 and mean pooling, 38,913 parameters,
on 33-dimensional atom features. Training uses AdamW, a 70/20/10 **scaffold** split, and selects
the epoch with the best validation AUPRC.

> **A hyperparameter sweep has already been run** — 40 configurations per isoform over hidden size,
> dropout and weight decay. The winning values are baked into each `train_cyp*.py`, and the full
> results are in `CYP/sweep_<isoform>.json`. You do not need to run `sweep.py`.

## Reward functions

All CYP rewards multiply the sEH binding proxy by a safety term computed from the **CYP1A2 and
CYP3A4** proxies only. The other three isoforms are never optimised against — they are held out,
and used at evaluation time to check whether the policy generalises.

| `--reward`        | Formula             | Notes                                                |
| ----------------- | ------------------- | ---------------------------------------------------- |
| `seh_reaction`    | sEH                 | unconstrained baseline                               |
| `seh_cyp1a2`      | sEH · (1 − p₁ₐ₂)    | single isoform                                       |
| `seh_cyp1a2_3a4`  | sEH · (1 − s · p̄)   | **the main reward**, `s` set by `--scale`            |
| `seh_cyp_product` | sEH · (1 − p̄)       | the above at `s` = 1                                 |
| `seh_cyp_pref`    | sEH^w₁ · (1 − p̄)^w₂ | preference-conditioned, one model spans the frontier |

where p̄ = mean(p₁ₐ₂, p₃ₐ₄).

### The smoothing parameter

`--scale` sets how hard the penalty bites: the safety term spans `[1 − s, 1]`, so `s` = 1 can
annihilate the reward entirely while `s` = 0.25 costs at most a quarter of it.

**Penalising two isoforms reduces inhibition of all five.** CYP2C19, CYP2C9 and CYP2D6 fall along
with the two that are optimised, because the penalty pushes the policy toward less lipophilic
chemotypes (mean logP drops from 3.7 to 1.7 as `s` rises). This is generalisation, not reward
hacking — those three proxies never enter the reward.

Pick `--scale` according to what you want:

- **`0.25`** — the cheapest useful setting. At the same β = 26 it costs 0.023 in mean sEH (0.850
  vs 0.873) and returns 2.6× the clean rate and 1.9× the useful molecules. The best yield of the
  three.
- **`0.50`** — balanced. Three quarters of the output is clean, at a 0.04 cost in mean sEH.
- **`0.75`** — clean-heavy. Over 90% clean, but mean sEH drops to 0.79 and the molecules become
  noticeably less lipophilic.

Going to `1.00` buys almost no extra cleanliness over `0.75` and costs a further 0.02 sEH, so the
raw product is not recommended.

The β for each model was found by first training a temperature-conditioned policy over
`--sample_dist uniform --dist_params 16 32` and then generating at each β in that range with
`--final_beta`, picking the best. Those sweeps are done — use the β listed below.

## Training

Set `PYTHONPATH` first (see Installation). All runs use
`src/synflownet/tasks/reactions_task_cyp.py`.

```bash
# scale 0.25 -- maximum sEH retention
python src/synflownet/tasks/reactions_task_cyp.py \
    --wandb_run_name sfn_cyp_025 \
    --reward seh_cyp1a2_3a4 --scale 0.25 \
    --sample_dist constant --dist_params 26

# scale 0.50 -- balanced
python src/synflownet/tasks/reactions_task_cyp.py \
    --wandb_run_name sfn_cyp_050 \
    --reward seh_cyp1a2_3a4 --scale 0.50 \
    --sample_dist constant --dist_params 20

# scale 0.75 -- clean-heavy
python src/synflownet/tasks/reactions_task_cyp.py \
    --wandb_run_name sfn_cyp_075 \
    --reward seh_cyp1a2_3a4 --scale 0.75 \
    --sample_dist constant --dist_params 18

# preference-conditioned -- one model for the whole trade-off
python src/synflownet/tasks/reactions_task_cyp.py \
    --wandb_run_name sfn_cyp_pc \
    --reward seh_cyp_pref \
    --sample_dist constant --dist_params 32
```

Each run trains for 5,000 steps and then generates, writing
`logs/<run name>_<timestamp>/final/generated_objs_*.db`.

## Generating

Every training run ends with a generation pass (`num_final_gen_steps = 100`), so a fresh set of
molecules already exists when training finishes — as SQLite databases in
`logs/<run>/final/generated_objs_*.db`

To draw a further set from an already-trained model, re-invoke the same script with `--backoff`,
which loads the finished checkpoint and skips straight to generation.

`score_all.py` turns a generation directory into one fully scored CSV — RDKit descriptors, the sEH
proxy re-run as its own column, and all five CYP isoform probabilities:

```bash
python score_all.py logs/<run>/final --out Generated_molecules/myrun_scored.csv
```

With no `--out` the file lands in `Generated_molecules/<run directory name>_scored.csv`.

**Smoothed-penalty models** — the model already trains at its chosen β, so generation takes no
extra arguments beyond the scale:

```bash
python src/synflownet/tasks/reactions_task_cyp.py \
    --wandb_run_name sfn_cyp_050 --backoff --scale 0.50 || exit 1
python score_all.py "logs/${RUN}/final" --out Generated_molecules/cyp050_scored.csv
```

**Preference-conditioned model** — sweep ω₁, the weight on sEH, from pure safety to pure affinity:

```bash
for W in 0.0 0.25 0.5 0.75 1.0; do
    python src/synflownet/tasks/reactions_task_cyp.py \
        --wandb_run_name sfn_cyp_pc --backoff --final_pref "$W" || exit 1
    python score_all.py "logs/${RUN}/final" --out "Generated_molecules/pc_w${W}_scored.csv"
done
```

- **Generation overwrites `logs/<run>/final` each time.** Score before moving to the next ω, and
  keep the `|| exit 1` so a failed run does not leave you scoring the previous molecules twice.

## Analysis report of generated molecules

`report.py` summarises one scored CSV, printing the report and saving it as a one-row CSV:

```bash
python report.py Generated_molecules/myrun_scored.csv
```

```
  samples                              6398
  unique                               1581
  scaffolds                            1480
  diversity (1-Tanimoto)             0.8466

  mean sEH                           0.8152
  sEH > 0.875                         27.5%

  inhibits CYP1A2                      5.6%
  inhibits CYP2C19                     6.5%
  inhibits CYP2C9                      2.2%
  inhibits CYP2D6                      2.3%
  inhibits CYP3A4                      3.2%
  clean 0/5                           84.2%
  useful (hot & clean)                  345

  SA                                  4.240
  QED                                 0.674
  logP                                2.155
  Lipinski pass                       0.955
```

Every metric except `samples` is computed over unique canonical SMILES. Generated samples are drawn
in proportion to reward, so a molecule sampled 700 times would otherwise count 700 times and the
quality numbers would mostly be measuring how collapsed the policy is.

# Citation

If you use this code in your research, please cite the following paper:

```
@article{cretu2024synflownet,
      title={SynFlowNet: Design of Diverse and Novel Molecules with Synthesis Constraints},
      author={Miruna Cretu, Charles Harris, Ilia Igashov, Arne Schneuing, Marwin Segler, Bruno Correia, Julien Roy, Emmanuel Bengio and Pietro Liò},
      journal={arXiv preprint arXiv},
      year={2024}
}
```

The CYP inhibition data comes from the Therapeutics Data Commons ADME benchmarks, originally from:

```
@article{veith2009comprehensive,
      title={Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries},
      author={Veith, Henrike and Southall, Noel and Huang, Ruili and James, Tim and Fayne, Darren and Artemenko, Natalia and Shen, Min and Inglese, James and Austin, Christopher P and Lloyd, David G and Auld, Douglas S},
      journal={Nature Biotechnology},
      volume={27}, number={11}, pages={1050--1055},
      year={2009}
}
```
