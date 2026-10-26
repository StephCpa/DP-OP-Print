#!/usr/bin/env bash
# Runs every toy arm used in docs/results (about 30 min on 4 CPU threads; do not run two jobs at once).
set -euo pipefail
cd "$(dirname "$0")/../.."
P=experiments/toy_opsd/toy_opsd.py
# Main configuration (TOY_DQ=64, TOY_RANK=16) -> experiments/toy_opsd/runs/
python3 $P pretrain
python3 $P theta0
# lr 3e-3: calibrated so the clip-off arm matches OP-Print's AUC level (a toy calibration on attack AUC that the real study forbids)
for c in none 0.2 0.05 0.01 1e-6; do python3 $P run --clip $c --lr 3e-3; done
for c in none 0.05; do python3 $P run --clip $c --lr 3e-3 --think 0; done                 # no template token at t = 1
# lr 3e-4: the smallest lr meeting the design's f_rm in [0.2, 0.5] rule at clip 0.05
for c in none 0.05; do python3 $P run --clip $c --lr 3e-4 --suffix _lr3e-4; done
# Low-capacity configuration (TOY_DQ=16, TOY_RANK=4) at its f_rm-rule lr -> experiments/toy_opsd/runs_lowcap/
export TOY_DQ=16 TOY_RANK=4 TOY_RUNS=experiments/toy_opsd/runs_lowcap
python3 $P pretrain
python3 $P theta0
python3 $P run --clip none --lr 3e-4
