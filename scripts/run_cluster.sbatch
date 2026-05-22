#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=10 # nber of env = #cpus - 2 (main process + OS)
#SBATCH --ntasks=1
#SBATCH --mem=40g # 4GB /env
##SBATCH --partition 3090
##SBATCH --partition P100
#SBATCH --gpus 1 # SQAC uses only one GPU for a MLP

#SBATCH --acctg-freq=task=10
#SBATCH --job-name=sac
#SBATCH --error=logs/%j.out
#SBATCH --out=logs/%j.out


# HELP
# sinteractive --gpus 1 --mem 40 --cpus-per-task 10 --time 10:00:00 --partition 3090

echo "=== JOB START ==="

cd ~/DataChallenge2026
uv sync
# source ~/.bashrc
source .venv/bin/activate

python main.py


echo "=== JOB END ==="
echo "timestamp: $(date -Is)"
