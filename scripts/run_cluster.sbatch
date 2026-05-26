#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=10 # nber of env = #cpus - 2 (main process + OS)
#SBATCH --ntasks=1
#SBATCH --mem=64G # 4GB /env
##SBATCH --partition 3090
##SBATCH --partition P100
#SBATCH --gpus 1 # SQAC uses only one GPU for a MLP

#SBATCH --acctg-freq=task=10
#SBATCH --job-name=test
#SBATCH --error=logs/%j.out
#SBATCH --out=logs/%j.out


# HELP
# == Connect ==
# ssh gpu

# == Interactive session ==
# srun --partition=P100 --gres=gpu:1 --mem=16G --cpus-per-task=10 --pty bash
# module load cuda/12.9
# uv sync
# uv run python main.py

#== Run jobs ==
# sbatch scripts/run_cluster.sbatch # copy the job number
# tail -f logs/831118.out
# squeue -j 831118
# rsync -vr DataChallenge2026.zip name@xxx:/home/infres/cmergny-25/DataChallenge2026/data

echo "=== JOB START ==="

cd ~/DataChallenge2026
module load cuda/12.9
uv sync
# source ~/.bashrc
source .venv/bin/activate

uv run python main.py


echo "=== JOB END ==="
echo "timestamp: $(date -Is)"
