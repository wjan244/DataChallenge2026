#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=10 # nber of env = #cpus - 2 (main process + OS)
#SBATCH --ntasks=1
#SBATCH --mem=40G # 4GB /env
##SBATCH --partition 3090
##SBATCH --partition P100
#SBATCH --gpus 1 # SQAC uses only one GPU for a MLP

#SBATCH --acctg-freq=task=10
#SBATCH --job-name=test
#SBATCH --error=logs/%j.out
#SBATCH --out=logs/%j.out


# HELP
# connect:

# sinteractive --gpus 1 --mem=40G --cpus-per-task 10 --time 10:00:00 --partition 3090
# srun --partition=3090 --gres=gpu:1 --mem=16G --cpus-per-task=10 --pty bash
# sbatch scripts/run_cluster.sbatch
# tail -f logs/831118.out
# squeue -j 831118
# rsync -vr DataChallenge2026.zip name@xxx:/home/infres/cmergny-25/DataChallenge2026/data

echo "=== JOB START ==="

cd ~/DataChallenge2026
uv sync
# source ~/.bashrc
source .venv/bin/activate

python main.py


echo "=== JOB END ==="
echo "timestamp: $(date -Is)"
