#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8 # nber of env = #cpus - 2 (main process + OS)
#SBATCH --ntasks=1
#SBATCH --mem=64G 
##SBATCH --partition 3090
#SBATCH --partition P100
#SBATCH --gpus 1

#SBATCH --acctg-freq=task=10
#SBATCH --job-name=test
#SBATCH --error=logs/%j.out
#SBATCH --output=logs/%j.out


# HELP
# == Connect ==
# ssh gpu

# == Interactive session ==
# srun --partition=P100 --gres=gpu:1 --mem=64G --cpus-per-task=10 --pty bash
# module load cuda/12.9
# uv sync
# uv run python main.py

#== Run jobs ==
# sbatch scripts/run_cluster.sbatch # copy the job number
# tail -f logs/836907.log
# squeue -j 836898
# squeue -p 836898
# cat logs/836898.log

# rsync -vr DataChallenge2026.zip name@xxx:/home/infres/cmergny-25/DataChallenge2026/data

# == Debug ==
# srun --partition=P100 --gres=gpu:1 --mem=64G --cpus-per-task=10 --pty bash
# git pull
# rm -fr .venv
# rm .python-version
# rm uv.lock
# module load cuda/12.9
# uv sync
# launch interactive session
# scancel 834049 -> arrêter le job
# sacct --format=JobId,Node,MaxRSS,AveCPU,State -u wjan-25 --state=RUNNING,PENDING -> voir tous les jobs actifs


echo "=== JOB START ==="

cd ~/DataChallenge2026
uv sync
# source ~/.bashrc
source .venv/bin/activate

echo "args: $*"
if [ $# -eq 0 ]; then
  python main.py
else
  python main.py --config "$@"
fi

echo "=== JOB END ==="
echo "timestamp: $(date -Is)"
