#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=12 # nber of env = #cpus - 2 (main process + OS)
#SBATCH --ntasks=1
#SBATCH --mem=128G 
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --account p_test_rma

#SBATCH --acctg-freq=task=10
#SBATCH --job-name=amo
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
module load EasyBuild/2025a
module load CUDA/13.1.0
uv sync
# source ~/.bashrc
source .venv/bin/activate
uv pip install "transformers<5.0"
echo "args: $*"

python launch_amo.py

echo "=== JOB END ==="
echo "timestamp: $(date -Is)"
