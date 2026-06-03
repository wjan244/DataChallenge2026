#!/bin/bash

#SBATCH --time=25:00:00
#SBATCH --cpus-per-task=10 # nber of env = #cpus - 2 (main process + OS)
#SBATCH --ntasks=1
#SBATCH --mem=64G 
##SBATCH --partition 3090
#SBATCH --partition P100
#SBATCH --gpus 1

#SBATCH --acctg-freq=task=10
#SBATCH --job-name=test
#SBATCH --error=logs/%j.err
#SBATCH --output=logs/%j.log


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
# tail -f logs/ 839760.log
# squeue -j 837419
# squeue -p 836898
# cat logs/838673.log

# rsync -vr DataChallenge2026.zip name@xxx:/home/infres/cmergny-25/DataChallenge2026/data

# == Debug ==
# srun --partition=P100 --gres=gpu:1 --mem=64G --cpus-per-task=10 --pty bash
# git pull
# rm -fr .venv
# rm .python-version
# rm uv.lock
# module load cuda/12.9
# uv sync
# instercative session

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



# scancel 834049 -> arrêter je job
# sacct --format=JobId,Node,MaxRSS,AveCPU,State -u wjan-25 --state=RUNNING,PENDING -> voir tous les jobs actifs
# sacct -u wjan-25 --format="JobID,JobName,NodeList,Elapsed,State"