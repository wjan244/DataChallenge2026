#!/bin/bash

#SBATCH --time=20:00:00
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
# tail -f logs/836898.log
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
# instercative session

echo "=== JOB START ==="

cd /home/infres/wjan-25/DataChallenge2026
module load cuda/12.9

# garantir que la racine est dans PYTHONPATH pour les imports src.*
export PYTHONPATH=$(pwd):$PYTHONPATH

# activer l'environnement Python
source .venv/bin/activate

# synchroniser/installer si besoin (uv accessible depuis la venv)
uv sync

# diagnostics rapides avant le run
echo "CWD: $(pwd)"
echo "which python: $(which python)"
python --version || true
python -c "print('PYTHON_OK')" || true

# Extra debug: afficher sys.path et tester l'import de 'src' dans le même interpréteur
DEBUG_LOG=logs/debug_${SLURM_JOB_ID:-manual}.log
echo "Running debug import check -> $DEBUG_LOG"
python - <<'PY'
import sys, pkgutil
print('sys.path (head):', sys.path[:8])
loader = pkgutil.find_loader('src')
print('pkgutil.find_loader("src") ->', loader)
try:
	import src
	print('import src OK, src.__file__ =', getattr(src, '__file__', None))
except Exception as e:
	print('import src FAILED:', repr(e))
PY
tail -n 200 "$DEBUG_LOG" || true

# exécuter une seule fois et capturer la sortie dans un fichier spécifique au job
LOGFILE=logs/uv_${SLURM_JOB_ID:-manual}.log
echo "Running: uv run python main.py -> $LOGFILE"
uv run python main.py > "$LOGFILE" 2>&1
RC=$?
echo "uv run exited with code: $RC"
echo "--- tail of $LOGFILE ---"
tail -n 200 "$LOGFILE" || true

echo "=== JOB END ==="
echo "timestamp: $(date -Is)"

# scancel 834049 -> arrêter je job
# sacct --format=JobId,Node,MaxRSS,AveCPU,State -u wjan-25 --state=RUNNING,PENDING -> voir tous les jobs actifs