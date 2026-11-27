#!/bin/bash
# Template GPU job for a Grid-Engine cluster (e.g. TSUBAME 4.0). Submit with:
#   qsub -g <your-group> -l gpu_1=1 -l h_rt=12:00:00 -cwd -N li3ps4_md hpc/run_md_job.sh
# Adjust module / environment lines for your site.
set -euo pipefail
# module load cuda
# source ~/venvs/li3ps4/bin/activate
python scripts/run_md.py --temps 600 700 800 900 1000 --steps 50000 --device cuda --out runs/prod
python scripts/analyze.py runs/prod
