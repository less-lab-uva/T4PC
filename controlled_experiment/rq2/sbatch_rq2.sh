#!/bin/bash
#SBATCH --job-name=rq2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

params=$*

source .env

export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

python -u controlled_experiment/rq2/atomic_rq2.py $params