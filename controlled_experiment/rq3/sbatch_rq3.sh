#!/bin/bash
#SBATCH --job-name=rq3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

params=$*

source .env

export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

python -u controlled_experiment/rq3/atomic_rq3.py $params