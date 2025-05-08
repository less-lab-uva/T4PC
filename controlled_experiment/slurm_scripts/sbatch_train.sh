#!/bin/bash
#SBATCH --job-name=t4pc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="a100_80gb"
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

params=$*

echo "Running with the following parameters: $params"
echo "Running job $SLURM_JOB_ID in $HOSTNAME"
echo "$(nvidia-smi)"
date
echo "Number of cpus: $SLURM_JOB_CPUS_PER_NODE"
echo "$(lscpu)"

source .env

export PYTHONPATH=$PYTHONPATH:$ROOT_DIR/
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR/carla-scene-graphs/
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/PythonAPI
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

python -u controlled_experiment/common/train.py $params