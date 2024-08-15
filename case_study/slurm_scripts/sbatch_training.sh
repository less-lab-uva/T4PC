#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --time=25:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

module load cuda/11.4.2

source .env

export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/PythonAPI
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/

cd $1
shift 1
params=$*

echo "Training..."
python -u ./case_study/TCP/train.py --gpus 2 --batch_size 256 $params
