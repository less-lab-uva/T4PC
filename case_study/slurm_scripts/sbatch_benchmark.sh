#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

source .env

starting_path=$1
cd $starting_path
save_path=$2
model_name=$3
team_config_folder=$4
if [[ -d $team_config_folder ]]; then
    team_config=$(find $team_config_folder -type f -name "*-last.ckpt")
elif [[ -f $team_config_folder ]]; then
    team_config=$team_config_folder
fi
tcp_output_type=$5
slurm_output_dir=$6

sleep $((RANDOM % 60 + 1))

for i in $(seq 1 5); do

    new_save_path=$save_path"run$i"/$model_name/
    new_slurm_output_dir=$slurm_output_dir"run$i"/
    mkdir -p $new_slurm_output_dir

    . case_study/slurm_scripts/run_evaluation.sh --save_path $new_save_path --team_config $team_config --checkpoint_endpoint results.json --tcp_output_type $tcp_output_type > $new_slurm_output_dir/"$model_name"_$SLURM_JOB_ID.txt &

done

wait

echo "Benchmark finished."