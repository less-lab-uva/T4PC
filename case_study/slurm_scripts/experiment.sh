#!/bin/bash

source .env

original_pwd=$ROOT_DIR
cd $original_pwd


############################################################################################################
# 15 Epochs
############################################################################################################


###################################
### NO PL | lr=5e-5 | MSE | 15e ###
###################################
exp_id=final_nopl_5e5_15e

for mn in $(seq 1 5)
do
    ### Training
    model_name="TCP_nopl_lr5e-5_mse_15e_mn_"$mn

    slurm_output_dir=case_study/slurm_outputs/"$exp_id"/training
    mkdir -p $slurm_output_dir

    # Train without property loss, using pre-trained weights
    output=$(sbatch -o $slurm_output_dir/training_"$model_name"_%j.txt \
        case_study/slurm_scripts/sbatch_training.sh $original_pwd \
            --logdir case_study/logs/"$exp_id" \
            --id "$model_name" \
            --num_workers 4 \
            --val_every 1 \
            --lrs_step 7 \
            --lrs_cut 0.5 \
            --lr 0.00005 \
            --optimizer adam \
            --epochs 15 \
            --resume_from $ROOT_DIR/case_study/TCP/best_model.ckpt \
            --properties "phi_ss_acc_v6" "phi_ss_stop_v6" \
            --lambda_pl 1.0)
    job_id=${output##* }
    echo $job_id

    ### Benchmark
    data_collection_dir=./case_study/results_summary/"$exp_id"/
    slurm_output_dir=./case_study/slurm_outputs/"$exp_id"/

    mkdir -p $data_collection_dir
    mkdir -p $slurm_output_dir

    sbatch -o $slurm_output_dir/"$model_name"_%j.txt \
        --dependency=afterok:$job_id \
        case_study/slurm_scripts/sbatch_benchmark.sh \
            $original_pwd \
            $data_collection_dir \
            "$model_name" \
            logs/"$exp_id"/"$model_name"/ \
            "dnn" \
            $slurm_output_dir
done


############################################################################################################


################################
### PL | lr=5e-5 | MSE | 15e ###
################################
exp_id=final_pl_5e5_15e

for mn in $(seq 1 5)
do
    ### Training
    model_name="TCP_pl_lr5e-5_mse_15e_mn_"$mn

    slurm_output_dir=case_study/slurm_outputs/"$exp_id"/training
    mkdir -p $slurm_output_dir

    # Train with property loss, using pre-trained weights
    output=$(sbatch -o $slurm_output_dir/training_"$model_name"_%j.txt \
        case_study/slurm_scripts/sbatch_training.sh $original_pwd \
            --logdir case_study/logs/"$exp_id" \
            --id "$model_name" \
            --num_workers 4 \
            --val_every 1 \
            --prop_loss \
            --lrs_step 7 \
            --lrs_cut 0.5 \
            --lr 0.00005 \
            --mask_loss \
            --optimizer adam \
            --epochs 15 \
            --augment_data \
            --resume_from $ROOT_DIR/case_study/TCP/best_model.ckpt \
            --properties "phi_ss_acc_v6" "phi_ss_stop_v6" \
            --lambda_pl 1.0)
    job_id=${output##* }
    echo $job_id

    ### Benchmark
    data_collection_dir=./case_study/results_summary/"$exp_id"/
    slurm_output_dir=./case_study/slurm_outputs/"$exp_id"/

    mkdir -p $data_collection_dir
    mkdir -p $slurm_output_dir

    sbatch -o $slurm_output_dir/"$model_name"_%j.txt \
        --dependency=afterok:$job_id \
        case_study/slurm_scripts/sbatch_benchmark.sh \
            $original_pwd \
            $data_collection_dir \
            "$model_name" \
            logs/"$exp_id"/"$model_name"/ \
            "dnn" \
            $slurm_output_dir
done


############################################################################################################
# 10 Epochs
############################################################################################################


###################################
### NO PL | lr=5e-5 | MSE | 10e ###
###################################
exp_id=final_nopl_5e5_10e

for mn in $(seq 1 5)
do
    ### Training
    model_name="TCP_nopl_lr5e-5_mse_10e_mn_"$mn

    slurm_output_dir=case_study/slurm_outputs/"$exp_id"/training
    mkdir -p $slurm_output_dir

    # Train without property loss, using pre-trained weights
    output=$(sbatch -o $slurm_output_dir/training_"$model_name"_%j.txt \
        case_study/slurm_scripts/sbatch_training.sh $original_pwd \
            --logdir case_study/logs/"$exp_id" \
            --id "$model_name" \
            --num_workers 4 \
            --val_every 1 \
            --lrs_step 5 \
            --lrs_cut 0.5 \
            --lr 0.00005 \
            --optimizer adam \
            --epochs 10 \
            --resume_from $ROOT_DIR/case_study/TCP/best_model.ckpt \
            --properties "phi_ss_acc_v6" "phi_ss_stop_v6" \
            --lambda_pl 1.0)
    job_id=${output##* }
    echo $job_id

    ### Benchmark
    data_collection_dir=./case_study/results_summary/"$exp_id"/
    slurm_output_dir=./case_study/slurm_outputs/"$exp_id"/

    mkdir -p $data_collection_dir
    mkdir -p $slurm_output_dir

    sbatch -o $slurm_output_dir/"$model_name"_%j.txt \
        --dependency=afterok:$job_id \
        case_study/slurm_scripts/sbatch_benchmark.sh \
            $original_pwd \
            $data_collection_dir \
            "$model_name" \
            logs/"$exp_id"/"$model_name"/ \
            "dnn" \
            $slurm_output_dir
done


############################################################################################################


################################
### PL | lr=5e-5 | MSE | 10e ###
################################
exp_id=final_pl_5e5_10e

for mn in $(seq 1 5)
do
    ### Training
    model_name="TCP_pl_lr5e-5_mse_10e_mn_"$mn

    slurm_output_dir=case_study/slurm_outputs/"$exp_id"/training
    mkdir -p $slurm_output_dir

    # Train with property loss, using pre-trained weights
    output=$(sbatch -o $slurm_output_dir/training_"$model_name"_%j.txt \
        case_study/slurm_scripts/sbatch_training.sh $original_pwd \
            --logdir case_study/logs/"$exp_id" \
            --id "$model_name" \
            --num_workers 4 \
            --val_every 1 \
            --prop_loss \
            --lrs_step 5 \
            --lrs_cut 0.5 \
            --lr 0.00005 \
            --mask_loss \
            --optimizer adam \
            --epochs 10 \
            --augment_data \
            --resume_from $ROOT_DIR/case_study/TCP/best_model.ckpt \
            --properties "phi_ss_acc_v6" "phi_ss_stop_v6" \
            --lambda_pl 1.0)
    job_id=${output##* }
    echo $job_id

    ### Benchmark
    data_collection_dir=./case_study/results_summary/"$exp_id"/
    slurm_output_dir=./case_study/slurm_outputs/"$exp_id"/

    mkdir -p $data_collection_dir
    mkdir -p $slurm_output_dir

    sbatch -o $slurm_output_dir/"$model_name"_%j.txt \
        --dependency=afterok:$job_id \
        case_study/slurm_scripts/sbatch_benchmark.sh \
            $original_pwd \
            $data_collection_dir \
            "$model_name" \
            logs/"$exp_id"/"$model_name"/ \
            "dnn" \
            $slurm_output_dir
done


############################################################################################################
# 5 Epochs
############################################################################################################


###################################
### NO PL | lr=5e-5 | MSE | 5e ###
###################################
exp_id=final_nopl_5e5_5e

for mn in $(seq 1 5)
do
    ### Training
    model_name="TCP_nopl_lr5e-5_mse_5e_mn_"$mn

    slurm_output_dir=case_study/slurm_outputs/"$exp_id"/training
    mkdir -p $slurm_output_dir

    # Train without property loss, using pre-trained weights
    output=$(sbatch -o $slurm_output_dir/training_"$model_name"_%j.txt \
        case_study/slurm_scripts/sbatch_training.sh $original_pwd \
            --logdir case_study/logs/"$exp_id" \
            --id "$model_name" \
            --num_workers 4 \
            --val_every 1 \
            --lrs_step 2 \
            --lrs_cut 0.5 \
            --lr 0.00005 \
            --optimizer adam \
            --epochs 5 \
            --resume_from $ROOT_DIR/case_study/TCP/best_model.ckpt \
            --properties "phi_ss_acc_v6" "phi_ss_stop_v6" \
            --lambda_pl 1.0)
    job_id=${output##* }
    echo $job_id

    ### Benchmark
    data_collection_dir=./case_study/results_summary/"$exp_id"/
    slurm_output_dir=./case_study/slurm_outputs/"$exp_id"/

    mkdir -p $data_collection_dir
    mkdir -p $slurm_output_dir

    sbatch -o $slurm_output_dir/"$model_name"_%j.txt \
        --dependency=afterok:$job_id \
        case_study/slurm_scripts/sbatch_benchmark.sh \
            $original_pwd \
            $data_collection_dir \
            "$model_name" \
            logs/"$exp_id"/"$model_name"/ \
            "dnn" \
            $slurm_output_dir
done


############################################################################################################


################################
### PL | lr=5e-5 | MSE | 5e ###
################################
exp_id=final_pl_5e5_5e

for mn in $(seq 1 5)
do
    ### Training
    model_name="TCP_pl_lr5e-5_mse_5e_mn_"$mn

    slurm_output_dir=case_study/slurm_outputs/"$exp_id"/training
    mkdir -p $slurm_output_dir

    # Train with property loss, using pre-trained weights
    output=$(sbatch -o $slurm_output_dir/training_"$model_name"_%j.txt \
        case_study/slurm_scripts/sbatch_training.sh $original_pwd \
            --logdir case_study/logs/"$exp_id" \
            --id "$model_name" \
            --num_workers 4 \
            --val_every 1 \
            --prop_loss \
            --lrs_step 2 \
            --lrs_cut 0.5 \
            --lr 0.00005 \
            --mask_loss \
            --optimizer adam \
            --epochs 5 \
            --augment_data \
            --resume_from $ROOT_DIR/case_study/TCP/best_model.ckpt \
            --properties "phi_ss_acc_v6" "phi_ss_stop_v6" \
            --lambda_pl 1.0)
    job_id=${output##* }
    echo $job_id

    ### Benchmark
    data_collection_dir=./case_study/results_summary/"$exp_id"/
    slurm_output_dir=./case_study/slurm_outputs/"$exp_id"/

    mkdir -p $data_collection_dir
    mkdir -p $slurm_output_dir

    sbatch -o $slurm_output_dir/"$model_name"_%j.txt \
        --dependency=afterok:$job_id \
        case_study/slurm_scripts/sbatch_benchmark.sh \
            $original_pwd \
            $data_collection_dir \
            "$model_name" \
            logs/"$exp_id"/"$model_name"/ \
            "dnn" \
            $slurm_output_dir
done