import argparse
import subprocess
import random
import dotenv
import pandas as pd
import os
import time
from itertools import combinations
from pathlib import Path
import time

# Use the .env file to load environment variables
dotenv.load_dotenv('.env', override=True)


def wait_for_available_slot(job_name="t4pc", max_jobs=20, sleep_seconds=600, debug=False):
    while True:
        # Count number of jobs with the given name in the queue
        result = subprocess.run(
            f"squeue --name={job_name} | wc -l",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            # Subtract 1 for the header line
            job_count = int(result.stdout.strip()) - 1
        except ValueError:
            job_count = 0

        if job_count < max_jobs:
            print(f"Current jobs with name '{job_name}': {job_count}")
            break
        else:
            print(f"Reached job limit ({job_count}/{max_jobs}), sleeping for {sleep_seconds} seconds...")
        
        time.sleep(sleep_seconds)


def dispatch_sbatch_job(script_path, output_file, model_type, properties, batch_size, split, seed, job_id, prop_loss, mask_loss, log_dir, model_id, epochs=15, lrscheduler=False, debug=False, model_ckpt=None, lr=None):
    dependency = f"--dependency=afterok:{job_id}" if job_id is not None else ""
    command = f"sbatch --output={output_file} \
        {dependency} \
        {script_path} \
        --model_type {model_type} \
        --properties {' '.join(properties)} \
        --batch_size {batch_size} \
        --split_n {split} \
        --seed {seed} \
        --log_dir {log_dir} \
        --id {model_id} \
        --epochs {epochs} \
        {f'--lr {lr}' if lr else ''} \
        {f'--model_ckpt {model_ckpt}' if model_ckpt else ''} \
        {'--lrscheduler' if lrscheduler else ''} \
        {'--prop_loss' if prop_loss else ''} \
        {'--mask_loss' if mask_loss else ''}"
    wait_for_available_slot(job_name="t4pc", max_jobs=20, sleep_seconds=600, debug=debug)
    if debug:
        print(' '.join(command.split()) + "\n")
    else:
        print(' '.join(command.split()) + "\n")
        subprocess.run(command, shell=True)


def get_combinations(property_list):
    comb_dict = {}
    for sample in [1, 2, 4, 6]: # Modify this to set the number of properties in the optimization
        comb_list = list(combinations(property_list, sample))
        comb_dict[sample] = comb_list
    return comb_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Base directory
    base_dir = Path(os.getenv("EXP_DIR"))

    # Slurm output directory
    slurm_output_bd = base_dir / "slurm_output/exp/"
    if not slurm_output_bd.exists():
        slurm_output_bd.mkdir(parents=True, exist_ok=True)
    # Load log summary
    df = None
    if (slurm_output_bd / "summary.csv").exists():
        df = pd.read_csv(slurm_output_bd / "summary.csv")

    # Model output directory
    model_output_bd = base_dir / "exp/"
    if not model_output_bd.exists():
        model_output_bd.mkdir(parents=True, exist_ok=True)

    # Parameters
    script_path = "controlled_experiment/slurm_scripts/sbatch_train.sh"
    all_properties = ["leftmost_lane", "rightmost_lane", "green_light",
                      "yel_red_light", "entities_within_10m", "stopped_no_reason25"]
    epochs = 15
    batch_size = 256
    lrscheduler = False

    # List of town indexes (0-5)
    indexes = [0, 1, 2, 3, 4, 5]
    # Define random seed per split
    random.seed(0)
    seeds = {}
    for i in range(0, len(indexes), 2):
        seed = random.randint(0, 1000000)
        for j in [i, i + 1]:
            seeds[j] = seed
    # Get property combinations for different sample sizes
    all_combinations = get_combinations(all_properties)

    if not args.finetune:
        for model_type in ["resnet"]:
            for sample_size, comb_list in all_combinations.items():
                for c_num, comb in enumerate(comb_list):
                    for split in indexes:

                        # Random seed for this split
                        seed = seeds[split]

                        # Base model without masking and property loss (1 for all properties)
                        if sample_size == 2 and c_num == 0:
                            # Avoid running the same model
                            if df is not None:
                                try:
                                    flag = df[(df["properties"] == "base") & (df["split"] == split)]["error"].item()
                                except:
                                    flag = True
                            if df is None or flag:
                                output_file = slurm_output_bd / f"{model_type}__base__split{split}__%j.txt"
                                log_dir = model_output_bd / f"{model_type}"
                                model_id = f"base__split{split}"
                                dispatch_sbatch_job(script_path, output_file, model_type, all_properties, batch_size,
                                                    split, seed, args.job_id, prop_loss=False, mask_loss=False,
                                                    log_dir=log_dir, model_id=model_id, epochs=epochs,
                                                    lrscheduler=lrscheduler, debug=args.debug)

                        props_list = list(comb)
                        properties = '-'.join(props_list)
                        # Avoid running the same model
                        if df is not None:
                            try:
                                flag = df[(df["properties"] == properties) & (df["split"] == split)]["error"].item()
                            except:
                                flag = True
                        if df is None or flag:
                            # Model with property loss and masking
                            output_file = slurm_output_bd / f"{model_type}__{properties}__split{split}__%j.txt"
                            log_dir = model_output_bd / f"{model_type}"
                            model_id = f"{properties}__split{split}"
                            dispatch_sbatch_job(script_path, output_file, model_type, props_list, batch_size,
                                                split, seed, args.job_id, prop_loss=True, mask_loss=True,
                                                log_dir=log_dir, model_id=model_id, epochs=epochs,
                                                lrscheduler=lrscheduler, debug=args.debug)
    else:
        for model_type in ["resnet"]:
            for sample_size, comb_list in all_combinations.items():
                for c_num, comb in enumerate(comb_list):
                    for split in indexes:

                        # Random seed for this split
                        seed = seeds[split]
                        base_ckpt = model_output_bd / f"{model_type}" / f"base__split{split}/version_0/epoch_14.ckpt"

                        # Finetune base model without masking and property loss
                        if c_num == 0:
                            output_file = slurm_output_bd / f"{model_type}__base_finetuned__split{split}__%j.txt"
                            log_dir = model_output_bd / f"{model_type}"
                            model_id = f"base_finetuned__split{split}"
                            dispatch_sbatch_job(script_path, output_file, model_type, all_properties, batch_size,
                                                split, seed, args.job_id, prop_loss=False, mask_loss=False,
                                                log_dir=log_dir, model_id=model_id, epochs=epochs,
                                                lrscheduler=lrscheduler, debug=args.debug, model_ckpt=base_ckpt, lr=1e-5)

                        props_list = list(comb)
                        properties = '-'.join(props_list)
                        # Finetune model with property loss and masking
                        output_file = slurm_output_bd / f"{model_type}__{properties}_finetuned__split{split}__%j.txt"
                        log_dir = model_output_bd / f"{model_type}"
                        model_id = f"{properties}_finetuned__split{split}"
                        dispatch_sbatch_job(script_path, output_file, model_type, props_list, batch_size,
                                            split, seed, args.job_id, prop_loss=True, mask_loss=True,
                                            log_dir=log_dir, model_id=model_id, epochs=epochs,
                                            lrscheduler=lrscheduler, debug=args.debug, model_ckpt=base_ckpt, lr=1e-5)

if __name__ == "__main__":
    main()
