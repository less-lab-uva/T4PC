import argparse
import subprocess
import random
from pathlib import Path
from itertools import combinations
from utils import check_slurm_queue

def dispatch_sbatch_job(script_path, output_file, model_type, properties, batch_size, split_n, seed, job_id, prop_loss, mask_loss):
    dependency = f"--dependency=afterok:{job_id}" if job_id is not None else ""
    command = f"sbatch --output={output_file} \
        {dependency} \
        {script_path} \
        --model_type {model_type} \
        --properties {' '.join(properties)} \
        --batch_size {batch_size} \
        --split_n {split_n} \
        --seed {seed} \
        {'--prop_loss' if prop_loss else ''} \
        {'--mask_loss' if mask_loss else ''}"
    subprocess.run(command, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)
    
    base_dir = Path("controlled_experiment/")
    slurm_output_bd = base_dir / "slurm_output/rq3_all/"
    if not slurm_output_bd.exists():
        slurm_output_bd.mkdir(parents=True, exist_ok=True)
    script_path = "controlled_experiment/rq3/sbatch_rq3_all.sh"
    
    properties = ["leftmost_lane", "rightmost_lane", "green_light", "yel_red_light", "entities_within_10m", "stopped_no_reason25"]
    batch_size = 256

    # Generate all combinations of 2 to n elements
    all_combinations = []
    for r in range(2, len(properties) + 1):
        all_combinations.extend(combinations(properties, r))

    indexes = list(range(5))

    for model_type in ["resnet"]:
        for split in indexes:
            seed = random.randint(0,1000000)

            # Base model without masking and property loss (1 for all properties)
            output_file = slurm_output_bd / f"{model_type}__base__split{split}__%j.txt"
            dispatch_sbatch_job(script_path, output_file, model_type, properties, batch_size, split, seed, args.job_id, prop_loss=False, mask_loss=False)

            # for props in all_combinations:
            for props in [properties]:

                # Define the parameters for the sbatch job
                props_list = list(props)
                output_file = slurm_output_bd / f"{model_type}__{'-'.join(props_list)}__split{split}__%j.txt"                
                
                dispatch_sbatch_job(script_path, output_file, model_type, props_list, batch_size, split, seed, args.job_id, prop_loss=True, mask_loss=True)
            
            # Check slurm queue
            check_slurm_queue("rq3")

if __name__ == "__main__":
    main()