import argparse
import subprocess
import random
from pathlib import Path
from utils import check_slurm_queue

def dispatch_sbatch_job(script_path, output_file, model_type, properties, batch_size, split, seed, job_id, prop_loss, mask_loss):
    dependency = f"--dependency=afterok:{job_id}" if job_id is not None else ""
    command = f"sbatch --output={output_file} \
        {dependency} \
        {script_path} \
        --model_type {model_type} \
        --properties {' '.join(properties)} \
        --batch_size {batch_size} \
        --split_n {split} \
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
    slurm_output_bd = base_dir / "slurm_output/rq1/"
    if not slurm_output_bd.exists():
        slurm_output_bd.mkdir(parents=True, exist_ok=True)
    script_path = "controlled_experiment/rq1/sbatch_rq1.sh"
    all_properties = ["leftmost_lane", "rightmost_lane", "green_light", "yel_red_light", "entities_within_10m", "stopped_no_reason25"]
    batch_size = 256
    
    # indexes = list(range(6))
    indexes = [0]

    for model_type in ["resnet"]:
        for split in indexes:
            seed = random.randint(0,1000000)

            # Base model without masking and property loss (1 for all properties)
            output_file = slurm_output_bd / f"{model_type}__base__split{split}__%j.txt"
            dispatch_sbatch_job(script_path, output_file, model_type, all_properties, batch_size, split, seed, args.job_id, prop_loss=False, mask_loss=False)
            
            for prop in all_properties:
                    
                # Define the parameters for the sbatch job
                output_file = slurm_output_bd / f"{model_type}__{prop}__pl__split{split}__%j.txt"
                properties = [prop]
                
                # dispatch_sbatch_job(script_path, output_file, model_type, properties, batch_size, split, seed, args.job_id, prop_loss=True, mask_loss=True)

            # Check slurm queue
            check_slurm_queue("rq1")

if __name__ == "__main__":
    main()