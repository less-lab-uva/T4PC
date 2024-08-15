import argparse
import subprocess
import random
from pathlib import Path
from utils import check_slurm_queue

def dispatch_sbatch_job(script_path, output_file, model_ckpt, model_type, properties, batch_size, split, seed, job_id, prop_loss, mask_loss):
    dependency = f"--dependency=afterok:{job_id}" if job_id is not None else ""
    command = f"sbatch --output={output_file} \
        {dependency} \
        {script_path} \
        --model_ckpt {model_ckpt} \
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
    slurm_output_bd = base_dir / "slurm_output/rq2/"
    if not slurm_output_bd.exists():
        slurm_output_bd.mkdir(parents=True, exist_ok=True)
    script_path = "controlled_experiment/rq2/sbatch_rq2.sh"
    base_models = {
        "resnet": {
            0: base_dir / "rq1/data/resnet/models/base__split0/version_0/last_model.ckpt",
            1: base_dir / "rq1/data/resnet/models/base__split1/version_0/last_model.ckpt",
            2: base_dir / "rq1/data/resnet/models/base__split2/version_0/last_model.ckpt",
            3: base_dir / "rq1/data/resnet/models/base__split3/version_0/last_model.ckpt",
            4: base_dir / "rq1/data/resnet/models/base__split4/version_0/last_model.ckpt",
            5: base_dir / "rq1/data/resnet/models/base__split5/version_0/last_model.ckpt"
        }
    }
    
    indexes = list(range(6))

    for model_type in ["resnet"]:
        for split in indexes:
            seed = random.randint(0,1000000)
            for prop in ["green_light", "yel_red_light", "entities_within_10m", "stopped_no_reason25","leftmost_lane", "rightmost_lane"]:
                for prop_loss in [True, False]:
                    
                    # Define the parameters for the sbatch job
                    output_file = slurm_output_bd / f"{model_type}__{prop}__split{split}__%j.txt"
                    model_ckpt = base_models[model_type][split]
                    properties = [prop]
                    batch_size = 256
                    
                    dispatch_sbatch_job(script_path, output_file, model_ckpt, model_type, properties, batch_size, split, seed, args.job_id, prop_loss, mask_loss=True)

            # Check slurm queue
            check_slurm_queue("rq2")

if __name__ == "__main__":
    main()