import pandas as pd
import argparse
import os
import torch
import dotenv
from torch.nn import functional as F
from pathlib import Path
from itertools import combinations

PROP_THRESHOLD = {
    "leftmost_lane": -0.07,
    "rightmost_lane": 0.07,
    "green_light": 0.25,
    "yel_red_light": -0.25,
    "entities_within_10m": -0.25,
    "stopped_no_reason25": 0.25
}
PROP_OUTPUT = {
    "leftmost_lane": "steer",
    "rightmost_lane": "steer",
    "green_light": "acc",
    "yel_red_light": "acc",
    "entities_within_10m": "acc",
    "stopped_no_reason25": "acc"
}

# Use the .env file to load environment variables
dotenv.load_dotenv('.env', override=True)

def loss_with_mask(v_df):
    masks = v_df.groupby("img_id")[["acc_prop_mask","steer_prop_mask"]].min().reset_index()
    df = v_df.groupby("img_id")[["acc_label","steer_label","acc_pred","steer_pred"]].first().reset_index()
    val_steering_loss = F.mse_loss(torch.tensor((df['steer_label'] * masks["steer_prop_mask"]).tolist()), torch.tensor((df['steer_pred'] * masks["steer_prop_mask"]).tolist()))
    val_acceleration_loss = F.mse_loss(torch.tensor((df['acc_label'] * masks["acc_prop_mask"]).tolist()), torch.tensor((df['acc_pred'] * masks["acc_prop_mask"]).tolist()))
    if (masks["acc_prop_mask"] > 0).any():
        val_acceleration_loss *= len(masks) / len(masks[masks["acc_prop_mask"] > 0])
    if (masks["steer_prop_mask"] > 0).any():
        val_steering_loss *= len(masks["steer_prop_mask"]) / len(masks[masks["steer_prop_mask"] > 0])
    return val_acceleration_loss, val_steering_loss

def loss(v_df):
    df = v_df.groupby("img_id")[["acc_label","steer_label","acc_pred","steer_pred"]].first().reset_index()
    val_steering_loss = F.mse_loss(torch.tensor(df['steer_label']), torch.tensor(df['steer_pred']))
    val_acceleration_loss = F.mse_loss(torch.tensor(df['acc_label']), torch.tensor(df['acc_pred']))
    return val_acceleration_loss, val_steering_loss

def get_version(path):
    versions = [int(folder.name.split('_')[-1]) for folder in path.iterdir() if folder.name.startswith('version_')]
    return max(versions) if versions else None

def get_combinations(property_list, sample):
    comb_list = list(combinations(property_list, sample))
    return comb_list

def get_violations_and_loss(rq, model_type, finetuned=False):
    if rq in ['1',"2","4",'6']:
        data_dir = Path(f"")
        # Base directory
        base_dir = Path(os.getenv("EXP_DIR"))
        # Slurm output directory
        data_dir = base_dir / "slurm_output/exp/"

    all_properties = ["leftmost_lane", "rightmost_lane", "green_light", "yel_red_light", "entities_within_10m", "stopped_no_reason25"]

    data = {
        "model":[],
        "property":[],
        "rq":[],
        "split":[],
        "ploss":[],
        "violations":[],
        "val_steering_loss":[],
        "val_acceleration_loss":[],
        "violation_loss":[]
    }

    for split in [0,1,2,3,4,5]:
        # Load the bl csv file
        if rq in ['1',"2","4",'6']:
            if finetuned:
                bl_base_path = data_dir / f"base_finetuned__split{split}/"
            else:
                bl_base_path = data_dir / f"base__split{split}/"
            version_number = get_version(bl_base_path)
            bl_path = bl_base_path / f"version_{version_number}/violations/test__e14.csv"
        
        bl_df = pd.read_csv(bl_path)

        prop_comb = get_combinations(all_properties, int(rq))
        for comb in prop_comb:
            # Load the csv file
            p_name = "-".join(comb)
            if rq in ['1',"2","4",'6']:
                if finetuned:
                    df_path = data_dir / f"{p_name}_finetuned__split{split}/"
                else:
                    df_path = data_dir / f"{p_name}__split{split}/"
                version_number = get_version(df_path)
                df = pd.read_csv(df_path / f"version_{version_number}/violations/test__e14.csv")
            else:
                raise NotImplementedError(f"RQ {rq} not implemented.")
            
            # Add data to the dictionary
            for prop in comb:
                for ploss in [False, True]:
                    if ploss:
                        violations = len(df[(df['prop_name'] == prop) & (df['violation'] == True)])
                        val_acc_loss, val_steer_loss = loss_with_mask(df)
                        violation_loss = (abs(df[f"{PROP_OUTPUT[prop]}_pred"] - PROP_THRESHOLD[prop])).mean()
                    else:
                        violations = len(bl_df[(bl_df['prop_name'] == prop) & (bl_df['violation'] == True)])
                        val_acc_loss, val_steer_loss = loss(bl_df)
                        violation_loss = (abs(bl_df[f"{PROP_OUTPUT[prop]}_pred"] - PROP_THRESHOLD[prop])).mean()

                    data["model"].append("-".join(comb))
                    data["property"].append(prop)
                    data["rq"].append(rq)
                    data["split"].append(split)
                    data["ploss"].append(ploss)
                    data["violations"].append(violations)
                    data["val_steering_loss"].append(val_steer_loss.item())
                    data["val_acceleration_loss"].append(val_acc_loss.item())
                    data["violation_loss"].append(violation_loss)

    # Create the dataframe
    df = pd.DataFrame(data)
    return df

def main():
    # Define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'vit'], help='Model architecture.')
    parser.add_argument('--finetuned', action='store_true', help='Use finetuned model.')
    args = parser.parse_args()

    all_df = []

    if args.finetuned:
        print("Finetuned.")
        df = get_violations_and_loss("1", args.model_type, args.finetuned)
        all_df.append(df)
    else:
        print("Full Training.")
        for i, rq in enumerate(["1","2","4","6"]):
            df = get_violations_and_loss(rq, args.model_type, args.finetuned)
            all_df.append(df)

    combined_df = pd.concat(all_df, ignore_index=True)
    combined_df.to_csv(f"/Data3/Research/TCP_fork/exp_results_rivanna/all_data_new_old_params2_finetuned.csv", index=False)

            
if __name__ == '__main__':
    main()