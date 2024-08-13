import pandas as pd
import argparse
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from torch.nn import functional as F
from pathlib import Path

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

def get_violations_and_loss(rq, model_type):
    data_dir = Path(f"controlled_experiment/rq{rq}/data/{model_type}/")
    all_properties = ["leftmost_lane", "rightmost_lane", "green_light", "yel_red_light", "entities_within_10m", "stopped_no_reason25"]

    data = {
        "model":[],
        "split":[],
        "ploss":[],
        "violations":[],
        "val_steering_loss":[],
        "val_acceleration_loss":[],
        "violation_loss":[]
    }

    indexes = list(range(6))

    for split in indexes:
        bl_df = pd.read_csv(f"controlled_experiment/rq{rq}/data/{model_type}/violations/test__base__split{split}.csv")
        for prop in all_properties:
            # Load the csv file
            if rq == "1" or rq == "2":
                df = pd.read_csv(data_dir / f"violations/test__{prop}__pl__split{split}.csv")
            elif rq == "3":
                p_name = "-".join(all_properties)
                df = pd.read_csv(data_dir / f"violations/test__{p_name}__pl__split{split}.csv")
            else:
                raise NotImplementedError(f"RQ {rq} not implemented.")
            # Add data to the dictionary
            for ploss in [False, True]:
                if ploss:
                    violations = len(df[(df['prop_name'] == prop) & (df['violation'] == True)])
                    val_acc_loss, val_steer_loss = loss_with_mask(df)
                    violation_loss = (abs(df[f"{PROP_OUTPUT[prop]}_pred"] - PROP_THRESHOLD[prop])).mean()
                else:
                    violations = len(bl_df[(bl_df['prop_name'] == prop) & (bl_df['violation'] == True)])
                    val_acc_loss, val_steer_loss = loss(bl_df)
                    violation_loss = (abs(bl_df[f"{PROP_OUTPUT[prop]}_pred"] - PROP_THRESHOLD[prop])).mean()

                data["model"].append(prop)
                data["split"].append(split)
                data["ploss"].append(ploss)
                data["violations"].append(violations)
                data["val_steering_loss"].append(val_steer_loss.item())
                data["val_acceleration_loss"].append(val_acc_loss.item())
                data["violation_loss"].append(violation_loss.item())

    # Create the dataframe
    df = pd.DataFrame(data)

    # Calculate violations difference percentage
    q = df.groupby(["model","ploss"]).sum()["violations"].reset_index()
    v_df = q.pivot(index="model", columns="ploss", values="violations")
    v_df["diff"] = (v_df[False] - v_df[True])
    v_df["diff_percentage"] = v_df.apply(lambda row: row["diff"] / row[False] * 100 if row[False] > 0 else row[True] * 100, axis=1)

    # Calculate loss difference percentage
    q = df.groupby(["model","ploss"]).sum()[["val_steering_loss","val_acceleration_loss"]].reset_index()
    loss_df = q.pivot(index="model", columns="ploss", values=["val_steering_loss","val_acceleration_loss"])
    loss_df["diff_steering"] = (loss_df["val_steering_loss"][False] - loss_df["val_steering_loss"][True])
    loss_df["diff_steering_percentage"] = loss_df.apply(lambda row: row["diff_steering"] / row["val_steering_loss"][False] * 100 if row["val_steering_loss"][False] > 0 else row["val_steering_loss"][True] * 100, axis=1)
    loss_df["diff_acceleration"] = (loss_df["val_acceleration_loss"][False] - loss_df["val_acceleration_loss"][True])
    loss_df["diff_acceleration_percentage"] = loss_df.apply(lambda row: row["diff_acceleration"] / row["val_acceleration_loss"][False] * 100 if row["val_acceleration_loss"][False] > 0 else row["val_acceleration_loss"][True] * 100, axis=1)

    # Get model names
    models = df['model'].unique()

    return v_df, loss_df, models

def main():
    # Define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'vit'], help='Model architecture.')
    args = parser.parse_args()

    fig, axs = plt.subplots(1,3, figsize=(15,5))
    v_dict = {}
    loss_dict = {}

    for i, rq in enumerate(["1","2","3"]):
        v_df, loss_df, models = get_violations_and_loss(rq, args.model_type)
        v_dict[rq] = v_df
        loss_dict[rq] = loss_df

        palette = sns.color_palette("muted", len(models))
        color_dict = {model: color for model, color in zip(models, palette)}

        # Plotting horizontal lines for violations difference percentage
        for index, row in v_df.iterrows():
            # Steering property
            if index in ["leftmost_lane", "rightmost_lane"]:
                axs[i].axhline(y=row['diff_percentage'], label=index, linestyle='-', color="black", alpha=0.5, linewidth=1)
            # Acceleration property
            else:
                axs[i].axhline(y=row['diff_percentage'], label=index, linestyle=(0, (5, 5)), color="black", alpha=0.5, linewidth=1)
            steering_diff = loss_df.loc[index]["diff_steering_percentage"].item()
            acceleration_diff = loss_df.loc[index]["diff_acceleration_percentage"].item()
            axs[i].plot(steering_diff, row["diff_percentage"], marker="o", color=color_dict[index], markersize=10)
            axs[i].plot(acceleration_diff, row["diff_percentage"], marker="x", color=color_dict[index], markersize=10)
        # Plot vertical line at 0
        axs[i].axvline(x=0, color='black', linestyle=(0, (1, 5)), alpha=1, zorder=0)
        axs[i].axhline(y=0, color='black', linestyle=(0, (1, 5)), alpha=1, zorder=0)

        if i == 0:
            axs[i].set_ylabel(r"Violation improvement (%) $\uparrow$", fontsize='x-large')
        axs[i].set_ylim(-10,110)
        axs[i].set_xlabel(r"Loss improvement (%) $\uparrow$", fontsize='x-large')
        axs[i].set_xlim(-31,31)

        # Set the font size of x and y ticks
        axs[i].tick_params(axis='both', which='major', labelsize='large')  # Set major ticks font size
        
        # Get existing legend
        old_handles, old_labels = axs[i].get_legend_handles_labels()
        # Rename the labels
        name_map = {
            "leftmost_lane": "$\phi_6$ - NoSteerLeftOutRoad",
            "rightmost_lane": "$\phi_5$ - NoSteerRightOutRoad",
            "green_light": "$\phi_4$ - AccelerateForGreen",
            "yel_red_light": "$\phi_2$ - StopForYellowRed",
            "entities_within_10m": "$\phi_1$ - StopToAvoidCollision",
            "stopped_no_reason25": "$\phi_3$ - NoStopForNoReason"
        }
        new_labels = [name_map[label] for label in old_labels]
        # Change handle colors
        custom_handles = []
        for handle, label in zip(old_handles, old_labels):
            custom_handle = mlines.Line2D([0], [0], color=color_dict[label], alpha=1, linewidth=2, linestyle=handle.get_linestyle())
            custom_handles.append(custom_handle)

        # Reindex custom_handles and new_labels to match the order of the properties
        idx_order = [0,5,4,1,3,2]
        custom_handles = [custom_handles[i] for i in idx_order]
        new_labels = [new_labels[i] for i in idx_order]

        # Create proxy artists for the custom labels
        steer_loss_diff_label = plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', linestyle='None', markersize=10)
        acc_loss_diff_label = plt.Line2D([0], [0], marker='x', color='black', markerfacecolor='black', linestyle='None', markersize=10)
        # Add the custom labels to the legend
        handles = custom_handles + [steer_loss_diff_label, acc_loss_diff_label]
        labels = new_labels + ['Steering loss diff', 'Acceleration loss diff']

        # plt.legend(handles=handles, labels=labels)
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize='x-large')
        
        axs[i].text(0.05, 0.95, f'RQ {i+1}', transform=axs[i].transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(facecolor='darkgray', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.5'), zorder=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.27)
        plt.savefig(f"./controlled_experiment/results.png")

    plt.clf()
            
if __name__ == '__main__':
    main()