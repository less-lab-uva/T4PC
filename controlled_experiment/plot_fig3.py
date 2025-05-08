import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

df = pd.read_csv('./controlled_experiment/training_data.csv')
dff = pd.read_csv('./controlled_experiment/finetuned_data.csv')

# Training data
q = df.groupby(["property","rq","ploss"])["violations"].mean().reset_index()
v_df = q.pivot(index=["property","rq"], columns="ploss", values="violations")
v_df["diff"] = (v_df[False] - v_df[True])
v_df["diff_percentage"] = v_df.apply(lambda row: row["diff"] / row[False] * 100 if row[False] > 0 else row[True] * 100, axis=1)

q = df.groupby(["property","rq","ploss"]).sum()[["val_steering_loss","val_acceleration_loss"]].reset_index()
loss_df = q.pivot(index=["property","rq"], columns="ploss", values=["val_steering_loss","val_acceleration_loss"])
loss_df["diff_steering"] = (loss_df["val_steering_loss"][False] - loss_df["val_steering_loss"][True])
loss_df["diff_steering_percentage"] = loss_df.apply(lambda row: row["diff_steering"] / row["val_steering_loss"][False] * 100 if row["val_steering_loss"][False] > 0 else row["val_steering_loss"][True] * 100, axis=1)
loss_df["diff_acceleration"] = (loss_df["val_acceleration_loss"][False] - loss_df["val_acceleration_loss"][True])
loss_df["diff_acceleration_percentage"] = loss_df.apply(lambda row: row["diff_acceleration"] / row["val_acceleration_loss"][False] * 100 if row["val_acceleration_loss"][False] > 0 else row["val_acceleration_loss"][True] * 100, axis=1)

fig = plt.figure(figsize=(20, 5))
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1.3])  # Make the last subplot twice as wide
axs = [fig.add_subplot(gs[i]) for i in range(5)]

avg_num = [1, 5, 10, 1]
for i, rq in enumerate(["1","2","4","6"]):

    palette = sns.color_palette("muted", 6)
    all_properties = ["leftmost_lane", "rightmost_lane", "green_light", "yel_red_light", "entities_within_10m", "stopped_no_reason25"]
    color_dict = {prop: color for prop, color in zip(all_properties, palette)}

    # Plotting horizontal lines for violations difference percentage
    for index, row in v_df.iterrows():
        if index[1] == int(rq):
            # Steering property
            if index[0] in ["leftmost_lane", "rightmost_lane"]:
                axs[i].axhline(y=row['diff_percentage'], label=index[0], linestyle='-', color="black", alpha=0.5, linewidth=1)
                # Determine label position
                label_y = row['diff_percentage'] + 5 if row['diff_percentage'] < axs[i].get_ylim()[1] - 10 else row['diff_percentage'] - 5
                axs[i].text(0, label_y, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
            # Acceleration property
            else:
                axs[i].axhline(y=row['diff_percentage'], label=index[0], linestyle=(0, (5, 5)), color="black", alpha=0.5, linewidth=1)
                # Determine label position
                label_y = row['diff_percentage'] + 5 if row['diff_percentage'] < axs[i].get_ylim()[1] - 10 else row['diff_percentage'] - 5
                flag = False
                if index[1] == 1:
                    if index[0] == "green_light":
                        axs[i].text(0, label_y + 12, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                        flag = True
                elif index[1] == 2:
                    if index[0] == "green_light":
                        axs[i].text(0, label_y + 12, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                        flag = True
                elif index[1] == 4:
                    if index[0] == "entities_within_10m":
                        axs[i].text(0, label_y + 12, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                        flag = True
                    if index[0] == "green_light":
                        axs[i].text(0, label_y + 2, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                        flag = True
                    if index[0] == "stopped_no_reason25":
                        axs[i].text(0, label_y - 2, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                        flag = True
                elif index[1] == 6:
                    if index[0] == "entities_within_10m":
                        axs[i].text(0, label_y + 2, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                        flag = True
                    if index[0] == "green_light":
                        axs[i].text(0, label_y + 15, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                        flag = True
                    if index[0] == "stopped_no_reason25":
                        axs[i].text(0, label_y + 5, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                        flag = True
                if not flag:
                    axs[i].text(0, label_y, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')

            steering_diff = loss_df.loc[index]["diff_steering_percentage"].item()
            acceleration_diff = loss_df.loc[index]["diff_acceleration_percentage"].item()
            axs[i].plot(steering_diff, row["diff_percentage"], marker="o", color=color_dict[index[0]], markersize=10)
            axs[i].plot(acceleration_diff, row["diff_percentage"], marker="x", color=color_dict[index[0]], markersize=10)
    # Plot vertical line at 0
    axs[i].axvline(x=0, color='black', linestyle=(0, (1, 5)), alpha=1, zorder=0)
    axs[i].axhline(y=0, color='black', linestyle=(0, (1, 5)), alpha=1, zorder=0)

    if i == 0:
        axs[i].set_ylabel(r"Violation improvement (%) $\uparrow$", fontsize='x-large')
    axs[i].set_ylim(-10,110)
    axs[i].set_xlabel(r"Loss improvement (%) $\uparrow$", fontsize='x-large')
    axs[i].set_xlim(-16, 16)
    axs[i].set_xticks(list(range(-15, 16, 5)))

    # Set the font size of x and y ticks
    axs[i].tick_params(axis='both', which='major', labelsize='large')  # Set major ticks font size
    
    props = ["1 prop", "2 props", "4 props", "6 props"]
    axs[i].text(0.05, 0.95, f'{props[i]}', transform=axs[i].transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(facecolor='darkgray', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.5'), zorder=10)


# Finetuned data
q = dff.groupby(["property","rq","ploss"]).sum()["violations"].reset_index()
v_dff = q.pivot(index=["property","rq"], columns="ploss", values="violations")
v_dff["diff"] = (v_dff[False] - v_dff[True])
v_dff["diff_percentage"] = v_dff.apply(lambda row: row["diff"] / row[False] * 100 if row[False] > 0 else row[True] * 100, axis=1)

q = dff.groupby(["property","rq","ploss"]).sum()[["val_steering_loss","val_acceleration_loss"]].reset_index()
loss_dff = q.pivot(index=["property","rq"], columns="ploss", values=["val_steering_loss","val_acceleration_loss"])
loss_dff["diff_steering"] = (loss_dff["val_steering_loss"][False] - loss_dff["val_steering_loss"][True])
loss_dff["diff_steering_percentage"] = loss_dff.apply(lambda row: row["diff_steering"] / row["val_steering_loss"][False] * 100 if row["val_steering_loss"][False] > 0 else row["val_steering_loss"][True] * 100, axis=1)
loss_dff["diff_acceleration"] = (loss_dff["val_acceleration_loss"][False] - loss_dff["val_acceleration_loss"][True])
loss_dff["diff_acceleration_percentage"] = loss_dff.apply(lambda row: row["diff_acceleration"] / row["val_acceleration_loss"][False] * 100 if row["val_acceleration_loss"][False] > 0 else row["val_acceleration_loss"][True] * 100, axis=1)

palette = sns.color_palette("muted", 6)
all_properties = ["leftmost_lane", "rightmost_lane", "green_light", "yel_red_light", "entities_within_10m", "stopped_no_reason25"]
color_dict = {prop: color for prop, color in zip(all_properties, palette)}

# Plotting horizontal lines for violations difference percentage
for index, row in v_dff.iterrows():
    if index[1] == 1:
        # Steering property
        if index[0] in ["leftmost_lane", "rightmost_lane"]:
            axs[4].axhline(y=row['diff_percentage'], label=index[0], linestyle='-', color="black", alpha=0.5, linewidth=1)
            # Determine label position
            label_y = row['diff_percentage'] + 5 if row['diff_percentage'] < axs[4].get_ylim()[1] - 10 else row['diff_percentage'] - 5
            axs[4].text(0, label_y, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
        # Acceleration property
        else:
            axs[4].axhline(y=row['diff_percentage'], label=index[0], linestyle=(0, (5, 5)), color="black", alpha=0.5, linewidth=1)
            label_y = row['diff_percentage'] + 5 if row['diff_percentage'] < axs[4].get_ylim()[1] - 10 else row['diff_percentage'] - 5
            flag = False
            if index[0] == "entities_within_10m":
                axs[4].text(0, label_y - 3, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                flag = True
            if index[0] == "green_light":
                axs[4].text(0, label_y - 3, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                flag = True
            if index[0] == "stopped_no_reason25":
                axs[4].text(0, label_y, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')
                flag = True
            if not flag:
                axs[4].text(0, label_y, f"{row[False]:.0f} $\\to$ {row[True]:.0f}", color=color_dict[index[0]], fontsize=10, ha='center', va='bottom' if label_y > row['diff_percentage'] else 'top')

        steering_diff = loss_dff.loc[index]["diff_steering_percentage"].item()
        acceleration_diff = loss_dff.loc[index]["diff_acceleration_percentage"].item()
        axs[4].plot(steering_diff, row["diff_percentage"], marker="o", color=color_dict[index[0]], markersize=10)
        axs[4].plot(acceleration_diff, row["diff_percentage"], marker="x", color=color_dict[index[0]], markersize=10)

# Plot vertical and horizontal lines at 0
axs[4].axvline(x=0, color='black', linestyle=(0, (1, 5)), alpha=1, zorder=0)
axs[4].axhline(y=0, color='black', linestyle=(0, (1, 5)), alpha=1, zorder=0)

# axs[4].set_ylabel(r"Violation improvement (%) $\uparrow$", fontsize='x-large')
axs[4].set_ylim(-10, 110)
axs[4].set_xlabel(r"Loss improvement (%) $\uparrow$", fontsize='x-large')
axs[4].set_xlim(-26, 26)
axs[4].set_xticks(list(range(-25, 26, 5)))

# Set the font size of x and y ticks
axs[4].tick_params(axis='both', which='major', labelsize='large')

# Get existing legend
old_handles, old_labels = axs[4].get_legend_handles_labels()
# Rename the labels
name_map = {
    "leftmost_lane": "$\\phi_6$ - NoSteerLeftOutRoad",
    "rightmost_lane": "$\\phi_5$ - NoSteerRightOutRoad",
    "green_light": "$\\phi_4$ - AccelerateForGreen",
    "yel_red_light": "$\\phi_2$ - StopForYellowRed",
    "entities_within_10m": "$\\phi_1$ - StopToAvoidCollision",
    "stopped_no_reason25": "$\\phi_3$ - NoStopForNoReason"
}
new_labels = [name_map[label] for label in old_labels]
# Change handle colors
custom_handles = []
for handle, label in zip(old_handles, old_labels):
    custom_handle = mlines.Line2D([0], [0], color=color_dict[label], alpha=1, linewidth=2, linestyle=handle.get_linestyle())
    custom_handles.append(custom_handle)

# Reindex custom_handles and new_labels to match the order of the properties
idx_order = [0, 5, 4, 1, 3, 2]
custom_handles = [custom_handles[i] for i in idx_order]
new_labels = [new_labels[i] for i in idx_order]

# Create proxy artists for the custom labels
steer_loss_diff_label = plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', linestyle='None', markersize=10)
acc_loss_diff_label = plt.Line2D([0], [0], marker='x', color='black', markerfacecolor='black', linestyle='None', markersize=10)
# Add the custom labels to the legend
handles = custom_handles + [steer_loss_diff_label, acc_loss_diff_label]
labels = new_labels + ['Steering loss diff', 'Acceleration loss diff']

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize='x-large')

axs[4].text(0.05, 0.95, f'Ft 1 prop', transform=axs[4].transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(facecolor='darkgray', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.5'), zorder=10)

fig.subplots_adjust(bottom=0.27)

# fig.savefig("./controlled_experiment/fig_3.pdf", format="pdf", bbox_inches="tight")
fig.savefig("./controlled_experiment/fig_3.png", format="png", bbox_inches="tight")