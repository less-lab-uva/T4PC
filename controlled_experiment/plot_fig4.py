import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the JSON file
with open("./controlled_experiment/all_data.json", "r") as file:
    all_data = json.load(file)

def get_times(models_data):
    batch_times = {}
    sum_batch_times = {}
    batch_times_without_loss = {}
    sum_batch_times_without_loss = {}
    batch_main_loss_times = {}
    sum_batch_main_loss_times = {}
    batch_prop_loss_times = {}
    sum_batch_prop_loss_times = {}
    batch_other_times = {}
    sum_batch_other_times = {}
    for epoch in range(15):
        batch_times[epoch] = []
        batch_times_without_loss[epoch] = []
        batch_main_loss_times[epoch] = []
        batch_prop_loss_times[epoch] = []
        batch_other_times[epoch] = []
        sum_batch_times[epoch] = []
        sum_batch_times_without_loss[epoch] = []
        sum_batch_main_loss_times[epoch] = []
        sum_batch_prop_loss_times[epoch] = []
        sum_batch_other_times[epoch] = []

    for model in models_data:
        for epoch in range(15):
            batch_times[epoch].extend(model[str(epoch)]["batch_times"])
            batch_main_loss_times[epoch].extend(model[str(epoch)]["batch_main_loss_times"])
            batch_prop_loss_times[epoch].extend(model[str(epoch)]["batch_prop_loss_times"])
            sum_batch_times[epoch].append(sum(model[str(epoch)]["batch_times"]))
            sum_batch_main_loss_times[epoch].append(sum(model[str(epoch)]["batch_main_loss_times"]))
            sum_batch_prop_loss_times[epoch].append(sum(model[str(epoch)]["batch_prop_loss_times"]))

            # Compute other times
            other_times = []
            times_without_loss = []
            for i in range(len(model[str(epoch)]["batch_times"])):
                other_times.append(model[str(epoch)]["batch_times"][i] - model[str(epoch)]["batch_main_loss_times"][i] - model[str(epoch)]["batch_prop_loss_times"][i])
                times_without_loss.append(model[str(epoch)]["batch_times"][i] - model[str(epoch)]["batch_prop_loss_times"][i])
            batch_other_times[epoch].extend(other_times)
            batch_times_without_loss[epoch].extend(times_without_loss)
            sum_batch_other_times[epoch].append(sum(other_times))
            sum_batch_times_without_loss[epoch].append(sum(times_without_loss))

    epoch_times = []
    epoch_times_without_loss = []
    epoch_main_loss_times = []
    epoch_prop_loss_times = []
    epoch_other_times = []
    for epoch in range(15):
        epoch_times.extend(batch_times[epoch])
        epoch_times_without_loss.extend(batch_times_without_loss[epoch])
        epoch_main_loss_times.extend(batch_main_loss_times[epoch])
        epoch_prop_loss_times.extend(batch_prop_loss_times[epoch])
        epoch_other_times.extend(batch_other_times[epoch])
    
    sum_epoch_times = np.array(sum_batch_times[0])
    sum_epoch_times_without_loss = np.array(sum_batch_times_without_loss[0])
    sum_epoch_main_loss_times = np.array(sum_batch_main_loss_times[0])
    sum_epoch_prop_loss_times = np.array(sum_batch_prop_loss_times[0])
    sum_epoch_other_times = np.array(sum_batch_other_times[0])
    for epoch in range(1,15):
        sum_epoch_times += np.array(sum_batch_times[epoch])
        sum_epoch_times_without_loss += np.array(sum_batch_times_without_loss[epoch])
        sum_epoch_main_loss_times += np.array(sum_batch_main_loss_times[epoch])
        sum_epoch_prop_loss_times += np.array(sum_batch_prop_loss_times[epoch])
        sum_epoch_other_times += np.array(sum_batch_other_times[epoch])

    return {
        "batch_times": batch_times,
        "batch_main_loss_times": batch_main_loss_times,
        "batch_prop_loss_times": batch_prop_loss_times,
        "batch_other_times": batch_other_times,
        "epoch_times": epoch_times,
        "epoch_times_without_loss": epoch_times_without_loss,
        "epoch_main_loss_times": epoch_main_loss_times,
        "epoch_prop_loss_times": epoch_prop_loss_times,
        "epoch_other_times": epoch_other_times,
        "sum_batch_times": sum_batch_times,
        "sum_batch_main_loss_times": sum_batch_main_loss_times,
        "sum_batch_prop_loss_times": sum_batch_prop_loss_times,
        "sum_batch_other_times": sum_batch_other_times,
        "sum_batch_times_without_loss": sum_batch_times_without_loss,
        "sum_epoch_times": sum_epoch_times.tolist(),
        "sum_epoch_times_without_loss": sum_epoch_times_without_loss.tolist(),
        "sum_epoch_main_loss_times": sum_epoch_main_loss_times.tolist(),
        "sum_epoch_prop_loss_times": sum_epoch_prop_loss_times.tolist(),
        "sum_epoch_other_times": sum_epoch_other_times.tolist()
    }

baseline_times = get_times(all_data['0'])
prop1_times = get_times(all_data['1'])
prop2_times = get_times(all_data['2'])
prop4_times = get_times(all_data['4'])
prop6_times = get_times(all_data['6'])

data_to_plot = [
    baseline_times["epoch_times_without_loss"],
    prop1_times["epoch_times"],
    prop2_times["epoch_times"],
    prop4_times["epoch_times"],
    prop6_times["epoch_times"]
]

# Plotting with custom y-axis breaks using subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})

labels = ["Baseline", "1 Prop", "2 Props", "4 Props", "6 Props"]

# Define y-axis limits for the two subplots (original left y-axis)
ax1.set_ylim(0.6, 0.626)
ax2.set_ylim(0.225, 0.25)

# Plot the left y-axis data (batch times)
bx1 = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False, widths=0.2)
bx2 = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False, widths=0.2)

# Annotate median values
for i, line in enumerate(bx1['medians']):
    x, y = line.get_xdata()[1], line.get_ydata()[1]
    if i == 0:
        ax2.text(x - 0.5, y, f'{y*1000:.1f}', verticalalignment='center')
    else:
        ax1.text(x - 0.5, y, f'{y*1000:.1f}', verticalalignment='center')

# Create secondary y-axes
ax1_right = ax1.twinx()
ax2_right = ax2.twinx()

# Define y-axis limits for the right-side (different scale)
ax1_right.set_ylim(195, 220)
ax2_right.set_ylim(70, 95)

# Plot right y-axis data (epoch times)
right_data = [
    [x / 60 for x in baseline_times["sum_epoch_times_without_loss"]],
    [x / 60 for x in prop1_times["sum_epoch_times"]],
    [x / 60 for x in prop2_times["sum_epoch_times"]],
    [x / 60 for x in prop4_times["sum_epoch_times"]],
    [x / 60 for x in prop6_times["sum_epoch_times"]]
]

positions = [i + 1.17 for i in range(len(right_data))]  # Slight shift right

# Plot the median of right_data with a 'o'
medians = [np.median(dataset) for dataset in right_data]
for idx, (pos, median) in enumerate(zip(positions, medians)):
    if idx == 0:
        ax2_right.plot(pos, median, "o", markersize=10, color="black")
        ax2_right.text(pos + 0.07, median, f'{median:.1f}', verticalalignment='center')
    else:
        ax1_right.plot(pos, median, "o", markersize=10, color="black")
        ax1_right.text(pos + 0.07, median, f'{median:.1f}', verticalalignment='center')

# Hide the spines between the left subplots
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax2.tick_params(labelbottom=True)

# Hide spines between the right subplots
ax1_right.spines['bottom'].set_visible(False)
ax2_right.spines['top'].set_visible(False)
ax1_right.tick_params(labeltop=False, bottom=False)
ax2_right.tick_params(labelbottom=False)

# Add diagonal lines for break (left side)
d = 0.015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Add diagonal lines for break (right side)
kwargs_right = dict(transform=ax1_right.transAxes, color='k', clip_on=False)
ax1_right.plot((-d, +d), (-d, +d), **kwargs_right)
ax1_right.plot((1 - d, 1 + d), (-d, +d), **kwargs_right)
kwargs_right.update(transform=ax2_right.transAxes)
ax2_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_right)
ax2_right.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_right)

# Set labels
fig.text(0.04, 0.5, 'Batch Times (in miliseconds)', va='center', rotation='vertical', fontsize=12)
fig.text(0.95, 0.5, '15 Epoch Training Times (in minutes)', va='center', rotation='vertical', fontsize=12)
fig.text(0.5, 0.04, 'Model Type', ha='center', fontsize=12)

fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

# fig.savefig("./controlled_experiment/fig4.pdf", format="pdf", bbox_inches="tight")
fig.savefig("./controlled_experiment/fig4.png", format="png", bbox_inches="tight")