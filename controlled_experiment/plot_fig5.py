import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from collections import OrderedDict

# Hardcoded color mapping for known violations
cmap = get_cmap("tab10")  # Standard Matplotlib color cycle
violation_color_map = {
    'violation_count_leftmost_lane': cmap(0),  # Red
    'violation_count_rightmost_lane': cmap(1),  # Blue
    'violation_count_green_light': cmap(2),  # Green
    'violation_count_yel_red_light': cmap(3),  # Yellow
    'violation_count_entities_within_10m': cmap(4),  # Purple
    'violation_count_stopped_no_reason25': cmap(5)  # Orange
}
name_map = {
    "leftmost_lane": "$\\phi_6$ - NoSteerLeftOutRoad",
    "rightmost_lane": "$\\phi_5$ - NoSteerRightOutRoad",
    "green_light": "$\\phi_4$ - AccelerateForGreen",
    "yel_red_light": "$\\phi_2$ - StopForYellowRed",
    "entities_within_10m": "$\\phi_1$ - StopToAvoidCollision",
    "stopped_no_reason25": "$\\phi_3$ - NoStopForNoReason"
}

validation_data = json.load(open('./controlled_experiment/validation_logs_1.json'))
validation_data2 = json.load(open('./controlled_experiment/validation_logs_2.json'))
validation_data3 = json.load(open('./controlled_experiment/validation_logs_3.json'))

def plot_prop_violations_with_confidence_interval(df1, df2, df3):
    # Get the maximum number of epochs among the three dataframes
    max_epochs = min(len(df1), len(df2), len(df3))

    # Combine the three dataframes into a single dictionary for processing
    combined_data = {epoch: {
        key: [df1[epoch]['prop_violation_count_dict'][key],
              df2[epoch]['prop_violation_count_dict'][key],
              df3[epoch]['prop_violation_count_dict'][key]]
        for key in df1[epoch]['prop_violation_count_dict'].keys()
    } for epoch in df1.keys() if int(epoch) < max_epochs}

    # Extract epochs and property violation keys
    epochs = sorted(combined_data.keys(), key=int)
    prop_violation_keys = df1[epochs[0]]['prop_violation_count_dict'].keys()

    # Prepare property violation values and compute confidence intervals
    prop_violation_values = {}
    prop_violation_cis = {}
    total_violations = []  # To store total violations per epoch
    total_cis = []  # To store confidence intervals for total violations

    for key in prop_violation_keys:
        values = np.array([[combined_data[epoch][key][i] for epoch in epochs] for i in range(3)])
        mean_values = values.mean(axis=0)
        ci_values = 1.96 * values.std(axis=0) / np.sqrt(3)  # 95% confidence interval
        prop_violation_values[key] = mean_values
        prop_violation_cis[key] = ci_values

    # Calculate total violations for each epoch
    for epoch_idx, epoch in enumerate(epochs):
        # Sum violations across all datasets for the current epoch
        epoch_totals = [
            sum(combined_data[epoch][key][i] for key in prop_violation_keys)
            for i in range(3)  # Iterate over df1, df2, df3
        ]
        
        # Compute mean, standard deviation, and confidence interval for the total violations
        mean_total = np.mean(epoch_totals)
        std_total = np.std(epoch_totals)
        ci_total = 2*std_total
        
        # Append the results
        total_violations.append(mean_total)
        total_cis.append(ci_total)

    # Find the epoch with the minimum total violations
    min_epoch_idx = np.argmin(total_violations)
    min_epoch = epochs[min_epoch_idx]
    min_violation = total_violations[min_epoch_idx]

    # Plot the data
    plt.figure(figsize=(24, 6))
    # Re-order the dictionary keys as needed
    ordered_keys = ['validation_violation_count_entities_within_10m', 'validation_violation_count_yel_red_light',
                    'validation_violation_count_stopped_no_reason25', 'validation_violation_count_green_light',
                    'validation_violation_count_rightmost_lane', 'validation_violation_count_leftmost_lane']
    prop_violation_values = OrderedDict((key, prop_violation_values[key]) for key in ordered_keys if key in prop_violation_values)

    for key, mean_values in prop_violation_values.items():
        # Normalize the key by removing prefixes like 'test_', 'train_', etc.
        normalized_key = key.split('_', 1)[-1]
        # Use the hardcoded color if available, otherwise default to black
        color = violation_color_map.get(normalized_key, 'black')
        ci_values = prop_violation_cis[key]
        lower_bound = np.maximum(mean_values - ci_values, 0)  # Ensure lower bound is not less than 0
        updated_label_name = name_map[key.split("count_")[1]]
        plt.plot(epochs, mean_values, marker='o', label=updated_label_name, color=color)
        plt.fill_between(epochs, lower_bound, mean_values + ci_values, color=color, alpha=0.2)

    # Plot the total violations with confidence intervals
    plt.plot(epochs, total_violations, marker='o', label='Total Violations', color='black', linewidth=2)
    plt.fill_between(epochs, 
                     np.array(total_violations) - np.array(total_cis), 
                     np.array(total_violations) + np.array(total_cis), 
                     color='gray', alpha=0.3)

    # Add a vertical line at epoch 0
    plt.axvline(x=0, linestyle='--', color='black', alpha=0.7)
    plt.text(0.4, plt.ylim()[1] * 0.95, f'E0: {total_violations[0]:,.0f}', rotation=0, verticalalignment='top', color='black', fontsize=16)
    # Add a vertical line at epoch 15
    plt.axvline(x=15, linestyle='--', color='black', alpha=0.7)
    plt.text(15 * 1.035, plt.ylim()[1] * 0.95, f'E15: {total_violations[15]:.0f}', rotation=0, verticalalignment='top', color='black', fontsize=16)
    # Add a vertical line at the epoch with the minimum total violations
    plt.axvline(x=int(min_epoch), linestyle='--', color='black', alpha=0.7)
    plt.text(int(min_epoch) * 1.01, plt.ylim()[1] * 0.95, f'E{min_epoch}: {min_violation:.0f}', rotation=0, verticalalignment='top', color='black', fontsize=16)

    plt.xlabel('Epochs')
    plt.ylabel('Property Violations')
    plt.xticks(range(0, len(epochs), 5))  # Set x-axis ticks every 5 epochs
    plt.legend(loc='upper right', fontsize=16)
    plt.grid(True)
    # plt.savefig("./controlled_experiment/fig_5.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("./controlled_experiment/fig_5.png", format="png", bbox_inches="tight")

plot_prop_violations_with_confidence_interval(validation_data, validation_data2, validation_data3)