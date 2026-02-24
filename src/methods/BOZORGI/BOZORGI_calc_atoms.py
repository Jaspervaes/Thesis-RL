data_folder = "data"
results_folder = "res"
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)
sys.path.append(path + "\\SimBank")

from src.utils.tools import save_data, load_data
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from SimBank.confounding_level import set_delta














intervention_name = ["time_contact_HQ"]
train_size = 10000
biased = sys.argv[1] == "True"
dataset_params = load_data(os.path.join(os.getcwd(), data_folder, "loan_log_" + str(intervention_name) + "_" + str(train_size) + "_dataset_params_BOZORGI_RealCause"))
data_path = os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
if biased:
    bias_path = '_biased'
    delta = 0.999
    train_normal = load_data(data_path + "_train_normal" + "_BOZORGI_RealCause")
    train_normal_val = load_data(data_path + "_train_normal_val" + "_BOZORGI_RealCause")
    RCT = load_data(data_path + "_train_RCT" + "_BOZORGI_RealCause")
    RCT_val = load_data(data_path + "_train_RCT_val" + "_BOZORGI_RealCause")
    train_RCT = set_delta(data=train_normal, data_RCT=RCT, delta=delta)
    train_RCT_val = set_delta(data=train_normal_val, data_RCT=RCT_val, delta=delta)
else:
    bias_path = ''
    train_RCT = load_data(data_path + "_train_RCT" + "_BOZORGI_RealCause")
    train_RCT_val = load_data(data_path + "_train_RCT_val" + "_BOZORGI_RealCause")

int_dataset_params = []
for int_index, int_name in enumerate(intervention_name):
    params = deepcopy(dataset_params)
    for key, value in params["intervention_info"].items():
        if isinstance(value, list):
            params["intervention_info"][key] = value[int_index]
    int_dataset_params.append(params)













cutoff = -249.9999
# plot the outcome distribution per case_nr (group per case_nr and plot the outcome)
max_outcomes = train_RCT.groupby('case_nr')['outcome'].max().reset_index()
plt.figure(figsize=(10, 6))
plt.hist(max_outcomes["outcome"], bins=500)
plt.show()
# plot the outcome distribution per case_nr (group per case_nr and plot the outcome) and do this only for negative outcomes
max_neg_outcomes = max_outcomes[max_outcomes["outcome"] <= cutoff]
plt.figure(figsize=(10, 6))
plt.hist(max_neg_outcomes["outcome"], bins=500)
plt.show()
















# Sort the DataFrame by 'values'
df = max_neg_outcomes.sort_values(by='outcome').reset_index(drop=True)

# Initialize variables
groups = []
current_group = []
last_value = None
threshold = 0

# Grouping values within 100 of each other
for value in df['outcome']:
    if last_value is None:
        last_value = value
        current_group.append(value)
    elif abs(value - last_value) <= threshold:
        current_group.append(value)
    else:
        groups.append(current_group)
        current_group = [value]
        last_value = value

# Append the last group
if current_group:
    groups.append(current_group)

# Filter out groups that have fewer than 1 elements IMPORTANT NOTE: IMPORTANT TO KEEP THIS AT ONE, THEN THE BINS CAN BE BASED ON THE OUTCOME PER CASE_NR, OTHERWISE THE BINS SHOULD BE MADE AFTER RETAINING AND SCALING THE DATA
filtered_groups = [group for group in groups if len(group) >= 5]

# Compute the average for each valid group
grouped_averages = [sum(group) / len(group) for group in filtered_groups]

# Also compute the standard deviation for each valid group
grouped_stds = [np.std(group) for group in filtered_groups]

#if a group has a standard dev lower than 1, than just take the most frequent value, and the stdev should be set to 0
for index, std in enumerate(grouped_stds):
    if std < 1:
        grouped_averages[index] = max(set(filtered_groups[index]), key=filtered_groups[index].count)
        grouped_stds[index] = 0

# Create a DataFrame to show the results
result_df = pd.DataFrame(grouped_averages, columns=['Average'])

print("Filtered Groups with >= 1 elements:")
for group in filtered_groups:
    print(group)

print("\nAverages of those groups:")
print(result_df)
# make a list of the averages
averages = result_df['Average'].tolist()

print("\nStandard deviations of those groups:")
print(grouped_stds)

# plot the points of result_df on the histogram as vertical lines
plt.figure(figsize=(10, 6))
plt.hist(max_outcomes["outcome"], bins=500)
for value in result_df['Average']:
    plt.axvline(value, color='r')
plt.show()

save_data(averages, os.path.join(os.getcwd(), data_folder, "grouped_averages_" + str(intervention_name) + "_" + str(train_size) + "_BOZORGI_RealCause" + bias_path))
save_data(threshold, os.path.join(os.getcwd(), data_folder, "bin_width_" + str(intervention_name) + "_" + str(train_size) + "_BOZORGI_RealCause" + bias_path))
save_data(grouped_stds, os.path.join(os.getcwd(), data_folder, "grouped_stds_" + str(intervention_name) + "_" + str(train_size) + "_BOZORGI_RealCause" + bias_path))
split_list = [-249.99999]
save_data(split_list, os.path.join(os.getcwd(), data_folder, "split_list_" + str(intervention_name) + "_" + str(train_size) + "_BOZORGI_RealCause" + bias_path))