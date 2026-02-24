from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class LoanProcessPreprocessor():
    def __init__(self, dataset_params, data_train = pd.DataFrame(), data_train_val = pd.DataFrame()):
        #GENERAL params
        self.dataset_params = dataset_params

        #PREPROCESSING params
        self.case_cols = dataset_params["case_cols"]
        self.case_cols_encoded = []
        self.event_cols = dataset_params["event_cols"]
        self.event_cols_encoded = []
        self.cat_cols = dataset_params["cat_cols"]
        self.scale_cols = dataset_params["scale_cols"]
        self.nr_treatment_columns = dataset_params["intervention_info"]["action_width"] if dataset_params["intervention_info"]["action_width"] > 2 else 1
        if not (data_train.empty and data_train_val.empty):
            self.set_data(data_train, data_train_val)

    
    def set_data(self, data_train, data_train_val):
        self.data_train = data_train
        self.data_train_val = data_train_val
        self.max_process_len = 0
        max_process_len_train = data_train.groupby(["case_nr"]).size().max()
        max_process_len_train_val = data_train_val.groupby(["case_nr"]).size().max()
        self.max_process_len = max(max_process_len_train, max_process_len_train_val)

    

    def preprocess_for_kmeans(self):
        train_prep_k_means = pd.DataFrame()
        # create an activity dictionary with each unique activity in the event log as a key
        unique_activities = self.data_train["activity"].unique()
        activity_count_dict_max = {str(activity) + "_count": 0 for activity in unique_activities}
        activity_pos_dict_max = {str(activity) + "_pos": 0 for activity in unique_activities}
        grouped_train_RCT = self.data_train.groupby("case_nr")

        for case_nr, group in grouped_train_RCT:
            if case_nr == 10000:
                break
            activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
            activity_pos_dict = {str(activity) + "_pos": 0 for activity in unique_activities}
            for current_pos, (index, row) in enumerate(group.iterrows(), start=1):
                activity_count_dict[row["activity"] + "_count"] += 1
                # grab the position of the activity in the case length (starting from 1), so check which position the current row has in the case
                activity_pos_dict[row["activity"] + "_pos"] = current_pos

                # put both dictionaries in a dataframe
                prefix = pd.DataFrame({**activity_count_dict, **activity_pos_dict}, index=[0])

                for scale_col in self.dataset_params["scale_cols"]:
                    if scale_col == "outcome" and row["activity"] != "cancel_application" and row["activity"] != "receive_acceptance":
                        prefix[scale_col] = 0
                    else:
                        # check whether NaN or not
                        if pd.isna(row[scale_col]):
                            prefix[scale_col] = -1
                        else:
                            prefix[scale_col] = row[scale_col]
                
                # add prefix to the dataframe
                train_prep_k_means = pd.concat([train_prep_k_means, prefix], axis=0, ignore_index=True)

        # divide the count of each activity by the maximum count of that activity
        # divide the position of each activity by the maximum position of that activity
        for activity in unique_activities:
            activity_count_dict_max[activity + "_count"] = train_prep_k_means[activity + "_count"].max()
            activity_pos_dict_max[activity + "_pos"] = train_prep_k_means[activity + "_pos"].max()

        # grab the all around maximum of the activity counts and positions
        activity_count_max = max(activity_count_dict_max.values())
        activity_pos_max = max(activity_pos_dict_max.values())

        for activity in unique_activities:
            train_prep_k_means[activity + "_count"] = train_prep_k_means[activity + "_count"] / activity_count_max
            train_prep_k_means[activity + "_pos"] = train_prep_k_means[activity + "_pos"] / activity_pos_max

        print(train_prep_k_means, 'train_prep_k_means')

        # Standardize all columns
        scaler_outcome = StandardScaler()
        # Get non-zero values from the column
        non_zero_outcomes = train_prep_k_means['outcome'][train_prep_k_means['outcome'] != 0].values.reshape(-1, 1)
        # Fit and transform only non-zero values
        scaled_non_zero_outcomes = scaler_outcome.fit_transform(non_zero_outcomes)
        # Replace the non-zero values in the original DataFrame with scaled values
        train_prep_k_means.loc[train_prep_k_means['outcome'] != 0, 'outcome'] = scaled_non_zero_outcomes

        scaler_without_outcome = StandardScaler()
        columns_to_scale_without_outcome = train_prep_k_means.columns.tolist()
        columns_to_scale_without_outcome.remove("outcome")
        print(columns_to_scale_without_outcome, 'columns_to_scale_without_outcome')
        train_prep_k_means_scaled = pd.DataFrame(scaler_without_outcome.fit_transform(train_prep_k_means[columns_to_scale_without_outcome]))

        # concat the scaled columns with the outcome column, but make sure the column names are still present
        train_prep_k_means_scaled.columns = columns_to_scale_without_outcome
        train_prep_k_means = pd.concat([train_prep_k_means_scaled, train_prep_k_means[["outcome"]]], axis=1)

        prep_utils = {"scaler_without_outcome": scaler_without_outcome, 
                           "scale_cols": self.dataset_params["scale_cols"],
                           "activity_count_max": activity_count_max,
                           "activity_pos_max": activity_pos_max,
                           "scaler_outcome": scaler_outcome,
                           "unique_activities": unique_activities}
        
        print(train_prep_k_means, 'train_prep_k_means')

        return train_prep_k_means, prep_utils
    

    def kmeans_clustering(self, train_prep_k_means, k_values = [50, 100, 500], random_seed = 42):
        silhouette_scores = []
        best_k = None
        best_silhouette_score = -1
        best_km = None

        # Iterate through different K values
        for k in k_values:
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_seed)
            km.fit(train_prep_k_means)
            
            # Calculate the average silhouette score
            silhouette_avg = silhouette_score(train_prep_k_means, km.labels_)  
            silhouette_scores.append(silhouette_avg)
            
            # Update the best K value if a higher silhouette score is found
            if silhouette_avg > best_silhouette_score:
                best_k = k
                best_silhouette_score = silhouette_avg
                best_km = km

        # Output the best K value and corresponding Silhouette Score
        print("Best K value:", best_k)
        print("Best Silhouette Score:", best_silhouette_score)
        plt.plot(k_values, silhouette_scores)
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs. Number of Clusters")

        return best_km, best_k, best_silhouette_score


    def preprocess_sample_RL(self, data_sample, data_t, prep_utils, device, treat_len = 1, data_full_action = None):
        # Case and event
        _, data_encoded_sample = self.one_hot_encode_columns(data = data_sample, oh_encoder_dict = prep_utils["oh_encoder_dict_train"])
        _, data_scaled_sample = self.scale_columns(data = data_encoded_sample, scaler_dict = prep_utils["scaler_dict_train"])
        data_fill_sample = self.handle_missing_values(data = data_scaled_sample)
        X_case = torch.unsqueeze(prep_utils["pad_case"], 0)
        X_event = torch.unsqueeze(torch.unsqueeze(prep_utils["pad_event"], 0), 2).repeat(1, 1, prep_utils["max_process_len"])
        X_case[0,:] = torch.Tensor(data_fill_sample[self.case_cols_encoded].values)[0,:]
        index_contact_headquarters = self.event_cols_encoded.index('activity_contact_headquarters')
        X_event[0, :, -len(data_fill_sample):] = torch.Tensor(np.transpose(data_fill_sample[self.event_cols_encoded].values))
        # Treatment
        X_t = [[0] * treat_len] * prep_utils["max_process_len"]
        X_t[-len(data_t):] = data_t
        X_t = torch.unsqueeze(torch.Tensor(X_t), 0)
        # convert 0/1 to boolean
        X_t = X_t.bool()
        # Send to device
        X_case = X_case.to(device)
        X_event = X_event.to(device)
        X_t = X_t.to(device)
        if data_full_action is not None:
            full_action_prep = torch.tensor([data_full_action], device=device, dtype=torch.long)
        else:
            full_action_prep = None
        return X_case, X_event, X_t, full_action_prep
    

    def preprocess_kmeans_RL(self, data_sample, reward_preproc, prep_utils, kmeans):
        # drop case_nr and outcome
        unique_activities = prep_utils["unique_activities"]
        activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
        activity_pos_dict = {str(activity) + "_pos": 0 for activity in unique_activities}
        for current_pos, (index, row) in enumerate(data_sample.iterrows(), start=1):
            activity_count_dict[row["activity"] + "_count"] += 1
            # grab the position of the activity in the case length (starting from 1), so check which position the current row has in the case
            activity_pos_dict[row["activity"] + "_pos"] = current_pos
        data_sample_activities = pd.DataFrame({**activity_count_dict, **activity_pos_dict}, index=[0])

        # get the scale columns from original data but without case_nr, outcome and activity
        data_sample_other_cols = data_sample[prep_utils["scale_cols"]]
        data_sample_other_cols.drop(columns=["outcome"], inplace=True)
        # just get the last row as a dataframe
        data_sample_other_cols = data_sample_other_cols.iloc[[-1]]
        data_sample_other_cols.fillna(-1, inplace=True)
        data_sample_preproc = data_sample_activities
        for col in data_sample_other_cols.columns:
            data_sample_preproc[col] = data_sample_other_cols[col].values
        columns = data_sample_preproc.columns
        data_sample_preproc = prep_utils["scaler_without_outcome"].transform(data_sample_preproc)
        data_sample_preproc = pd.DataFrame(data_sample_preproc, columns=columns)

        for activity in unique_activities:
            data_sample_preproc[activity + "_count"] = data_sample_preproc[activity + "_count"] / prep_utils["activity_count_max"]
            data_sample_preproc[activity + "_pos"] = data_sample_preproc[activity + "_pos"] / prep_utils["activity_pos_max"]

        # find the cluster of the data_sample_preproc + reward
        data_sample_preproc["outcome"] = reward_preproc

        kmeans_result = kmeans.predict(data_sample_preproc)
        last_activity = data_sample["activity"].iloc[-1]
        state_preproc = (kmeans_result[0], last_activity)

        return state_preproc    


    def preprocess_reward_kmeans_RL(self, reward, prep_utils, device):
        # just output a number, not a tensor or anything
        outcome_scaler = prep_utils["scaler_outcome"]
        reward_peproc = outcome_scaler.transform([[reward]])
        return reward_peproc[0][0]