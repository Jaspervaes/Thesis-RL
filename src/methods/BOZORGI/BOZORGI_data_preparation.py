import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from copy import deepcopy

class LoanProcessPreprocessor():
    def __init__(self, dataset_params, data_train = pd.DataFrame(), data_to_generate = pd.DataFrame(), data_to_generate_val = pd.DataFrame(), eps_to_add_outcome = 10, big_data = True, atoms = None, stdev_atoms = None, bin_with = None, split_list = None, data_size = 10000):
        #GENERAL params
        self.dataset_params = dataset_params
        self.eps_to_add_outcome = eps_to_add_outcome
        self.big_data = big_data
        self.atoms = atoms
        self.stdev_atoms = stdev_atoms
        self.bin_with = bin_with
        self.split_list = split_list
        self.data_size = data_size

        #PREPROCESSING params
        self.case_cols = dataset_params["case_cols"]
        self.case_cols_encoded = []
        self.event_cols = dataset_params["event_cols"]
        self.event_cols_encoded = []
        self.cat_cols = dataset_params["cat_cols"]
        self.scale_cols = dataset_params["scale_cols"]
        self.last_state_cols = ["elapsed_time", "cum_cost"]

        if not (data_train.empty and data_to_generate.empty and data_to_generate_val.empty):
            self.set_data(data_train, data_to_generate, data_to_generate_val)

    
    def set_data(self, data_train, data_to_generate, data_to_generate_val):
        self.data_train = data_train
        self.data_to_generate = data_to_generate
        self.data_to_generate_val = data_to_generate_val
        self.max_process_len = 0
        max_process_len_train = data_train.groupby(["case_nr"]).size().max()
        max_process_len_to_generate = data_to_generate.groupby(["case_nr"]).size().max()
        max_process_len_to_generate_val = data_to_generate_val.groupby(["case_nr"]).size().max()
        self.max_process_len = max(max_process_len_train, max_process_len_to_generate, max_process_len_to_generate_val)

    
    def preprocess(self, only_preprocess_realcause_data = False):
        self.data_treat = self.add_treatment_column(self.data_train)
        print("data_treat", self.data_treat[:20])
        self.data_pref = self.create_and_encode_prefixes(self.data_treat)
        self.data_prep, self.scaler_dict, min_outcome_scaled, max_outcome_scaled, atoms_scaled, bin_width_scaled, stdev_atoms_scaled, split_list_scaled = self.scale(self.data_pref)
        self.prep_utils = {"max_process_len": self.max_process_len, 
                           "scaler_dict": self.scaler_dict, 
                           "min_outcome_scaled": min_outcome_scaled, 
                           "max_outcome_scaled": max_outcome_scaled, 
                           "atoms_scaled": atoms_scaled, 
                           "bin_width_scaled": bin_width_scaled,
                           "stdev_atoms_scaled": stdev_atoms_scaled,
                           "split_list_scaled": split_list_scaled,
                           "column_names": self.data_pref.columns}
        
        if only_preprocess_realcause_data:
            return self.data_prep, self.prep_utils, None, None
        
        self.data_treat_to_generate = self.add_treatment_column(self.data_to_generate)
        self.data_pref_to_generate = self.create_and_encode_prefixes(self.data_treat_to_generate)
        self.data_prep_to_generate, _, _, _, _, _, _, _ = self.scale(self.data_pref_to_generate, self.scaler_dict)

        self.data_treat_to_generate_val = self.add_treatment_column(self.data_to_generate_val)
        self.data_pref_to_generate_val = self.create_and_encode_prefixes(self.data_treat_to_generate_val)
        self.data_prep_to_generate_val, _, _, _, _, _, _, _ = self.scale(self.data_pref_to_generate_val, self.scaler_dict)
        return self.data_prep, self.prep_utils, self.data_prep_to_generate, self.data_prep_to_generate_val
    

    def preprocess_bozorgi_retaining(self, only_preprocess_realcause_data = False):
        self.data_treat = self.add_treatment_column(self.data_train)
        print("data_treat", self.data_treat[:20])
        self.data_pref_treat = self.create_and_encode_prefixes_treat(self.data_treat)
        self.data_pref_control = self.create_and_encode_prefixes_control(self.data_treat, self.prefix_length_treat_dict)
        # combine and shuffle
        self.data_pref = pd.concat([self.data_pref_treat, self.data_pref_control], axis=0, ignore_index=True)
        self.data_pref = self.data_pref.sample(frac=1, random_state=42).reset_index(drop=True)
        # delete any columns that have zero everywhere
        self.data_pref = self.data_pref.loc[:, (self.data_pref != 0).any(axis=0)]
        self.data_prep, self.scaler_dict, min_outcome_scaled, max_outcome_scaled, atoms_scaled, bin_width_scaled, stdev_atoms_scaled, split_list_scaled = self.scale(self.data_pref)
        self.prep_utils = {"max_process_len": self.max_process_len,
                            "scaler_dict": self.scaler_dict,
                            "min_outcome_scaled": min_outcome_scaled,
                            "max_outcome_scaled": max_outcome_scaled,
                            "atoms_scaled": atoms_scaled,
                            "bin_width_scaled": bin_width_scaled,
                            "stdev_atoms_scaled": stdev_atoms_scaled,
                            "split_list_scaled": split_list_scaled,
                            "column_names": self.data_pref.columns}
        
        if only_preprocess_realcause_data:
            return self.data_prep, self.prep_utils, None, None
        
        self.data_treat_to_generate = self.add_treatment_column(self.data_to_generate)
        self.data_pref_treat_to_generate = self.create_and_encode_prefixes_treat(self.data_treat_to_generate)
        self.data_pref_control_to_generate = self.create_and_encode_prefixes_control(self.data_treat_to_generate, self.prefix_length_treat_dict)
        # combine and shuffle
        self.data_pref_to_generate = pd.concat([self.data_pref_treat_to_generate, self.data_pref_control_to_generate], axis=0, ignore_index=True)
        self.data_pref_to_generate = self.data_pref_to_generate.sample(frac=1, random_state=42).reset_index(drop=True)
        # delete any columns that have zero everywhere
        self.data_pref_to_generate = self.data_pref_to_generate.loc[:, (self.data_pref_to_generate != 0).any(axis=0)]
        self.data_prep_to_generate, _, _, _, _, _, _, _ = self.scale(self.data_pref_to_generate, self.scaler_dict)

        self.data_treat_to_generate_val = self.add_treatment_column(self.data_to_generate_val)
        self.data_pref_treat_to_generate_val = self.create_and_encode_prefixes_treat(self.data_treat_to_generate_val)
        self.data_pref_control_to_generate_val = self.create_and_encode_prefixes_control(self.data_treat_to_generate_val, self.prefix_length_treat_dict)
        # combine and shuffle
        self.data_pref_to_generate_val = pd.concat([self.data_pref_treat_to_generate_val, self.data_pref_control_to_generate_val], axis=0, ignore_index=True)
        self.data_pref_to_generate_val = self.data_pref_to_generate_val.sample(frac=1, random_state=42).reset_index(drop=True)
        # delete any columns that have zero everywhere
        self.data_pref_to_generate_val = self.data_pref_to_generate_val.loc[:, (self.data_pref_to_generate_val != 0).any(axis=0)]
        self.data_prep_to_generate_val, _, _, _, _, _, _, _ = self.scale(self.data_pref_to_generate_val, self.scaler_dict)
        return self.data_prep, self.prep_utils, self.data_prep_to_generate, self.data_prep_to_generate_val

    
    def preprocess_sample_bozorgi_retaining(self, data_sample, prep_utils, print_cases = False):
        data_treat_sample = self.add_treatment_column(data_sample)
        data_pref_sample = self.create_and_encode_prefix_sample(data_treat_sample, prep_utils)
        if print_cases:
            print("data_pref_sample", data_pref_sample)
        data_prep_sample, _, _, _, _, _, _, _ = self.scale(data_pref_sample, prep_utils["scaler_dict"])
        # drop outcome column
        data_prep_sample = data_prep_sample.drop(columns=["outcome"])
        # the desired order is prep_utils["column_names"] but first without treatment, then with treatment and also without outcome
        desired_order = [col for col in prep_utils["column_names"] if col != "treatment" and col != "outcome"] + ["treatment"]
        data_prep_sample = data_prep_sample.reindex(columns=desired_order)
        return data_prep_sample

    
    def add_treatment_column(self, data, print_debug=False, treatment_index=None, scaler_dict_train=None):
        if self.dataset_params["intervention_info"]["column"] == "activity":
            intervention_activity = self.dataset_params["intervention_info"]["actions"][-1]
            data['treatment'] = np.where(data['activity'].shift(-1) == intervention_activity, 1, 0)
        else:
            if self.dataset_params["intervention_info"]["column"] == "interest_rate":
                scaled_intervention_actions = pd.DataFrame(self.dataset_params["intervention_info"]["actions"], columns=["interest_rate"])
                scaled_intervention_actions = self.scale_column(col = "interest_rate", data = scaled_intervention_actions, scaler=scaler_dict_train["interest_rate"])[1]
                zeros_list = [0] * len(scaled_intervention_actions)
                data['treatment'] = [zeros_list for _ in range(len(data))]

                if treatment_index is not None:
                    new_zero_list = zeros_list.copy()
                    new_zero_list[treatment_index] = 1
                    activity_column = "activity_" + "calculate_offer"
                    case_nr_value_last_calc_offer = -1
                    for row_nr, row in data[data['interest_rate'] == scaled_intervention_actions["interest_rate"][treatment_index]].iterrows():
                        if row[activity_column] == 1.0:
                            if row["case_nr"] != case_nr_value_last_calc_offer:
                                case_nr_value_last_calc_offer = row["case_nr"]
                                data.at[row_nr, 'treatment'] = new_zero_list
                else:
                    activity_column = "activity_" + "calculate_offer"
                    case_nr_value_last_calc_offer = -1
                    for row_nr, row in data.iterrows():
                        if row[activity_column] == 1.0:
                            if row["case_nr"] != case_nr_value_last_calc_offer:
                                for i, option in enumerate(scaled_intervention_actions["interest_rate"]):
                                    if row["interest_rate"] == option:
                                        new_zero_list = zeros_list.copy()
                                        new_zero_list[i] = 1
                                        case_nr_value_last_calc_offer = row["case_nr"]
                                        data.at[row_nr - 1, 'treatment'] = new_zero_list

        if print_debug:
            print('data_treatment below')
        return data
    

    def create_and_encode_prefix_sample(self, data, prep_utils):
        train_prep = pd.DataFrame()
        sum_dict = {col: 0 for col in self.scale_cols}
        scale_col_count = {col: 0 for col in self.scale_cols}
        # if there is activity and count in the column name, add it to the activity_count_dict
        activity_count_dict = {col: 0 for col in prep_utils["column_names"] if "count" in col}
        last_state_dict = {col: 0 for col in self.last_state_cols}

        for current_pos, (index, row) in enumerate(data.iterrows(), start=1):
            for col in self.scale_cols:
                if not pd.isna(row[col]):
                    sum_dict[col] += row[col]
                    scale_col_count[col] += 1
            activity_count_dict[row["activity"] + "_count"] += 1

            for col in self.last_state_cols:
                last_state_dict[col] = row[col]
            
            mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols}
            
            # if we are at the last row of the data, we need to add the prefix to the dataframe
            if current_pos == len(data) - 1:
                prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict}, index=[0])
                prefix["treatment"] = row["treatment"]
                prefix["prefix_len"] = current_pos
                train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
                break
        
        return train_prep
    

    def create_and_encode_prefixes_treat(self, data, only_treated=True, only_prefix_including_treatment=True, count_prefix_length=True):
        grouped_train = data.groupby("case_nr")
        # check the amount of groups with no treatment == 1 in it
        self.scale_cols_aggregate = [col for col in self.scale_cols if col not in self.last_state_cols]
        unique_activities = data["activity"].unique()

        train_prep = pd.DataFrame()
        self.prefix_length_treat_dict = {}
        self.nr_treat_cases = 0

        for case_nr, group in grouped_train:
            if not self.big_data:
                if case_nr == 500:
                    break
            # initiate a mean for every scale aggregate column
            sum_dict = {col: 0 for col in self.scale_cols_aggregate}
            scale_col_count = {col: 0 for col in self.scale_cols_aggregate}
            # initiate a count for every activity
            activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
            # initiate a last state for every last state column
            last_state_dict = {col: 0 for col in self.last_state_cols}

            if only_treated:
                # check if there is a row in the group with treatment == 1
                if 1 not in group["treatment"].values:
                    continue
            
            self.nr_treat_cases += 1

            treated_case = False
            if only_prefix_including_treatment:
                if 1 in group["treatment"].values:
                    treated_case = True

            # if there is a row in the group with treatment == 1, only retain the the prefix ending at that row (not the previous ones)
            # if there is never a treatment == 1, retain all prefixes (for loop)
            for current_pos, (index, row) in enumerate(group.iterrows(), start=1):
                for col in self.scale_cols_aggregate:
                    if not pd.isna(row[col]):
                        sum_dict[col] += row[col]
                        scale_col_count[col] += 1
                activity_count_dict[row["activity"] + "_count"] += 1

                for col in self.last_state_cols:
                    last_state_dict[col] = row[col]
                
                # mean_dict = {col: sum_dict[col] / current_pos for col in self.scale_cols_aggregate}
                mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols_aggregate}

                # put all dictionaries in a dataframe
                prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict}, index=[0])
                prefix["treatment"] = row["treatment"]

                prefix["prefix_len"] = current_pos                

                # add prefix to the dataframe
                if only_prefix_including_treatment:
                    if treated_case:
                        if row["treatment"] == 1:
                            train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
                            if count_prefix_length:
                                if current_pos not in self.prefix_length_treat_dict:
                                    self.prefix_length_treat_dict[current_pos] = 0
                                self.prefix_length_treat_dict[current_pos] += 1
                    else:
                        train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
                        if count_prefix_length:
                            if current_pos not in self.prefix_length_treat_dict:
                                self.prefix_length_treat_dict[current_pos] = 0
                            self.prefix_length_treat_dict[current_pos] += 1
                else:
                    train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
                    if count_prefix_length:
                        if current_pos not in self.prefix_length_treat_dict:
                            self.prefix_length_treat_dict[current_pos] = 0
                        self.prefix_length_treat_dict[current_pos] += 1

                if row["treatment"] == 1:
                    break
        
        print('prefix_length_treat_dict', self.prefix_length_treat_dict)
        return train_prep
    

    def create_and_encode_prefixes_control(self, data, prefix_length_treat_dict):
        grouped_train = data.groupby("case_nr")
        self.scale_cols_aggregate = [col for col in self.scale_cols if col not in self.last_state_cols]
        unique_activities = data["activity"].unique()

        train_prep = pd.DataFrame()
        self.nr_control_cases = 0

        for case_nr, group in grouped_train:
            if not self.big_data:
                if case_nr == 500:
                    break
            if case_nr % 1000 == 0:
                print('Case nr in control:', case_nr)
            # initiate a mean for every scale aggregate column
            sum_dict = {col: 0 for col in self.scale_cols_aggregate}
            scale_col_count = {col: 0 for col in self.scale_cols_aggregate}
            # initiate a count for every activity
            activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
            # initiate a last state for every last state column
            last_state_dict = {col: 0 for col in self.last_state_cols}
            
            # check if there is a row in the group with treatment == 1
            if 1 in group["treatment"].values:
                continue
                
            self.nr_control_cases += 1
            # if there is a row in the group with treatment == 1, only retain the the prefix ending at that row (not the previous ones)
            # if there is never a treatment == 1, retain all prefixes (for loop)
            for current_pos, (index, row) in enumerate(group.iterrows(), start=1):
                for col in self.scale_cols_aggregate:
                    if not pd.isna(row[col]):
                        sum_dict[col] += row[col]
                        scale_col_count[col] += 1
                activity_count_dict[row["activity"] + "_count"] += 1

                for col in self.last_state_cols:
                    last_state_dict[col] = row[col]
                
                mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols_aggregate}

                # put all dictionaries in a dataframe
                prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict}, index=[0])
                prefix["treatment"] = row["treatment"]
                prefix["prefix_len"] = current_pos

                # add prefix to the dataframe
                train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
        
        self.proportion_prefix_length_treat_dict = {key: prefix_length_treat_dict[key] / self.nr_treat_cases for key in prefix_length_treat_dict}

        # take the same proportion of control cases with the corresponding prefix length as the treated cases
        # so just randomly sample, so that x% of control cases have prefix length y, and so on
        train_prep_control = pd.DataFrame()
        for key in prefix_length_treat_dict:
            nr_control_cases_to_sample = int(self.proportion_prefix_length_treat_dict[key] * self.nr_control_cases)
            if nr_control_cases_to_sample > 0:
                control_cases_sample = train_prep[train_prep["prefix_len"] == key].sample(n=nr_control_cases_to_sample, random_state=1)
                train_prep_control = pd.concat([train_prep_control, control_cases_sample], axis=0, ignore_index=True)
        
        # get the amount of control cases with a certain prefix length
        self.prefix_length_control_dict = {}
        for key in prefix_length_treat_dict:
            self.prefix_length_control_dict[key] = train_prep_control[train_prep_control["prefix_len"] == key].shape[0]
        self.proportion_prefix_length_control_dict = {key: self.prefix_length_control_dict[key] / self.nr_control_cases for key in self.prefix_length_control_dict}

        print('prefix_length_control_dict', self.prefix_length_control_dict)
        return train_prep_control
    

    def create_and_encode_prefixes(self, data):
        grouped_train = data.groupby("case_nr")
        self.scale_cols_aggregate = [col for col in self.scale_cols if col not in self.last_state_cols]
        unique_activities = data["activity"].unique()

        train_prep = pd.DataFrame()

        for case_nr, group in grouped_train:
            if not self.big_data:
                if case_nr == 500:
                    break
            # initiate a mean for every scale aggregate column
            sum_dict = {col: 0 for col in self.scale_cols_aggregate}
            scale_col_count = {col: 0 for col in self.scale_cols_aggregate}
            # initiate a count for every activity
            activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
            # initiate a last state for every last state column
            last_state_dict = {col: 0 for col in self.last_state_cols}

            # if there is a row in the group with treatment == 1, only retain the the prefix ending at that row (not the previous ones)
            # if there is never a treatment == 1, retain all prefixes (for loop)
            for current_pos, (index, row) in enumerate(group.iterrows(), start=1):
                for col in self.scale_cols_aggregate:
                    if not pd.isna(row[col]):
                        sum_dict[col] += row[col]
                        scale_col_count[col] += 1
                activity_count_dict[row["activity"] + "_count"] += 1

                for col in self.last_state_cols:
                    last_state_dict[col] = row[col]
                
                mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols_aggregate}

                # put all dictionaries in a dataframe
                prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict}, index=[0])
                prefix["treatment"] = row["treatment"]
                prefix["prefix_len"] = current_pos

                # add prefix to the dataframe
                train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)

                if row["treatment"] == 1:
                    break
        
        return train_prep
    

    def scale(self, data, scaler=None, only_atoms=False):
        # first append the activity counts to the scale cols and prefix_len
        self.scale_cols_currently = self.scale_cols + [col for col in data.columns if "_count" in col] + ["prefix_len"]

        if scaler is None:
            scaler_dict = {}
            for col in self.scale_cols_currently:
                scaler = StandardScaler()
                data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
                if col == "outcome":
                    min_outcome_scaled = data[col].min()
                    max_outcome_scaled = data[col].max()
                    if self.atoms is not None:
                        self.atoms_scaled = pd.DataFrame(self.atoms)
                        self.atoms_scaled = scaler.transform(self.atoms_scaled.values.reshape(-1, 1)).flatten()
                        stdev_scaler = scaler.scale_[0]
                        self.stdev_atoms_scaled = [self.stdev_atoms[i] / stdev_scaler for i in range(len(self.stdev_atoms))]
                        self.bin_width_scaled = abs(scaler.transform(np.array(self.bin_with).reshape(-1, 1)).flatten()[0])
                    else:
                        self.atoms_scaled = None
                        self.stdev_atoms_scaled = None
                        self.bin_width_scaled = None
                    if self.split_list is not None:
                        self.split_list_scaled = []
                        for i, split in enumerate(self.split_list):
                            self.split_list_scaled.append(scaler.transform(np.array(split).reshape(-1, 1)).flatten()[0])
                    else:
                        self.split_list_scaled = None
                scaler_dict[col] = deepcopy(scaler)
            return data, scaler_dict, min_outcome_scaled, max_outcome_scaled, self.atoms_scaled, self.bin_width_scaled, self.stdev_atoms_scaled, self.split_list_scaled
        elif only_atoms:
            scaler = scaler["outcome"]
            if self.atoms is not None:
                    self.atoms_scaled = pd.DataFrame(self.atoms)
                    self.atoms_scaled = scaler.transform(self.atoms_scaled.values.reshape(-1, 1)).flatten()
                    stdev_scaler = scaler.scale_[0]
                    self.stdev_atoms_scaled = [self.stdev_atoms[i] / stdev_scaler for i in range(len(self.stdev_atoms))]
                    self.bin_width_scaled = abs(scaler.transform(np.array(self.bin_with).reshape(-1, 1)).flatten()[0])
            else:
                self.atoms_scaled = None
                self.stdev_atoms_scaled = None
                self.bin_width_scaled = None
            return None, None, None, None, self.atoms_scaled, self.bin_width_scaled, self.stdev_atoms_scaled, None
        else:
            for col in self.scale_cols_currently:
                data[col] = scaler[col].transform(data[col].values.reshape(-1, 1))
            return data, None, None, None, None, None, None, None