import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from copy import deepcopy

class LoanProcessPreprocessor():
    def __init__(self, dataset_params, data_train = pd.DataFrame(), data_train_val = pd.DataFrame(), time_wise = True):
        #GENERAL params
        self.dataset_params = dataset_params
        self.time_wise = time_wise

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


    #PREPROCESSING
    def preprocess(self):
        #TRAIN
        self.oh_encoder_dict_train, self.data_encoded_train = self.one_hot_encode_columns(data = self.data_train)
        self.scaler_dict_train, self.data_scaled_train = self.scale_columns(data = self.data_encoded_train)
        self.data_fill_train = self.handle_missing_values(data = self.data_scaled_train)
        self.data_treat_train = self.add_treatment_column(data = self.data_fill_train, scaler_dict_train=self.scaler_dict_train)
        
        if not self.time_wise:
            self.data_treat_train = self.data_treat_train.loc[:, ~self.data_treat_train.columns.str.contains('activity')]
            self.event_cols_encoded = [col for col in self.event_cols_encoded if 'activity' not in col]
        
        self.create_pad_row()
        self.data_tensor_train = self.create_prefix_tensors(data = self.data_treat_train, max_process_len = self.max_process_len)
        
        if self.time_wise:
            self.data_padded_train = self.padding(data = self.data_tensor_train, pad_type="pre", max_process_len = self.max_process_len, pad_event=self.pad_event)
        else:
            self.data_padded_train = self.data_tensor_train
        
        self.data_retained_train = self.retain_normal(data_padded=self.data_padded_train, data_type="train")

        #VAL
        self.oh_encoder_dict_train_val, self.data_encoded_train_val = self.one_hot_encode_columns(data = self.data_train_val, oh_encoder_dict = self.oh_encoder_dict_train)
        self.scaler_dict_train_val, self.data_scaled_train_val = self.scale_columns(data = self.data_encoded_train_val, scaler_dict = self.scaler_dict_train)
        self.data_fill_train_val = self.handle_missing_values(data = self.data_scaled_train_val)
        self.data_treat_train_val = self.add_treatment_column(data = self.data_fill_train_val, scaler_dict_train=self.scaler_dict_train)
        
        if not self.time_wise:
            self.data_treat_train_val = self.data_treat_train_val.loc[:, ~self.data_treat_train_val.columns.str.contains('activity')]
            self.event_cols_encoded = [col for col in self.event_cols_encoded if 'activity' not in col]
        
        self.data_tensor_train_val = self.create_prefix_tensors(data = self.data_treat_train_val, max_process_len = self.max_process_len)
        
        if self.time_wise:
            self.data_padded_train_val = self.padding(data = self.data_tensor_train_val, pad_type="pre", max_process_len = self.max_process_len, pad_event=self.pad_event)
        else:
            self.data_padded_train_val = self.data_tensor_train_val
        
        self.data_retained_train_val = self.retain_normal(data_padded=self.data_padded_train_val, data_type="train")
        
        #Force data structure
        self.data_train_prep = self.force_data_structure(data = self.data_retained_train)
        self.data_train_val_prep = self.force_data_structure(data = self.data_retained_train_val)

        self.prep_utils = {"scaler_dict_train": self.scaler_dict_train, 
                                "oh_encoder_dict_train": self.oh_encoder_dict_train, 
                                "max_process_len": self.max_process_len,
                                "case_cols_encoded": self.case_cols_encoded,
                                "event_cols_encoded": self.event_cols_encoded,
                                "pad_case": self.pad_case,
                                "pad_event": self.pad_event}
        
        return self.data_train_prep, self.data_train_val_prep, self.prep_utils


    def preprocess_sample_CI(self, data_sample, prep_utils, device):
        _, data_encoded_sample = self.one_hot_encode_columns(data = data_sample, oh_encoder_dict = prep_utils["oh_encoder_dict_train"])
        _, data_scaled_sample = self.scale_columns(data = data_encoded_sample, scaler_dict = prep_utils["scaler_dict_train"])
        data_fill_sample = self.handle_missing_values(data = data_scaled_sample)
        data_treat_sample = self.add_treatment_column(data = data_fill_sample, scaler_dict_train=prep_utils["scaler_dict_train"])

        if not self.time_wise:
            data_treat_sample = data_treat_sample.loc[:, ~data_treat_sample.columns.str.contains('activity')]
            self.event_cols_encoded = [col for col in self.event_cols_encoded if 'activity' not in col]

        data_tensor_sample = self.create_prefix_tensors(data = data_treat_sample, max_process_len = prep_utils["max_process_len"])

        if self.time_wise:
            data_padded_sample = self.padding(data = data_tensor_sample, pad_type="pre", max_process_len = prep_utils["max_process_len"], pad_event=prep_utils["pad_event"])
        else:
            data_padded_sample = data_tensor_sample
        
        data_retained_sample = self.retain_normal(data_padded=data_padded_sample, data_type="inference")
        data_sample_prep = self.force_data_structure(data = data_retained_sample)
        return data_sample_prep
    

    #Preprocess sub functions
    def one_hot_encode_columns(self, data, oh_encoder_dict=None):
        self.case_cols_encoded = [col for col in self.case_cols if col not in self.cat_cols]
        self.event_cols_encoded = [col for col in self.event_cols if col not in self.cat_cols]

        if oh_encoder_dict is None:
            oh_encoder_dict = {}
            for col in self.cat_cols:
                oh_encoder_dict[col], data, cat_col_encoded = self.one_hot_encode_column(col = col, data = data)
                if col in self.case_cols:
                    self.case_cols_encoded.extend(cat_col_encoded)
                elif col in self.event_cols:
                    self.event_cols_encoded.extend(cat_col_encoded)
        else:
            #(oh_encoder_dict is known)
            for col, oh_encoder in oh_encoder_dict.items():
                _, data, cat_col_encoded = self.one_hot_encode_column(col = col, data = data, oh_encoder = oh_encoder)
                if col in self.case_cols:
                    self.case_cols_encoded.extend(cat_col_encoded)
                elif col in self.event_cols:
                    self.event_cols_encoded.extend(cat_col_encoded)

        return oh_encoder_dict, data
    
    def one_hot_encode_column(self, col, data, oh_encoder=None):
        if oh_encoder is None:
            oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_col = oh_encoder.fit_transform(data[[col]])
            cat_col_encoded = oh_encoder.get_feature_names_out(input_features=[col])
        else:
            #(oh_encoder is known)
            encoded_col = oh_encoder.transform(data[[col]])
            cat_col_encoded = oh_encoder.get_feature_names_out(input_features=[col])
        df_enc = pd.DataFrame(encoded_col, columns=cat_col_encoded)
        data = data.reset_index(drop=True).join(df_enc)
        data.drop(columns=[col], inplace=True)

        return oh_encoder, data, cat_col_encoded

    def scale_columns(self, data, scaler_dict = None):
        if scaler_dict is None:
            scaler_dict = {}
            for col in self.scale_cols:
                if col in data.columns:
                    scaler_dict[col], data = self.scale_column(col, data)
        else:
            #(scale_dict is known)
            for col, scaler in scaler_dict.items():
                if col in data.columns:
                    self.scale_column(col, data, scaler)

        return scaler_dict, data
    
    def scale_column(self, col, data, scaler=None):
        #don't standardize missing values
        non_null_col_rows = ~data[col].isnull()

        if not data.loc[non_null_col_rows, col].empty:
            if scaler is None:
                scaler = StandardScaler()
                data.loc[non_null_col_rows, col] = scaler.fit_transform(data.loc[non_null_col_rows, col].values.reshape(-1, 1)).flatten()
            else:
                #(scaler is known)
                data.loc[non_null_col_rows, col] = scaler.transform(data.loc[non_null_col_rows, col].values.reshape(-1, 1)).flatten()

        return scaler, data
    
    def handle_missing_values(self, data):
        #Only floats are missing normally
        data.fillna(-100, inplace=True)
        return data

    def add_treatment_column(self, data, print_debug=False, treatment_index=None, scaler_dict_train=None):
        if self.dataset_params["intervention_info"]["column"] == "activity":
            intervention_activity = "activity_" + self.dataset_params["intervention_info"]["actions"][-1]
            data["treatment"] = data[intervention_activity].shift(-1).fillna(0).astype(int)
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
                    # print("lollings ervoooooor")
                    for row_nr, row in data[data['interest_rate'] == scaled_intervention_actions["interest_rate"][treatment_index]].iterrows():
                        # print("lollings")
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

    def create_pad_row(self):
        if self.case_cols_encoded != [] or self.event_cols_encoded != []:
            df_pad_row = pd.DataFrame(index=[0], columns=self.case_cols_encoded + self.event_cols_encoded)
        else:
            print("case cat cols and process cat cols are empty")
        
        df_pad_row[self.case_cols_encoded + self.event_cols_encoded] = df_pad_row[self.case_cols_encoded + self.event_cols_encoded].astype(float)
        df_pad_row.loc[:, self.case_cols_encoded] = 0.0
        df_pad_row.loc[:, self.event_cols_encoded] = 0.0
        if not self.scale_cols == []:
            _, df_pad_row = self.scale_columns(data = df_pad_row, scaler_dict = self.scaler_dict_train)
        self.pad_case = torch.Tensor(df_pad_row[self.case_cols_encoded].values[0])#.to(self.device)
        self.pad_event = torch.Tensor(df_pad_row[self.event_cols_encoded].values[0])#.to(self.device)
        pass

    def create_tensor_not_time_wise(self, data, prefix_len = 1):
        Y = torch.Tensor(data["outcome"].values)
        case_nr = torch.Tensor(data["case_nr"].values)
        treatment = torch.Tensor(data["treatment"].values)
        X_case = torch.Tensor(data[self.case_cols_encoded].values)
        X_event = torch.Tensor(data[self.event_cols_encoded].values)
        # also create a prefix length tensor
        prefix_len = torch.Tensor([prefix_len] * len(data))
        return [Y, case_nr, treatment, prefix_len, X_case, X_event, self.case_cols_encoded, self.event_cols_encoded]

    def create_prefix_tensors(self, data, max_process_len):
        previous_case = -1
        X_cols = ["case_nr", "prefix_len"] + self.case_cols_encoded + self.event_cols_encoded
        X = torch.zeros(size=(len(data), len(X_cols) + self.nr_treatment_columns, max_process_len))
        for row_nr, row in data.iterrows():
            current_case = row["case_nr"]
            if current_case != previous_case:
                event_nr = 0
                previous_case = current_case
            else:
                # copy all previous prefixes
                event_nr += 1
                X[row_nr, :, 0:event_nr] = X[row_nr-1, :, 0:event_nr]
            # add an event
            X[row_nr, 0, event_nr] = current_case

            # Process variable-length treatment list
            treatment_list = row["treatment"]
            X[row_nr, 1:1 + self.nr_treatment_columns, event_nr] = torch.tensor(treatment_list, dtype=torch.float32)
            last_index = 1+self.nr_treatment_columns

            X[row_nr, last_index, 0:event_nr + 1] = event_nr + 1
            last_index += 1

            X[row_nr, last_index:last_index + len(self.case_cols_encoded), event_nr] = torch.tensor(row[self.case_cols_encoded].values.astype(float))
            last_index += len(self.case_cols_encoded)
            X[row_nr, last_index: last_index + len(self.event_cols_encoded), event_nr] = \
                torch.tensor(row[self.event_cols_encoded].values.astype(float))

        # create tensors for all kinds of variables
        Y = torch.Tensor(data["outcome"].values)
        case_nr = X[:, 0 ,0]
        treatment = X[:, 1:1 + self.nr_treatment_columns, :]
        last_index = 1+self.nr_treatment_columns
        prefix_len = X[:, 1 + self.nr_treatment_columns, 0]
        last_index += 1
        X_case = X[:, last_index:last_index + len(self.case_cols_encoded), 0] #, :]
        last_index += len(self.case_cols_encoded)
        X_process = X[:, last_index: last_index + len(self.event_cols_encoded), :]
        data_tensor = [Y, case_nr, treatment, prefix_len, X_case, X_process, self.case_cols_encoded, self.event_cols_encoded]

        return data_tensor
    
    def padding(self, data, max_process_len, pad_event, pad_type='pre'):
        [Y, case_nr, treatment, prefix_len, X_case, X_event, case_cols_encoded, proc_cols_encoded] = data

        #NORMAL
        new_X_event = torch.unsqueeze(torch.unsqueeze(pad_event, 0), 2).repeat(X_event.shape[0], 1, max_process_len)
        new_treatment = torch.zeros(size = treatment.shape)
        #adding zero's to beginning (pre) or end (post) of sequence to make it seq_len long
        for pref_len in torch.unique(prefix_len):
            seq_len = max_process_len
            idx = (prefix_len == pref_len).nonzero().squeeze()
            if pad_type == "post":
                # post padding
                start = int(max(pref_len - seq_len, 0))
                stop = int(min(seq_len, pref_len))
                new_X_event[idx, :, :stop] = X_event[idx, :, start:int(pref_len)]
                new_treatment[idx, :, :stop] = treatment[idx, :, start:int(pref_len)]
            else:
                # pre padding
                start = int(max(seq_len - pref_len, 0))
                stop = int(min(seq_len, pref_len))
                new_X_event[idx, :, start:seq_len] = X_event[idx, :, 0:stop]
                new_treatment[idx, :, start:seq_len] = treatment[idx, :, 0:stop]
       
        return [Y, case_nr, new_treatment, prefix_len, X_case, new_X_event, case_cols_encoded, proc_cols_encoded]
    
    def retain_normal(self, data_padded, data_type="train"):
        [Y, case_nr, treatment, prefix_len, X_case, X_event, case_cols_encoded, proc_cols_encoded] = data_padded

        #ONLY RELEVANT
        #Check which ones have all zeros
        all_zeros_indices = torch.all(treatment[:, :, :-1] == 0, dim=-1).all(dim=-1).nonzero().squeeze(dim=-1)

        if self.dataset_params["intervention_info"]["retain_method"] == "precise":
            if len(self.dataset_params["intervention_info"]["start_control_activity"]) > 0:
                #Check which ones have a 1 in the start_control_activities
                start_control_indices_list = []
                for start_control in self.dataset_params["intervention_info"]["start_control_activity"]:
                    if 'activity_' + start_control in self.event_cols_encoded:
                        start_control_index = self.event_cols_encoded.index("activity_" + start_control)
                        start_control_indices = torch.any(X_event[:, start_control_index, :] == 1, dim=-1).nonzero().squeeze(dim=-1)
                        start_control_indices_list.append(start_control_indices)
                # the final one is one that occurs in one of the start_control_indices_list
                final_start_control_indices = torch.cat(start_control_indices_list).unique()
                end_control_indices_list = []
                for end_control in self.dataset_params["intervention_info"]["end_control_activity"]:
                    if 'activity_' + end_control in self.event_cols_encoded:
                        end_control_index = self.event_cols_encoded.index("activity_" + end_control)
                        # get the ones that have a 1 in the last element in the end_control_activity
                        end_control_indices = (X_event[:, end_control_index, -1] == 1).nonzero().squeeze(dim=-1)
                        end_control_indices_list.append(end_control_indices)
                final_end_control_indices = torch.cat(end_control_indices_list).unique()
                
                # control_indices is than those that are in all_zeros_indices, in final_start_control_indices and in final_end_control_indices
                control_indices = all_zeros_indices
                cat, counts = torch.cat([control_indices, final_start_control_indices]).unique(return_counts=True)
                control_indices = cat[torch.where(counts.gt(1))]
                cat, counts = torch.cat([control_indices, final_end_control_indices]).unique(return_counts=True)
                control_indices = cat[torch.where(counts.gt(1))]
            else:
                # make it an empty tensor, same dimensions as all_zeros_indices (fill with -100)
                control_indices = torch.full((all_zeros_indices.size(0),), -100, dtype=torch.int64)
        else:
            control_indices = all_zeros_indices

        #Check which ones have a 1 in the last element
        last_element_indices = (treatment[:, :, -1] == 1).any(dim=-1).nonzero().squeeze(dim=-1)
        if data_type == "train":
            boolean_retainer = torch.isin(torch.arange(treatment.size(0)), torch.cat((control_indices, last_element_indices)))
        elif data_type == "inference":
            last_index = prefix_len.size(0) - 1 - 1 #NOTE, additional -1 to retain without intervention
            boolean_retainer = torch.isin(torch.arange(treatment.size(0)), (last_index))
        else:
            boolean_retainer = torch.isin(torch.arange(treatment.size(0)), (last_element_indices))
        retain_idx = boolean_retainer

        case_nr_retained = case_nr[retain_idx]
        prefix_len_retained = prefix_len[retain_idx]
        Y_retained = Y[retain_idx]
        new_treatment_retained = treatment[retain_idx]
        X_case_retained = X_case[retain_idx]
        new_X_event_retained = X_event[retain_idx]

        return [Y_retained, case_nr_retained, new_treatment_retained, prefix_len_retained, X_case_retained, new_X_event_retained, case_cols_encoded, proc_cols_encoded]
    

    def force_data_structure(self, data):
        [Y, case_nr, treatment, prefix_len, X_case, X_event, x_case_cols_encoded, x_event_cols_encoded] = data
        T = treatment.bool()
        return [T, prefix_len, X_case, X_event, Y, x_case_cols_encoded, x_event_cols_encoded, case_nr]