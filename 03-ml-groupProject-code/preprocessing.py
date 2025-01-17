# input library
import pandas as pd
import numpy as np
import category_encoders as ce
from imblearn.over_sampling import SMOTENC
import time
import os

def remove_duplicates_and_drop_empty(df):
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")
    return df

# See reference 10 in report for reasoning behind imoputing values this way.
def fill_missing_values(df):
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
    
# Drop non-relevant features from the DataFrame.
def drop_non_relevant_features(df, cols_to_drop):
    return df.drop(columns=cols_to_drop, errors="ignore")

# Create multiple stratified sub-sampled sets - based on code from reference 11 in the report.
def create_stratified_subsamples(df, target_col, frac_divisor=6, random_seed=42):
    from math import floor
    sampled_datasets = {
        "sampled_data_1": pd.DataFrame(),
        "sampled_data_2": pd.DataFrame(),
        "sampled_data_3": pd.DataFrame()
    }

    # Count how many data points we want in total in each subsampled set
    total_to_sample = floor(len(df) / frac_divisor)
    class_dist = df[target_col].value_counts(normalize=True)
    grouped = df.groupby(target_col)

    for name in sampled_datasets.keys():
        sub_df = pd.DataFrame()
        for class_label, group in grouped:
            num_samples_for_class = int(class_dist[class_label] * total_to_sample)
            sub_df = pd.concat([
                sub_df, 
                group.sample(n=num_samples_for_class, random_state=random_seed)
            ])
        # Shuffle to reduce data order bias
        sampled_datasets[name] = sub_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return sampled_datasets

# Applies SMOTE-NC to each sub-sampled dataset. The number of desired_ratios should match the number of sub-samples in sampled_data_dict.
def apply_smote_nc(sampled_data_dict, target_col, desired_ratios, random_seed=42):
    derived_datasets = {}
    data_keys = list(sampled_data_dict.keys())  # e.g. ["sampled_data_1", "sampled_data_2", ...]

    for i, ratio in enumerate(desired_ratios, start=1):
        current_key = data_keys[i-1]
        subset_df = sampled_data_dict[current_key]

        # Separate out categorical columns
        cat_cols = subset_df.select_dtypes(include=["object"]).columns.tolist()
        encoder = ce.BinaryEncoder(cols=cat_cols)
        encoded_df = encoder.fit_transform(subset_df)

        X = encoded_df.drop(columns=[target_col])
        y = encoded_df[target_col]

        # Figure out which columns are categorical (encoded)
        # code derived from references 12 and 13 in the report.
        cat_indices = [
            X.columns.get_loc(c)
            for c in X.columns
            if any(c.startswith(orig_col) for orig_col in cat_cols)
        ]

        # Calculate how many minority samples we need to create class imbalances
        minority_count = y.value_counts().get(1, 0)
        majority_count = y.value_counts().get(0, 0)
        target_minority = int(majority_count * (ratio / (1 - ratio)))

        # SMOTENC
        # Code derived from references 14 and 15 in the report.
        smote_nc = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy={1: target_minority},
            random_state=random_seed
        )
        X_resampled, y_resampled = smote_nc.fit_resample(X, y)

        # Merge back to a single DataFrame
        derived_data = pd.concat([
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.DataFrame(y_resampled, columns=[target_col])
        ], axis=1)

        # Shuffle to reduce data order bias
        derived_data = derived_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        derived_datasets[f"derived_data_{i}"] = derived_data

    return derived_datasets

# Splits a dataset into train/test sets in a stratified manner.
def stratified_split(df, target_col, train_ratio=0.7, random_seed=42):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for label, group in df.groupby(target_col):
        group = group.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        cutoff = int(train_ratio * len(group))
        train_df = pd.concat([train_df, group.iloc[:cutoff]])
        test_df = pd.concat([test_df, group.iloc[cutoff:]])

    # Shuffle final train/test to reduce data order bias
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    start_time = time.time()
    
    # 1) Load the dataset
    master_data = pd.read_csv("00-raw-dataset/dataset.csv")
    
    # 2) Basic cleaning
    master_data = remove_duplicates_and_drop_empty(master_data)
    master_data = fill_missing_values(master_data)

    # 3) Remove non-relevant features
    drop_cols = ["encounter_id", "patient_id", "hospital_id", "icu_id"]
    master_data = drop_non_relevant_features(master_data, drop_cols)
    
    target_column = "hospital_death"
    
    # 4) Create sub-samples
    sampled_datasets = create_stratified_subsamples(
        master_data, 
        target_col=target_column, 
        frac_divisor=6, 
        random_seed=42
    )

    # 5) Apply SMOTE-NC
    desired_minority_class_ratios = [0.10, 0.30, 0.50] 
    derived_datasets = apply_smote_nc(
        sampled_datasets, 
        target_col=target_column, 
        desired_ratios=desired_minority_class_ratios, 
        random_seed=42
    )
    
    # 6) Train/test splits for each derived dataset
    train_sets, test_sets = {}, {}
    for i in range(1, 4):
        data_key = f"derived_data_{i}"
        train_data, test_data = stratified_split(derived_datasets[data_key], target_column, train_ratio=0.7, random_seed=42)
        train_sets[f"train_data_{i}"] = train_data
        test_sets[f"test_data_{i}"] = test_data

    # Rename them for clarity
    train_data_1, test_data_1 = train_sets["train_data_1"], test_sets["test_data_1"]
    train_data_2, test_data_2 = train_sets["train_data_2"], test_sets["test_data_2"]
    train_data_3, test_data_3 = train_sets["train_data_3"], test_sets["test_data_3"]

    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")

print("----------save as csv------------")
# Save DataFrames directly to the folder "01-dataset-preprocessed"
train_data_1.to_csv("01-dataset-preprocessed/data_1_train.csv", index=False)
test_data_1.to_csv("01-dataset-preprocessed/data_1_test.csv", index=False)

train_data_2.to_csv("01-dataset-preprocessed/data_2_train.csv", index=False)
test_data_2.to_csv("01-dataset-preprocessed/data_2_test.csv", index=False)

train_data_3.to_csv("01-dataset-preprocessed/data_3_train.csv", index=False)
test_data_3.to_csv("01-dataset-preprocessed/data_3_test.csv", index=False)
