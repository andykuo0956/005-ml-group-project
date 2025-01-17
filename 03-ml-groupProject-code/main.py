# input library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import argparse


def remove_duplicates_and_drop_empty(df):
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")
    return df


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


# Create multiple stratified sub-sampled sets
def create_stratified_subsamples(df, target_col, frac_divisor=6, random_seed=42):
    from math import floor

    sampled_datasets = {
        "sampled_data_1": pd.DataFrame(),
        "sampled_data_2": pd.DataFrame(),
        "sampled_data_3": pd.DataFrame(),
    }

    # Count how many data points we want in total in each subsampled set
    total_to_sample = floor(len(df) / frac_divisor)
    class_dist = df[target_col].value_counts(normalize=True)
    grouped = df.groupby(target_col)

    for name in sampled_datasets.keys():
        sub_df = pd.DataFrame()
        for class_label, group in grouped:
            num_samples_for_class = int(class_dist[class_label] * total_to_sample)
            sub_df = pd.concat(
                [
                    sub_df,
                    group.sample(n=num_samples_for_class, random_state=random_seed),
                ]
            )
        # Shuffle to reduce data order bias
        sampled_datasets[name] = sub_df.sample(
            frac=1, random_state=random_seed
        ).reset_index(drop=True)

    return sampled_datasets


# Applies SMOTE-NC to each sub-sampled dataset. The number of desired_ratios should match the number of sub-samples in sampled_data_dict.
def apply_smote_nc(sampled_data_dict, target_col, desired_ratios, random_seed=42):
    derived_datasets = {}
    data_keys = list(
        sampled_data_dict.keys()
    )  # e.g. ["sampled_data_1", "sampled_data_2", ...]

    for i, ratio in enumerate(desired_ratios, start=1):
        current_key = data_keys[i - 1]
        subset_df = sampled_data_dict[current_key]

        # Separate out categorical columns
        cat_cols = subset_df.select_dtypes(include=["object"]).columns.tolist()
        encoder = ce.BinaryEncoder(cols=cat_cols)
        encoded_df = encoder.fit_transform(subset_df)

        X = encoded_df.drop(columns=[target_col])
        y = encoded_df[target_col]

        # Figure out which columns are categorical (encoded)
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
        smote_nc = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy={1: target_minority},
            random_state=random_seed,
        )
        X_resampled, y_resampled = smote_nc.fit_resample(X, y)

        # Merge back to a single DataFrame
        derived_data = pd.concat(
            [
                pd.DataFrame(X_resampled, columns=X.columns),
                pd.DataFrame(y_resampled, columns=[target_col]),
            ],
            axis=1,
        )

        # Shuffle to reduce data order bias
        derived_data = derived_data.sample(
            frac=1, random_state=random_seed
        ).reset_index(drop=True)
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
        master_data, target_col=target_column, frac_divisor=6, random_seed=42
    )

    # 5) Apply SMOTE-NC
    desired_minority_class_ratios = [0.10, 0.30, 0.50]
    derived_datasets = apply_smote_nc(
        sampled_datasets,
        target_col=target_column,
        desired_ratios=desired_minority_class_ratios,
        random_seed=42,
    )

    # 6) Train/test splits for each derived dataset
    train_sets, test_sets = {}, {}
    for i in range(1, 4):
        data_key = f"derived_data_{i}"
        train_data, test_data = stratified_split(
            derived_datasets[data_key], target_column, train_ratio=0.7, random_seed=42
        )
        train_sets[f"train_data_{i}"] = train_data
        test_sets[f"test_data_{i}"] = test_data

    # Rename them for clarity
    train_data_1, test_data_1 = train_sets["train_data_1"], test_sets["test_data_1"]
    train_data_2, test_data_2 = train_sets["train_data_2"], test_sets["test_data_2"]
    train_data_3, test_data_3 = train_sets["train_data_3"], test_sets["test_data_3"]

    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")

print(
    "---------------------------------------------------model---------------------------------------------------"
)


# Function to split data into features and target
def split_data(train_data, test_data):
    X_train, y_train = (
        train_data.drop(columns=["hospital_death"]),
        train_data["hospital_death"],
    )
    X_test, y_test = (
        test_data.drop(columns=["hospital_death"]),
        test_data["hospital_death"],
    )
    return X_train, y_train, X_test, y_test


# Function to calculate AUC and ROC curve
def calculate_auc(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return auc(fpr, tpr), fpr, tpr


# Custom cross-validation function to calculate AUC scores
def custom_cross_val_score(model, X, y, cv=5, seed=42):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    np.random.seed(seed)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    fold_size = len(y) // cv
    scores = []

    for i in range(cv):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)

        model.fit(X_train, y_train)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        auc_score = roc_auc_score(y_val, y_val_pred_proba)
        scores.append(auc_score)

    return scores


# Grid search for the best parameters for Random Forest, KNN and Logistic Regression
def search_best_parameter(X, y, model_type="random_forest", cv=5):
    param_results = []

    if model_type == "random_forest":
        n_estimators_range = [100, 150, 200]
        max_depth_range = [10, 15, 20]

        best_auc = 0
        best_params = None
        best_model = None

        for n_estimators in n_estimators_range:
            for max_depth in max_depth_range:
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth
                )
                auc_scores = custom_cross_val_score(model, X, y, cv=cv, seed=42)
                mean_auc = np.mean(auc_scores)

                param_results.append(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "mean_auc": mean_auc,
                    }
                )

                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                    best_model = model
        np.random.seed(42)
        best_model.fit(X, y)
        return best_model, param_results

    elif model_type == "knn":
        n_neighbors_range = [1, 3, 5, 10]
        weights_options = ["uniform", "distance"]
        metric_options = ["minkowski", "manhattan"]

        best_auc = 0
        best_params = None
        best_model = None

        for n_neighbors in n_neighbors_range:
            for weights in weights_options:
                for metric in metric_options:
                    try:
                        model = KNeighborsClassifier(
                            n_neighbors=n_neighbors, weights=weights, metric=metric
                        )
                        auc_scores = custom_cross_val_score(model, X, y, cv=cv, seed=42)
                        mean_auc = np.mean(auc_scores)

                        param_results.append(
                            {
                                "n_neighbors": n_neighbors,
                                "weights": weights,
                                "metric": metric,
                                "mean_auc": mean_auc,
                            }
                        )

                        if mean_auc > best_auc:
                            best_auc = mean_auc
                            best_params = {
                                "n_neighbors": n_neighbors,
                                "weights": weights,
                                "metric": metric,
                            }
                            best_model = model

                    except Exception as e:
                        param_results.append(
                            {
                                "n_neighbors": n_neighbors,
                                "weights": weights,
                                "metric": metric,
                                "mean_auc": np.nan,
                            }
                        )

        np.random.seed(42)

        best_model.fit(X, y)
        return best_model, param_results

    elif model_type == "logistic_regression":
        C_range = [0.1, 1.0, 10]
        solver_options = ["liblinear", "saga"]
        max_iter_options = [1000, 1500, 2000]

        best_auc = 0
        best_model = None
        best_params = None

        for C_val in C_range:
            for solver in solver_options:
                for max_iter_val in max_iter_options:
                    model = LogisticRegression(
                        C=C_val,
                        solver=solver,
                        max_iter=max_iter_val,
                        random_state=42,
                        tol=1e-3
                    )
                    auc_scores = custom_cross_val_score(model, X, y, cv=cv, seed=42)
                    mean_auc = np.mean(auc_scores)

                    param_results.append(
                        {
                            "C": C_val,
                            "solver": solver,
                            "max_iter": max_iter_val,
                            "mean_auc": mean_auc,
                        }
                    )

                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_params = {
                            "C": C_val,
                            "solver": solver,
                            "max_iter": max_iter_val,
                        }
                        best_model = model
        np.random.seed(42)
        # Fit on entire training data
        best_model.fit(X, y)
        return best_model, param_results


# Print grid search results as a table
def print_grid_search_results(param_results):
    df = pd.DataFrame(param_results)
    print("\nGrid Search Results:")
    print(df.sort_values(by="mean_auc", ascending=False).to_string(index=False))
    return df


# Test the model on the test dataset
def testing_data(best_model, X_test, y_test):
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc, fpr, tpr = calculate_auc(y_test, y_test_pred_proba)
    print(f"Test AUC: {test_auc:.4f}")
    return test_auc, fpr, tpr


# Plot combined ROC curves for multiple models
import matplotlib.pyplot as plt


def plot_combined_roc_curve(results, title_name, file_name):
    plt.figure(figsize=(10, 8))

    for result in results:
        plt.plot(
            result["fpr"],
            result["tpr"],
            label=f"Model trained on {result['model']} (AUC={result['auc']:.4f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.title(title_name, fontsize=24)
    plt.legend(loc="lower right", fontsize=16)
    plt.grid()
    plt.tight_layout(pad=0)
    plt.savefig(file_name, dpi=300, bbox_inches="tight")


# Main process to train and evaluate models for each dataset
def run_model(datasets, model_type="logistic_regression"):
    print(f"Running {model_type.title()}...")
    model_results = {}
    combined_results = []

    for data_name, (train_data, test_data) in datasets.items():
        print(f"----- {model_type.title()} - {data_name} -----")
        X_train, y_train, X_test, y_test = split_data(train_data, test_data)

        best_model, param_results = search_best_parameter(
            X_train, y_train, model_type=model_type, cv=5
        )
        print_grid_search_results(param_results)

        test_auc, fpr, tpr = testing_data(best_model, X_test, y_test)
        model_results[data_name] = {
            "model": best_model,
            "auc": test_auc,
            "fpr": fpr,
            "tpr": tpr,
        }

    for data_name, (_, test_data) in datasets.items():
        print(f"\n----- Evaluating All Models on {data_name} -----")
        X_test, y_test = (
            test_data.drop(columns=["hospital_death"]),
            test_data["hospital_death"],
        )
        results = []

        for train_data_name, result in model_results.items():
            model = result["model"]
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]
            test_auc, fpr, tpr = calculate_auc(y_test, y_test_pred_proba)
            results.append(
                {"model": train_data_name, "auc": test_auc, "fpr": fpr, "tpr": tpr}
            )
            print(
                f"Model trained on {train_data_name}, tested on {data_name}: AUC = {test_auc:.4f}"
            )

        plot_combined_roc_curve(
            results,
            f"ROC Curves-{model_type.title()} tested on {data_name}",
            f"02-image-output/{model_type}-combined-roc-{data_name}.jpg",
        )

    return model_results


# Parse command-line arguments
def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description="Run experiments for machine learning models."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logistic_regression", "knn", "random_forest", "all"],
        help="The type of model to run: 'logistic_regression', 'knn', 'random_forest', or 'all'.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    datasets = {
        "data1": (train_data_1, test_data_1),
        "data2": (train_data_2, test_data_2),
        "data3": (train_data_3, test_data_3),
    }

    args = parse_command_line_args()

    if args.model == "logistic_regression":
        run_model(datasets, model_type="logistic_regression")
    elif args.model == "knn":
        run_model(datasets, model_type="knn")
    elif args.model == "random_forest":
        run_model(datasets, model_type="random_forest")
    elif args.model == "all":
        print(
            "Running all models in order: Logistic Regression -> KNN -> Random Forest."
        )
        run_model(datasets, model_type="logistic_regression")
        run_model(datasets, model_type="knn")
        run_model(datasets, model_type="random_forest")
    else:
        print(
            "Invalid model type specified. Use 'logistic_regression', 'knn', 'random_forest', or 'all'."
        )

    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")
