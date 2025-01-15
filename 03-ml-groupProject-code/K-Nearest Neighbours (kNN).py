import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time

print("---------------------------------------------------read csv---------------------------------------------------")

# Define file paths
datasets = {
    "data1": {
        "train_path": "01-dataset-preprocessed/data_1_train.csv",
        "test_path": "01-dataset-preprocessed/data_1_test.csv",
    },
    "data2": {
        "train_path": "01-dataset-preprocessed/data_2_train.csv",
        "test_path": "01-dataset-preprocessed/data_2_test.csv",
    },
    "data3": {
        "train_path": "01-dataset-preprocessed/data_3_train.csv",
        "test_path": "01-dataset-preprocessed/data_3_test.csv",
    },
}

# Read CSV files into DataFrames
dataframes = {}
for key, paths in datasets.items():
    train_data = pd.read_csv(paths["train_path"])
    test_data = pd.read_csv(paths["test_path"])
    dataframes[key] = (train_data, test_data)

print("---------------------------------------------------model---------------------------------------------------")

# Function to split data into features and target
def split_data(train_data, test_data):
    X_train, y_train = train_data.drop(columns=["hospital_death"]), train_data["hospital_death"]
    X_test, y_test = test_data.drop(columns=["hospital_death"]), test_data["hospital_death"]
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

# Grid search for the best parameters
def search_best_parameter(X, y, cv=5):
    n_neighbors_options = [1, 3, 5, 10]
    weights_options = ['uniform', 'distance']
    metric_options = ['minkowski', 'manhattan']

    best_auc = 0
    best_params = None
    best_model = None
    param_results = []

    for n_neighbors in n_neighbors_options:
        for weights in weights_options:
            for metric in metric_options:
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric
                )
                auc_scores = custom_cross_val_score(model, X, y, cv=cv)
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

    best_model.fit(X, y)
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation AUC: {best_auc:.4f}")

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

# Plot and save the ROC curve
def print_roc_curve(test_auc, fpr, tpr, file_name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(file_name, dpi=300)

# Plot combined ROC curves for multiple models
def plot_combined_roc_curve(results, title_name, file_name):
    plt.figure(figsize=(10, 8))
    for result in results:
        plt.plot(
            result["fpr"],
            result["tpr"],
            label=f"Model {result['model']} (AUC={result['auc']:.4f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title_name)
    plt.legend()
    plt.grid()
    plt.savefig(file_name, dpi=300)

# Main process to train and evaluate models for each dataset
model_results = {}

for data_name, (train_data, test_data) in dataframes.items():
    print(f"----- kNN - {data_name} -----")
    start_time = time.time()
    X_train, y_train, X_test, y_test = split_data(train_data, test_data)

    best_model, param_results = search_best_parameter(X_train, y_train, cv=3)
    print_grid_search_results(param_results)
    test_auc, fpr, tpr = testing_data(best_model, X_test, y_test)

    print_roc_curve(test_auc, fpr, tpr, f"02-image-output/knn-{data_name}.jpg")
    model_results[data_name] = {
        "model": best_model,
        "fpr": fpr,
        "tpr": tpr,
        "auc": test_auc,
    }

    execution_time = time.time() - start_time
    print(f"{data_name} execution time: {execution_time:.4f} seconds")

# Evaluate each model on all datasets and plot combined ROC curves
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
        results, f"kNN {data_name} ROC Curve", f"02-image-output/knn-combined-roc-{data_name}.jpg"
    )
