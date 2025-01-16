# input library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time

print(
    "---------------------------------------------------read csv---------------------------------------------------"
)
# Define file paths
train_file_1 = "01-dataset-preprocessed/data_1_train.csv"
test_file_1 = "01-dataset-preprocessed/data_1_test.csv"

train_file_2 = "01-dataset-preprocessed/data_2_train.csv"
test_file_2 = "01-dataset-preprocessed/data_2_test.csv"

train_file_3 = "01-dataset-preprocessed/data_3_train.csv"
test_file_3 = "01-dataset-preprocessed/data_3_test.csv"

# Read CSV files into DataFrames
train_data_1 = pd.read_csv(train_file_1)
test_data_1 = pd.read_csv(test_file_1)

train_data_2 = pd.read_csv(train_file_2)
test_data_2 = pd.read_csv(test_file_2)

train_data_3 = pd.read_csv(train_file_3)
test_data_3 = pd.read_csv(test_file_3)

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

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Shuffle the data
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split data into folds
    fold_size = len(y) // cv
    scores = []

    for i in range(cv):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)

        # Train model and predict probabilities
        model.fit(X_train, y_train)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate AUC
        auc_score = roc_auc_score(y_val, y_val_pred_proba)
        scores.append(auc_score)

    return scores


# Grid search for the best parameters
def search_best_parameter(X, y, cv=5):
    n_estimators_range = [100, 150, 200]
    max_depth_range = [10, 15, 20]

    best_auc = 0
    best_params = None
    best_model = None
    param_results = []

    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
            auc_scores = custom_cross_val_score(model, X, y, cv=cv)
            mean_auc = np.mean(auc_scores)

            # Record the parameters and AUC
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
datasets = {
    "data1": (train_data_1, test_data_1),
    "data2": (train_data_2, test_data_2),
    "data3": (train_data_3, test_data_3),  # Add data3 for training and testing
}

model_results = {}

# Train and evaluate models for each dataset
for data_name, (train_data, test_data) in datasets.items():
    print(f"----- Random Forest - {data_name} -----")
    start_time = time.time()
    X_train, y_train, X_test, y_test = split_data(train_data, test_data)

    best_model, param_results = search_best_parameter(X_train, y_train, cv=3)
    print_grid_search_results(param_results)
    test_auc, fpr, tpr = testing_data(best_model, X_test, y_test)

    print_roc_curve(test_auc, fpr, tpr, f"02-image-output/random-forest-{data_name}.jpg")
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
        results, f"Random-Forest {data_name} ROC Curve", f"02-image-output/random-forest-combined-roc-{data_name}.jpg"
    )
