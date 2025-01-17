# input library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time

# Start measuring time
start_time = time.time()

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
    param_results = []

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


# Plot combined ROC curves
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
def run_model(datasets, model_type="knn"):
    print(f"Running {model_type.title()}...")
    model_results = {}
    combined_results = []

    for data_name, (train_data, test_data) in datasets.items():
        print(f"----- {model_type.title()} - {data_name} -----")
        X_train, y_train, X_test, y_test = split_data(train_data, test_data)

        best_model, param_results = search_best_parameter(X_train, y_train, cv=5)
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


if __name__ == "__main__":
    datasets = {
        "data1": (train_data_1, test_data_1),
        "data2": (train_data_2, test_data_2),
        "data3": (train_data_3, test_data_3),
    }

    run_model(datasets, model_type="knn")

# End measuring time
end_time = time.time()

# Calculate and print the total running time
elapsed_time = end_time - start_time
print(f"Total Running Time: {elapsed_time:.4f} seconds")
