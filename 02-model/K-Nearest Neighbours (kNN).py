import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def split_data(train_data, test_data):
    """
    Split train and test data into features and target.
    """
    X_train, y_train = train_data.drop(columns=['hospital_death']), train_data['hospital_death']
    X_test, y_test = test_data.drop(columns=['hospital_death']), test_data['hospital_death']
    return X_train, y_train, X_test, y_test

def calculate_auc(y_true, y_pred_proba):
    """
    Calculate AUC and return FPR, TPR, and AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return auc(fpr, tpr), fpr, tpr

def custom_cross_val_score(model, X, y, cv=10):
    """
    Perform manual cross-validation.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

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

def search_best_parameter_with_results(X, y, cv=10):
    """
    Search for the best parameters using grid search and cross-validation.
    """
    n_neighbors_options = [3, 5, 10]
    weights_options = ['uniform', 'distance']
    metric_options = ['minkowski', 'manhattan']

    results = []

    print("Starting parameter search...")

    for n_neighbors in n_neighbors_options:
        for weights in weights_options:
            for metric in metric_options:
                try:
                    # Define the kNN model
                    model = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        metric=metric
                    )

                    # Perform cross-validation
                    auc_scores = custom_cross_val_score(model, X, y, cv=cv)
                    mean_auc = np.mean(auc_scores)

                    print(f"Parameters: n_neighbors={n_neighbors}, weights={weights}, metric={metric}")
                    print(f"Cross-Validation Mean AUC: {mean_auc:.6f}")

                    # Append results
                    results.append({
                        'n_neighbors': n_neighbors,
                        'weights': weights,
                        'metric': metric,
                        'mean_auc': mean_auc
                    })

                except Exception as e:
                    print(f"Error with n_neighbors={n_neighbors}, weights={weights}, metric={metric}: {e}")
                    results.append({
                        'n_neighbors': n_neighbors,
                        'weights': weights,
                        'metric': metric,
                        'mean_auc': np.nan  # Record NaN for invalid configurations
                    })

    results_df = pd.DataFrame(results)
    return results_df

def plot_results(results_df):
    """
    Plot parameter search results for visualization.
    """
    for weights in results_df['weights'].unique():
        for metric in results_df['metric'].unique():
            subset = results_df[(results_df['weights'] == weights) & (results_df['metric'] == metric)]
            plt.plot(
                subset['n_neighbors'], subset['mean_auc'],
                label=f"Weights={weights}, Metric={metric}"
            )

    plt.xlabel('Number of Neighbors (n_neighbors)')
    plt.ylabel('Mean AUC')
    plt.title('Parameter Search Results')
    plt.legend()
    plt.grid()
    plt.show()

def testing_data(best_model, X_test, y_test):
    """
    Test the best model on the test set and calculate AUC.
    """
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc, fpr, tpr = calculate_auc(y_test, y_test_pred_proba)
    return test_auc, fpr, tpr

def print_roc_curve(test_auc, fpr, tpr, dataset_name):
    """
    Plot the ROC curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    print("Script execution started...")

    datasets = [
        {
            "train_path": '/Users/zhuanzhuan/Desktop/005-ml-group-project/00-dataset/01-data01/train_data_1.csv',
            "test_path": '/Users/zhuanzhuan/Desktop/005-ml-group-project/00-dataset/01-data01/test_data_1.csv',
            "name": "Dataset 1"
        },
        {
            "train_path": '/Users/zhuanzhuan/Desktop/005-ml-group-project/00-dataset/03-data03/train_data_3.csv',
            "test_path": '/Users/zhuanzhuan/Desktop/005-ml-group-project/00-dataset/03-data03/test_data_3.csv',
            "name": "Dataset 2"
        }
    ]

    for dataset in datasets:
        print(f"\nProcessing {dataset['name']}...")
        try:
            train_data = pd.read_csv(dataset["train_path"])
            test_data = pd.read_csv(dataset["test_path"])

            print("Data loaded successfully!")
            print(f"Train data shape: {train_data.shape}")
            print(f"Test data shape: {test_data.shape}")

            X_train, y_train, X_test, y_test = split_data(train_data, test_data)

            print("Training and testing...")

            results_df = search_best_parameter_with_results(X_train, y_train)
            best_params = results_df.loc[results_df['mean_auc'].idxmax()]
            print(f"Best Parameters: {best_params}")

            best_model = KNeighborsClassifier(
                n_neighbors=int(best_params['n_neighbors']),
                weights=best_params['weights'],
                metric=best_params['metric']
            )
            best_model.fit(X_train, y_train)

            test_auc, fpr, tpr = testing_data(best_model, X_test, y_test)
            print(f"Test AUC for {dataset['name']}: {test_auc:.4f}")
            print_roc_curve(test_auc, fpr, tpr, dataset["name"])

            print(f"{dataset['name']} processing completed.")

        except Exception as e:
            print(f"An error occurred while processing {dataset['name']}: {e}")

    print("Script execution completed.")