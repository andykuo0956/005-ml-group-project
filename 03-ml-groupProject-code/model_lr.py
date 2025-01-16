import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import time

# Start measuring time
start_time = time.time()

# Load Datasets
train_data_1 = pd.read_csv('/Users/aragoto/Desktop/FML/dataset/train_data_1.csv')  # 90:10
test_data_1  = pd.read_csv('/Users/aragoto/Desktop/FML/dataset/test_data_1.csv')   # 90:10

train_data_2 = pd.read_csv('/Users/aragoto/Desktop/FML/dataset/train_data_2.csv')  # 70:30
test_data_2  = pd.read_csv('/Users/aragoto/Desktop/FML/dataset/test_data_2.csv')   # 70:30

train_data_3 = pd.read_csv('/Users/aragoto/Desktop/FML/dataset/train_data_3.csv')  # 50:50
test_data_3  = pd.read_csv('/Users/aragoto/Desktop/FML/dataset/test_data_3.csv')   # 50:50

target_column = 'hospital_death'


def prepare_data(df: pd.DataFrame, target_col: str):
    """
    Splits a DataFrame into (X, y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# Prepare (X,y) for each dataset
X_train_1, y_train_1 = prepare_data(train_data_1, target_column)  # 90:10
X_test_1,  y_test_1  = prepare_data(test_data_1,  target_column)

X_train_2, y_train_2 = prepare_data(train_data_2, target_column)  # 70:30
X_test_2,  y_test_2  = prepare_data(test_data_2,  target_column)

X_train_3, y_train_3 = prepare_data(train_data_3, target_column)  # 50:50
X_test_3,  y_test_3  = prepare_data(test_data_3,  target_column)


# Scaling & Manual CV
def scale_dataframe(X_train: pd.DataFrame, X_val_or_test: pd.DataFrame):
    """
    Compute the training-set mean/std, then apply them to the other set.
    """
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds = stds.replace(to_replace=0, value=1.0)

    X_train_scaled = (X_train - means) / stds
    X_val_or_test_scaled = (X_val_or_test - means) / stds
    return X_train_scaled, X_val_or_test_scaled


def manual_cross_validation(X: pd.DataFrame,
                            y: pd.Series,
                            model,
                            n_splits=5) -> float:
    """
    Perform manual k-fold cross-validation (default = 5).
    Returns mean AUC.
    """
    fold_size = len(X) // n_splits
    auc_scores = []

    for i in range(n_splits):
        start = i * fold_size
        end   = (i + 1) * fold_size

        # Split out validation fold
        X_val_fold = X.iloc[start:end, :]
        y_val_fold = y.iloc[start:end]

        # Training fold is everything else
        X_train_fold = pd.concat([X.iloc[:start, :], X.iloc[end:, :]], axis=0)
        y_train_fold = pd.concat([y.iloc[:start], y.iloc[end:]], axis=0)

        model.fit(X_train_fold, y_train_fold)
        y_val_proba = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_val_proba)
        auc_scores.append(auc)

    return np.mean(auc_scores)


def find_best_hyperparams_via_CV(X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 param_grid: dict) -> dict:
    """
    Manual grid search + cross-validation to find best hyperparams for a single dataset.
    Returns the best params as a dict.
    """
    best_auc = 0.0
    best_params = None

    print("Starting hyperparameter tuning...")
    for C in param_grid['C']:
        for solver in param_grid['solver']:
            for max_iter in param_grid['max_iter']:
                print(f"Evaluating parameters: C={C}, solver={solver}, max_iter={max_iter}")

                # Build model with these hyperparams
                model = LogisticRegression(
                    C=C, solver=solver, max_iter=max_iter, random_state=42, tol=1e-3
                )

                # 5-fold CV on the *scaled* dataset
                cv_auc = manual_cross_validation(X_train, y_train, model)

                if cv_auc > best_auc:
                    best_auc = cv_auc
                    best_params = {
                        'C': C,
                        'solver': solver,
                        'max_iter': max_iter
                    }
                    print(f"New best AUC: {best_auc:.4f}")

    return best_params


# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [500, 1000, 3000]
}


# For each dataset_i (i=1..3), find best hyperparams, train final model
def get_best_model_for_dataset(X_train_raw, y_train, param_grid):
    """
    1. Scale X_train_raw (since we do cross-validation on that training set).
    2. Find best hyperparams (using the scaled X_train).
    3. Train final LogisticRegression on the entire scaled training set with best hyperparams.
    4. Return (best_model, best_params).
    """
    # Scale X_train_raw
    X_train_scaled, _ = scale_dataframe(X_train_raw, X_train_raw)

    # Find best hyperparams on scaled data
    best_params = find_best_hyperparams_via_CV(X_train_scaled, y_train, param_grid)

    # Train a final model using the entire training set (scaled) with the best params
    best_model = LogisticRegression(
        C=best_params['C'],
        solver=best_params['solver'],
        max_iter=best_params['max_iter'],
        random_state=42
    )
    best_model.fit(X_train_scaled, y_train)

    return best_model, best_params


# Get the "best" model on each distribution
best_model_1, best_params_1 = get_best_model_for_dataset(X_train_1, y_train_1, param_grid)  # 90:10
best_model_2, best_params_2 = get_best_model_for_dataset(X_train_2, y_train_2, param_grid)  # 70:30
best_model_3, best_params_3 = get_best_model_for_dataset(X_train_3, y_train_3, param_grid)  # 50:50


# Evaluate All Trained Models on Each of the Three Test Distributions
def scale_test_with_train_stats(X_train_raw, X_test_raw):
    """
    Scale X_test_raw using the mean/std from X_train_raw (the training distribution).
    """
    means = X_train_raw.mean(axis=0)
    stds  = X_train_raw.std(axis=0)
    stds  = stds.replace(to_replace=0, value=1.0)

    X_test_scaled = (X_test_raw - means) / stds
    return X_test_scaled


def plot_roc_for_test_dataset(test_name,
                              X_test_raw, y_test,
                              trained_models_dict,
                              train_data_dict,
                              save_path):
    """
    test_name: e.g. "dataset1 (90:10)"
    X_test_raw, y_test: the raw test set
    trained_models_dict: e.g. {"dataset1": best_model_1, "dataset2": best_model_2, "dataset3": best_model_3}
    train_data_dict:     e.g. {"dataset1": (X_train_1, y_train_1), ...} for scaling
    """
    plt.figure(figsize=(8, 6))

    colors = {
        "dataset1": "blue",
        "dataset2": "orange",
        "dataset3": "green"
    }

    for train_label, model in trained_models_dict.items():
        X_train_raw_for_scaling, _ = train_data_dict[train_label]

        # Scale this test set using the stats from the training distribution
        X_test_scaled = scale_test_with_train_stats(X_train_raw_for_scaling, X_test_raw)

        # Predict probabilities
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc_val = roc_auc_score(y_test, y_proba)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr,
                 tpr,
                 label=f"Trained on {train_label} (AUC={auc_val:.4f})",
                 color=colors[train_label])

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

    plt.title(f"Logistic Regression ROC Curve - Test on {test_name}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(save_path, format="jpg", dpi=300)
    plt.show()


# Prepare dictionaries for plotting
trained_models_dict = {
    "dataset1": best_model_1,  # trained on 90:10
    "dataset2": best_model_2,  # trained on 70:30
    "dataset3": best_model_3,  # trained on 50:50
}

train_data_dict = {
    "dataset1": (X_train_1, y_train_1),
    "dataset2": (X_train_2, y_train_2),
    "dataset3": (X_train_3, y_train_3),
}


# Plot for test dataset1 (90:10)
plot_roc_for_test_dataset(
    test_name="dataset1 (90:10)",
    X_test_raw=X_test_1,
    y_test=y_test_1,
    trained_models_dict=trained_models_dict,
    train_data_dict=train_data_dict,
    save_path="dataset1_roc_curve.jpg"
)

# Plot for test dataset2 (70:30)
plot_roc_for_test_dataset(
    test_name="dataset2 (70:30)",
    X_test_raw=X_test_2,
    y_test=y_test_2,
    trained_models_dict=trained_models_dict,
    train_data_dict=train_data_dict,
    save_path="dataset2_roc_curve.jpg"
)

# Plot for test dataset3 (50:50)
plot_roc_for_test_dataset(
    test_name="dataset3 (50:50)",
    X_test_raw=X_test_3,
    y_test=y_test_3,
    trained_models_dict=trained_models_dict,
    train_data_dict=train_data_dict,
    save_path="dataset3_roc_curve.jpg"
)

# End measuring time
end_time = time.time()

# Calculate and print the total running time
elapsed_time = end_time - start_time
print(f"Total Running Time: {elapsed_time:.4f} seconds")