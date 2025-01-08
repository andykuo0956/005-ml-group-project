# input library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time

# 1 data processing
total_execution_time = 0
start_time = time.time()
# 1.1 Loading the Dataset
# 1.1.1 Reloading the dataset
master_data = pd.read_csv("00-raw-dataset/dataset.csv")


# 1.1.2 Removing duplicate rows and dropping completely empty columns
master_data = master_data.drop_duplicates()
master_data = master_data.dropna(axis=1, how="all")

# 1.1.3 Checking the updated structure
print(master_data.head(6))

# 1.2 Missing Value Analysis
# 1.2.1 Calculate Missing Values and their Percentages
missing_values = master_data.isnull().sum()
missing_percentage = (missing_values / len(master_data)) * 100

# Create a missing data report
missing_data_report = pd.DataFrame(
    {
        "Column": master_data.columns,
        "Missing Values": missing_values,
        "Missing Percentage (%)": missing_percentage,
    }
).sort_values(by="Missing Percentage (%)", ascending=False)

# Display columns with missing values
print(missing_data_report[missing_data_report["Missing Values"] > 0])

# 1.2.2 Handling Missing Values
# Strategy 1: Impute missing values for numeric columns with mean
numeric_columns = master_data.select_dtypes(include=["float64", "int64"]).columns
master_data[numeric_columns] = master_data[numeric_columns].apply(
    lambda col: col.fillna(col.mean())
)

# Strategy 2: Impute missing values for categorical columns with mode
categorical_columns = master_data.select_dtypes(include=["object"]).columns
master_data[categorical_columns] = master_data[categorical_columns].apply(
    lambda col: col.fillna(col.mode()[0])
)

# Check for remaining missing values
remaining_missing_values = master_data.isnull().sum().sum()

# Display the updated dataset structure and check if missing values remain
print(master_data.info(), remaining_missing_values)

# 1.3 Identifying and removing non-relevant features
# Remove non-related columns based on domain knowledge
non_related_columns = ["encounter_id", "patient_id", "hospital_id", "icu_id"]
master_data = master_data.drop(columns=non_related_columns, errors="ignore")

# 2.1 Create two sub-sampled datasets
# 2.1.1 Use stratified sampling to keep intial class imbalnce of master dataset
# Identify column with target values for classifcation
target_column = "hospital_death"

# Determine distribution and absolute count for each class in the master dataset
master_class_distribution = master_data[target_column].value_counts(normalize=True)
master_class_count = master_data[target_column].value_counts()
print("Master dataset class distribution (Proportion):")
print(master_class_distribution)
print("\n Master dataset class absolute count:")
print(master_class_count)

# Equation to set number of samples in each derived dataset
# derived in to 6 parts because the scale of the original dataset is too large
num_samples = len(master_data) // 6

# Group data by target class
grouped = master_data.groupby(target_column)

# Initialise a dictionary to store sampled datasets
sampled_datasets = {
    "sampled_data_1": pd.DataFrame(),
    "sampled_data_2": pd.DataFrame(),
    "sampled_data_3": pd.DataFrame(),
}

# Create 2 sub-sampled datasets
for i, dataset_name in enumerate(sampled_datasets.keys(), start=1):
    sampled_data = pd.DataFrame()

    # Sample data from each class group
    # random_state is set to simulate semi-random behaviour for reproducability of experiments
    for class_label, group in grouped:
        samples_to_take = int(master_class_distribution[class_label] * num_samples)
        sampled_class_data = group.sample(
            n=samples_to_take, random_state=np.random.randint(1000)
        )
        sampled_data = pd.concat([sampled_data, sampled_class_data])

    # Shuffle data and reset index to reduce data order bias after concatination
    sampled_datasets[dataset_name] = sampled_data.sample(
        frac=1, random_state=np.random.randint(1000)
    ).reset_index(drop=True)

# Assign variables explicitly for easier access later on
sampled_data_1 = sampled_datasets["sampled_data_1"]
sampled_data_2 = sampled_datasets["sampled_data_2"]
sampled_data_3 = sampled_datasets["sampled_data_3"]

# Print summaries of each dataset to verify proportion of classes and absolute count of data values
for name, data in sampled_datasets.items():
    print(f"\n{name} class distribution:")
    print(data[target_column].value_counts(normalize=True))
    print(f"\n{name} class absolute count:")
    print(data[target_column].value_counts())

# 2.2 Create class-imbalanced derived dataset using SMOTE-NC (Nominal Continuous)
# Desired rations for the minority class
desired_minority_class_ratios = [0.10, 0.50, 0.30]

# Create empty dictionary for derived datasets
derived_datasets = {}

for i, (sampled_data, desired_minority_class_ratio) in enumerate(
    zip(
        [sampled_data_1, sampled_data_2, sampled_data_3], desired_minority_class_ratios
    ),
    start=1,
):
    # Step 1: identify categorical columns for sampled datasets
    categorical_columns = sampled_data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Step 2: apply binary coding for use in SMOTE-NC
    binary_encoder = ce.BinaryEncoder(cols=categorical_columns)
    encoded_data = binary_encoder.fit_transform(sampled_data)

    # Step 3: separate features and target
    X = encoded_data.drop(columns=[target_column])
    y = encoded_data[target_column]

    # Step 4: define categorical indices after encoding
    binary_categorical_indices = [
        X.columns.get_loc(col)
        for col in X.columns
        if any(col.startswith(cat) for cat in categorical_columns)
    ]

    # Step 5: define correct sampling strategy for SMOTE-NC to match the desired class ratio
    minority_class_count = y.value_counts().get(1, 0)
    majority_class_count = y.value_counts().get(0, 0)
    correct_sampling_strategy = int(
        majority_class_count
        * (desired_minority_class_ratio / (1 - desired_minority_class_ratio))
    )

    # Step 6: apply SMOTE-NC with defined sampling strategy
    smote_nc = SMOTENC(
        categorical_features=binary_categorical_indices,
        sampling_strategy={1: correct_sampling_strategy},
        random_state=42,
    )
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    # Step 7: Combine resampled data to create derived datasets
    derived_data = pd.concat(
        [
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.DataFrame(y_resampled, columns=[target_column]),
        ],
        axis=1,
    )

    # Shuffle data and reset index to reduce data order bias after concatination
    derived_data = derived_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 8: store the derived datasets
    derived_datasets[f"derived_data_{i}"] = derived_data

    # Print class distribution and absolute counts for verification
    print(
        f"\nClass distribution for derived_data_{i} (Minority class ratio: {int(desired_minority_class_ratio * 100)}%):"
    )
    print(derived_data[target_column].value_counts(normalize=True))
    print(f"\nClass absolute count for derived_data_{i}:")
    print(derived_data[target_column].value_counts())

# Step 9: assign variables for derived datasets
derived_data_1 = derived_datasets["derived_data_1"]
derived_data_2 = derived_datasets["derived_data_2"]
derived_data_3 = derived_datasets["derived_data_3"]

# 2.3 Create Train-Validation-Test split for derived datasets
# Create dictionaries for the training and test splits
train_sets, test_sets = {}, {}


# Function to stratify split on each derived dataset
def stratified_split(data, target_column, train_ratio=0.7, test_ratio=0.3):
    """
    Splits a dataset into training and test sets while maintaining the same
    class distribution as the original dataset using stratified sampling.

    Parameters:
    - data (DataFrame): The dataset to be split.
    - target_column (str): The name of the target column used for stratification.
    - train_ratio (float): Proportion of data to include in the Train set (set to 0.7).
    - test_ratio (float): Proportion of data to include in the Test set (set to 0.3).

    Returns:
    - train_data (DataFrame): stratified training set.
    - test_data (DataFrame): stratified test set.
    """
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # Group the data by class and split by class, ensuring no data points are shared between the splits
    for class_label, group in data.groupby(target_column):
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(train_ratio * len(group))
        train_data = pd.concat([train_data, group.iloc[:train_size]])
        test_data = pd.concat([test_data, group.iloc[train_size:]])

    # Shuffle each set after stratified sampling to reduce data order bias after concatination
    return (
        train_data.sample(frac=1, random_state=42).reset_index(drop=True),
        test_data.sample(frac=1, random_state=42).reset_index(drop=True),
    )

    # Split each derived dataset with stratified split function


for i, derived_data in enumerate(
    [derived_data_1, derived_data_2, derived_data_3], start=1
):
    train_set, test_set = stratified_split(derived_data, target_column)

    # Store the splits in dictionaries
    train_sets[f"train_data_{i}"] = train_set
    test_sets[f"test_data_{i}"] = test_set

    # Print class distribution and absolute counts for verification
    print(f"\nClass distribution for train_data_{i}:")
    print(train_set[target_column].value_counts(normalize=True))
    print(f"\nClass absolute count for train_data_{i}:")
    print(train_set[target_column].value_counts())

    print(f"\nClass distribution for test_data_{i}:")
    print(test_set[target_column].value_counts(normalize=True))
    print(f"\nClass absolute count for test_data_{i}:")
    print(test_set[target_column].value_counts())

    # Assign variables for derived datasets
train_data_1, test_data_1 = train_sets["train_data_1"], test_sets["test_data_1"]
train_data_2, test_data_2 = train_sets["train_data_2"], test_sets["test_data_2"]
train_data_3, test_data_3 = train_sets["train_data_3"], test_sets["test_data_3"]

end_time = time.time()

# caiculate  data processing execution time
execution_time = end_time - start_time
total_execution_time += execution_time
print(f"data processing execution time: {execution_time:.4f} second")
print(f"total execution time: {total_execution_time:.4f} second")

# ------------------------------------------------- 2 model -------------------------------------------------
print(
    "------------------------------------------------- 2 model -------------------------------------------------"
)
start_time = time.time()

# LR
# Feature and Target Separation (for dataset_1 and dataset_2)
target_column = "hospital_death"


def prepare_data(dataset):
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    return X, y


X_train_1, y_train_1 = prepare_data(train_data_1)
X_train_2, y_train_2 = prepare_data(train_data_2)
X_test_1, y_test_1 = prepare_data(test_data_1)
X_test_2, y_test_2 = prepare_data(test_data_2)

# Standardize Numerical Features
scaler = StandardScaler()
X_train_1_scaled = scaler.fit_transform(X_train_1)
X_train_2_scaled = scaler.fit_transform(X_train_2)
X_test_1_scaled = scaler.transform(X_test_1)
X_test_2_scaled = scaler.transform(X_test_2)

# Logistic Regression Model
logreg = LogisticRegression(solver="liblinear", random_state=42)

# Hyperparameter grid for Grid Search
param_grid = {
    "C": [0.1, 1, 10],  # Regularization strength
    "solver": ["liblinear", "lbfgs"],  # Solver types
    "max_iter": [100, 200],  # Maximum number of iterations for convergence
}

# Initialize variables to track best model and AUC score
best_auc = 0
best_params = None


# Function to perform 5-fold cross-validation manually
def manual_cross_validation(X, y, model, n_splits=5):
    auc_scores = []
    fold_size = len(X) // n_splits
    for i in range(n_splits):
        start, end = i * fold_size, (i + 1) * fold_size
        X_train_fold = np.concatenate([X[:start], X[end:]], axis=0)
        y_train_fold = np.concatenate([y[:start], y[end:]], axis=0)
        X_val_fold, y_val_fold = X[start:end], y[start:end]

        model.fit(X_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        auc_score = roc_auc_score(y_val_fold, y_pred_proba)
        auc_scores.append(auc_score)

    return np.mean(auc_scores)


# Grid Search (Manually for each dataset)
for C in param_grid["C"]:
    for solver in param_grid["solver"]:
        for max_iter in param_grid["max_iter"]:
            # Initialize model with specific hyperparameters
            logreg = LogisticRegression(
                C=C, solver=solver, max_iter=max_iter, random_state=42
            )

            # Evaluate on the imbalanced dataset (train_data_1)
            auc_score_1 = manual_cross_validation(X_train_1_scaled, y_train_1, logreg)

            # Evaluate on the balanced dataset (train_data_2)
            auc_score_2 = manual_cross_validation(X_train_2_scaled, y_train_2, logreg)

            # Print the AUC scores for both datasets
            print(
                f"AUC (Imbalanced - dataset_1): {auc_score_1:.4f} | AUC (Balanced - dataset_2): {auc_score_2:.4f}"
            )

            # Update best model if necessary based on the AUC score on the balanced dataset
            if auc_score_2 > best_auc:
                best_auc = auc_score_2
                best_params = {"C": C, "solver": solver, "max_iter": max_iter}

# Print the best hyperparameters and AUC score
print(f"\nBest Hyperparameters: {best_params}")
print(f"Best AUC-ROC Score (Balanced Dataset): {best_auc:.4f}")

# Train the best model on the entire training data (using the balanced dataset)
best_logreg = LogisticRegression(
    C=best_params["C"],
    solver=best_params["solver"],
    max_iter=best_params["max_iter"],
    random_state=42,
)
best_logreg.fit(X_train_2_scaled, y_train_2)

# Evaluate on the imbalanced test dataset (test_data_1)
y_pred_proba_1 = best_logreg.predict_proba(X_test_1_scaled)[:, 1]
auc_score_1_final = roc_auc_score(y_test_1, y_pred_proba_1)
print(f"Final AUC-ROC Score (Test Set - Imbalanced): {auc_score_1_final:.4f}")

# Evaluate on the balanced test dataset (test_data_2)
y_pred_proba_2 = best_logreg.predict_proba(X_test_2_scaled)[:, 1]
auc_score_2_final = roc_auc_score(y_test_2, y_pred_proba_2)
print(f"Final AUC-ROC Score (Test Set - Balanced): {auc_score_2_final:.4f}")

# Plot the AUC-ROC curve for the imbalanced dataset (test_data_1)
fpr_1, tpr_1, thresholds_1 = roc_curve(y_test_1, y_pred_proba_1)
plt.figure(figsize=(8, 6))
plt.plot(fpr_1, tpr_1, color="blue", label="Logistic Regression (Imbalanced)")
plt.plot(
    [0, 1], [0, 1], color="gray", linestyle="--"
)  # Random classifier (diagonal line)
plt.title("AUC-ROC Curve (Imbalanced Dataset)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("LR data 1", format="jpg", dpi=300)
# plt.show()

# Plot the AUC-ROC curve for the balanced dataset (test_data_2)
fpr_2, tpr_2, thresholds_2 = roc_curve(y_test_2, y_pred_proba_2)
plt.figure(figsize=(8, 6))
plt.plot(fpr_2, tpr_2, color="green", label="Logistic Regression (Balanced)")
plt.plot(
    [0, 1], [0, 1], color="gray", linestyle="--"
)  # Random classifier (diagonal line)
plt.title("AUC-ROC Curve (Balanced Dataset)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("LR data 2", format="jpg", dpi=300)
# plt.show()
execution_time = end_time - start_time
total_execution_time += execution_time
print(f"random forest data 2 execution time: {execution_time:.4f} second")
print(f"total execution time: {total_execution_time:.4f} second")


print("-----KNN-----")


def split_data(train_data, test_data):
    """
    Split train and test data into features and target.
    """
    X_train, y_train = (
        train_data.drop(columns=["hospital_death"]),
        train_data["hospital_death"],
    )
    X_test, y_test = (
        test_data.drop(columns=["hospital_death"]),
        test_data["hospital_death"],
    )
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
    weights_options = ["uniform", "distance"]
    metric_options = ["minkowski", "manhattan"]

    results = []

    print("Starting parameter search...")

    for n_neighbors in n_neighbors_options:
        for weights in weights_options:
            for metric in metric_options:
                try:
                    # Define the kNN model
                    model = KNeighborsClassifier(
                        n_neighbors=n_neighbors, weights=weights, metric=metric
                    )

                    # Perform cross-validation
                    auc_scores = custom_cross_val_score(model, X, y, cv=cv)
                    mean_auc = np.mean(auc_scores)

                    print(
                        f"Parameters: n_neighbors={n_neighbors}, weights={weights}, metric={metric}"
                    )
                    print(f"Cross-Validation Mean AUC: {mean_auc:.6f}")

                    # Append results
                    results.append(
                        {
                            "n_neighbors": n_neighbors,
                            "weights": weights,
                            "metric": metric,
                            "mean_auc": mean_auc,
                        }
                    )

                except Exception as e:
                    print(
                        f"Error with n_neighbors={n_neighbors}, weights={weights}, metric={metric}: {e}"
                    )
                    results.append(
                        {
                            "n_neighbors": n_neighbors,
                            "weights": weights,
                            "metric": metric,
                            "mean_auc": np.nan,  # Record NaN for invalid configurations
                        }
                    )

    results_df = pd.DataFrame(results)
    return results_df


def plot_results(results_df):
    """
    Plot parameter search results for visualization.
    """
    for weights in results_df["weights"].unique():
        for metric in results_df["metric"].unique():
            subset = results_df[
                (results_df["weights"] == weights) & (results_df["metric"] == metric)
            ]
            plt.plot(
                subset["n_neighbors"],
                subset["mean_auc"],
                label=f"Weights={weights}, Metric={metric}",
            )

    plt.xlabel("Number of Neighbors (n_neighbors)")
    plt.ylabel("Mean AUC")
    plt.title("Parameter Search Results")
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
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    print("Script execution started...")

    datasets = [
        {"train_data": train_data_1, "test_data": test_data_1, "name": "Dataset 1"},
        {"train_data": train_data_2, "test_data": test_data_2, "name": "Dataset 2"},
    ]

    for dataset in datasets:
        print(f"\nProcessing {dataset['name']}...")
        try:
            train_data = dataset["train_data"]
            test_data = dataset["test_data"]

            print("Data loaded successfully!")
            print(f"Train data shape: {train_data.shape}")
            print(f"Test data shape: {test_data.shape}")

            X_train, y_train, X_test, y_test = split_data(train_data, test_data)

            print("Training and testing...")

            results_df = search_best_parameter_with_results(X_train, y_train)
            best_params = results_df.loc[results_df["mean_auc"].idxmax()]
            print(f"Best Parameters: {best_params}")

            best_model = KNeighborsClassifier(
                n_neighbors=int(best_params["n_neighbors"]),
                weights=best_params["weights"],
                metric=best_params["metric"],
            )
            best_model.fit(X_train, y_train)

            test_auc, fpr, tpr = testing_data(best_model, X_test, y_test)
            print(f"Test AUC for {dataset['name']}: {test_auc:.4f}")
            print_roc_curve(test_auc, fpr, tpr, dataset["name"])

            print(f"{dataset['name']} processing completed.")

        except Exception as e:
            print(f"An error occurred while processing {dataset['name']}: {e}")

    print("Script execution completed.")


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


def calculate_auc(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return auc(fpr, tpr), fpr, tpr


def custom_cross_val_score(model, X, y, cv=5):
    """
    Perform cross-validation manually without using existing libraries.

    Args:
        model: Machine learning model with fit and predict methods.
        X (array-like): Feature matrix (Pandas DataFrame or NumPy array).
        y (array-like): Target vector (Pandas Series or NumPy array).
        cv (int): Number of cross-validation folds.

    Returns:
        scores (list): List of AUC scores for each fold.
    """
    # Convert to NumPy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Shuffle data
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

        # Fit model and predict probabilities
        model.fit(X_train, y_train)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate AUC
        auc = roc_auc_score(y_val, y_val_pred_proba)
        scores.append(auc)

    return scores


def search_best_parameter(X, y, cv=5):
    # Parameter settings
    n_estimators_range = [100, 150, 200]
    max_depth_range = [10, 15, 20]

    best_auc = 0
    best_params = None
    best_model = None
    param_results = []

    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            # Model
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

            # Custom cross-validation
            auc_scores = custom_cross_val_score(model, X, y, cv=cv)
            mean_auc = np.mean(auc_scores)

            # Record the parameters and their corresponding AUC
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


def plot_results(param_results):
    # Extract parameters and AUC values
    n_estimators = [result["n_estimators"] for result in param_results]
    max_depth = [result["max_depth"] for result in param_results]
    mean_auc = [result["mean_auc"] for result in param_results]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        n_estimators, max_depth, c=mean_auc, cmap="viridis", s=100, edgecolor="k"
    )
    plt.colorbar(scatter, label="Mean AUC")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Max Depth")
    plt.title("Grid Search Results (AUC)")
    plt.show()


# testing with best parameter
def testing_data(best_model, X_test, y_test):
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc, fpr, tpr = calculate_auc(y_test, y_test_pred_proba)
    print(f"Test AUC: {test_auc:.4f}")
    return test_auc, fpr, tpr


def print_roc_curve(test_auc, fpr, tpr, file_name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    # Save the figure as a JPG file
    plt.savefig(file_name, dpi=300)

    # plt.show() # maybe I should delete this, or the code will not continue exexuting without closing the picture's window


# random forest
print("-----random forest-----")
# data 01
print("-----random forest - data 01-----")
X_train, y_train, X_test, y_test = split_data(train_data_1, test_data_1)
best_model, param_results = search_best_parameter(X_train, y_train, 3)
test_auc, fpr, tpr = testing_data(best_model, X_test, y_test)
print_roc_curve(test_auc, fpr, tpr, "random-forest-data01")
end_time = time.time()

# caiculate execution time
execution_time = end_time - start_time
total_execution_time += execution_time
print(f"random forest data 1 execution time: {execution_time:.4f} second")
print(f"total execution time: {total_execution_time:.4f} second")

# data 02
print("-----random forest - data 02-----")
start_time = time.time()

X_train, y_train, X_test, y_test = split_data(train_data_2, test_data_2)
best_model, param_results = search_best_parameter(X_train, y_train, 3)
test_auc, fpr, tpr = testing_data(best_model, X_test, y_test)
print_roc_curve(test_auc, fpr, tpr, "random-forest-data02")

end_time = time.time()

# caiculate data processing execution time
execution_time = end_time - start_time
total_execution_time += execution_time
print(f"random forest data 2 execution time: {execution_time:.4f} second")
print(f"total execution time: {total_execution_time:.4f} second")
