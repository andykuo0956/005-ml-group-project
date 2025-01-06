from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import category_encoders as ce
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# input library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
num_samples = len(master_data) // 2

# Group data by target class
grouped = master_data.groupby(target_column)

# Initialise a dictionary to store sampled datasets
sampled_datasets = {
    "sampled_data_1": pd.DataFrame(),
    "sampled_data_2": pd.DataFrame(),
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

# Print summaries of each dataset to verify proportion of classes and absolute count of data values
for name, data in sampled_datasets.items():
    print(f"\n{name} class distribution:")
    print(data[target_column].value_counts(normalize=True))
    print(f"\n{name} class absolute count:")
    print(data[target_column].value_counts())

# 2.2 Create class-imbalanced derived dataset using SMOTE-NC (Nominal Continuous)
# Desired rations for the minority class
desired_minority_class_ratios = [0.10, 0.50]

# Create empty dictionary for derived datasets
derived_datasets = {}

for i, (sampled_data, desired_minority_class_ratio) in enumerate(
    zip([sampled_data_1, sampled_data_2], desired_minority_class_ratios), start=1
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


for i, derived_data in enumerate([derived_data_1, derived_data_2], start=1):
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

end_time = time.time()

# caiculate  data processing execution time
execution_time = end_time - start_time
total_execution_time += execution_time
print(f"data processing execution time: {execution_time:.4f} second")
print(f"total execution time: {total_execution_time:.4f} second")

# Common Functions
def split_data(df, target_column="hospital_death", test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Parameters:
    - df (DataFrame): The entire dataset.
    - target_column (str): Name of the target column.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - train_data (DataFrame): Training data.
    - test_data (DataFrame): Testing data.
    """
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_column])
    return train_data, test_data

def calculate_auc(y_true, y_pred_proba):
    """
    Calculate the AUC score and ROC curve values.

    Parameters:
    - y_true (array-like): True target values.
    - y_pred_proba (array-like): Predicted probabilities for the positive class.

    Returns:
    - auc_score (float): Calculated AUC score.
    - fpr (array): False positive rates.
    - tpr (array): True positive rates.
    """
    auc_score = roc_auc_score(y_true, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return auc_score, fpr, tpr

def plot_roc_curve(fpr, tpr, model_name):
    """
    Plot the ROC curve.

    Parameters:
    - fpr (array): False positive rates.
    - tpr (array): True positive rates.
    - model_name (str): Name of the model for labeling.
    """
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve ({model_name})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid()
    plt.show()

# Logistic Regression Pipeline
def logistic_regression_pipeline(train_data, test_data, target_column="hospital_death"):
    """
    Train and evaluate a Logistic Regression model.

    Parameters:
    - train_data (DataFrame): Training dataset.
    - test_data (DataFrame): Test dataset.
    - target_column (str): Target column for classification.

    Returns:
    - model (LogisticRegression): Trained model.
    - test_auc (float): AUC score on the test data.
    - fpr (array): False positive rates.
    - tpr (array): True positive rates.
    """
    X_train, y_train = train_data.drop(columns=[target_column]), train_data[target_column]
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate Logistic Regression
    y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_auc, fpr, tpr = calculate_auc(y_test, y_test_pred_proba)
    return model, test_auc, fpr, tpr

# Random Forest Pipeline
def random_forest_pipeline(train_data, test_data, target_column="hospital_death"):
    """
    Train and evaluate a Random Forest model.

    Parameters:
    - train_data (DataFrame): Training dataset.
    - test_data (DataFrame): Test dataset.
    - target_column (str): Target column for classification.

    Returns:
    - model (RandomForestClassifier): Trained model.
    - test_auc (float): AUC score on the test data.
    - fpr (array): False positive rates.
    - tpr (array): True positive rates.
    """
    X_train, y_train = train_data.drop(columns=[target_column]), train_data[target_column]
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]

    # Train Random Forest
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate Random Forest
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc, fpr, tpr = calculate_auc(y_test, y_test_pred_proba)
    return model, test_auc, fpr, tpr

# Example Workflow
def main_workflow():
    # Load dataset (replace with your dataset)
    df = pd.read_csv("your_dataset.csv")

    # Split data
    train_data, test_data = split_data(df, target_column="hospital_death")

    # Logistic Regression
    lr_model, lr_auc, lr_fpr, lr_tpr = logistic_regression_pipeline(train_data, test_data)
    print(f"Logistic Regression Test AUC: {lr_auc:.4f}")
    plot_roc_curve(lr_fpr, lr_tpr, "Logistic Regression")

    # Random Forest
    rf_model, rf_auc, rf_fpr, rf_tpr = random_forest_pipeline(train_data, test_data)
    print(f"Random Forest Test AUC: {rf_auc:.4f}")
    plot_roc_curve(rf_fpr, rf_tpr, "Random Forest")

if __name__ == "__main__":
    main_workflow()
