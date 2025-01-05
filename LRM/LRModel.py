import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Load the preprocessed data from CSV files
train_data_1 = pd.read_csv('/Users/aragoto/Desktop/FML/dataset1and3/train_data_1.csv')  # Imbalanced data (90% | 10%)
train_data_3 = pd.read_csv('/Users/aragoto/Desktop/FML/dataset1and3/train_data_3.csv')  # Balanced data (50% | 50%)
test_data_1 = pd.read_csv('/Users/aragoto/Desktop/FML/dataset1and3/test_data_1.csv')  # Imbalanced data (90% | 10%)
test_data_3 = pd.read_csv('/Users/aragoto/Desktop/FML/dataset1and3/test_data_3.csv')  # Balanced data (50% | 50%)

# Feature and Target Separation (for dataset_1 and dataset_3)
target_column = 'hospital_death'


def prepare_data(dataset):
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    return X, y


X_train_1, y_train_1 = prepare_data(train_data_1)
X_train_3, y_train_3 = prepare_data(train_data_3)
X_test_1, y_test_1 = prepare_data(test_data_1)
X_test_3, y_test_3 = prepare_data(test_data_3)

# Standardize Numerical Features
scaler = StandardScaler()
X_train_1_scaled = scaler.fit_transform(X_train_1)
X_train_3_scaled = scaler.fit_transform(X_train_3)
X_test_1_scaled = scaler.transform(X_test_1)
X_test_3_scaled = scaler.transform(X_test_3)

# Logistic Regression Model
logreg = LogisticRegression(solver='liblinear', random_state=42)

# Hyperparameter grid for Grid Search
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'lbfgs'],  # Solver types
    'max_iter': [100, 200]  # Maximum number of iterations for convergence
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
for C in param_grid['C']:
    for solver in param_grid['solver']:
        for max_iter in param_grid['max_iter']:
            # Initialize model with specific hyperparameters
            logreg = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)

            # Evaluate on the imbalanced dataset (train_data_1)
            auc_score_1 = manual_cross_validation(X_train_1_scaled, y_train_1, logreg)

            # Evaluate on the balanced dataset (train_data_3)
            auc_score_3 = manual_cross_validation(X_train_3_scaled, y_train_3, logreg)

            # Print the AUC scores for both datasets
            print(f"AUC (Imbalanced - dataset_1): {auc_score_1:.4f} | AUC (Balanced - dataset_3): {auc_score_3:.4f}")

            # Update best model if necessary based on the AUC score on the balanced dataset
            if auc_score_3 > best_auc:
                best_auc = auc_score_3
                best_params = {'C': C, 'solver': solver, 'max_iter': max_iter}

# Print the best hyperparameters and AUC score
print(f"\nBest Hyperparameters: {best_params}")
print(f"Best AUC-ROC Score (Balanced Dataset): {best_auc:.4f}")

# Train the best model on the entire training data (using the balanced dataset)
best_logreg = LogisticRegression(C=best_params['C'], solver=best_params['solver'], max_iter=best_params['max_iter'],
                                 random_state=42)
best_logreg.fit(X_train_3_scaled, y_train_3)

# Evaluate on the imbalanced test dataset (test_data_1)
y_pred_proba_1 = best_logreg.predict_proba(X_test_1_scaled)[:, 1]
auc_score_1_final = roc_auc_score(y_test_1, y_pred_proba_1)
print(f"Final AUC-ROC Score (Test Set - Imbalanced): {auc_score_1_final:.4f}")

# Evaluate on the balanced test dataset (test_data_3)
y_pred_proba_3 = best_logreg.predict_proba(X_test_3_scaled)[:, 1]
auc_score_3_final = roc_auc_score(y_test_3, y_pred_proba_3)
print(f"Final AUC-ROC Score (Test Set - Balanced): {auc_score_3_final:.4f}")

# Plot the AUC-ROC curve for the imbalanced dataset (test_data_1)
fpr_1, tpr_1, thresholds_1 = roc_curve(y_test_1, y_pred_proba_1)
plt.figure(figsize=(8, 6))
plt.plot(fpr_1, tpr_1, color='blue', label='Logistic Regression (Imbalanced)')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier (diagonal line)
plt.title('AUC-ROC Curve (Imbalanced Dataset)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Plot the AUC-ROC curve for the balanced dataset (test_data_3)
fpr_3, tpr_3, thresholds_3 = roc_curve(y_test_3, y_pred_proba_3)
plt.figure(figsize=(8, 6))
plt.plot(fpr_3, tpr_3, color='green', label='Logistic Regression (Balanced)')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier (diagonal line)
plt.title('AUC-ROC Curve (Balanced Dataset)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()