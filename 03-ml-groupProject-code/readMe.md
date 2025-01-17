# Robustness of Machine Learning Classification Models to Imbalanced In-Hospital Mortality Data

## Project Overview

This project investigates how class imbalance in training and testing datasets impacts the predictive performance of three models: Logistic Regression (LR), K-Nearest Neighbors (KNN), and Random Forest Classifier (RFC). LR assumes linear decision boundaries, KNN utilizes local instance-based learning, and RFC captures complex patterns with ensemble methods. The study addresses two key questions: how varying class imbalances affect true positive and false positive trade-offs (measured by AUC) and how linear and nonlinear models handle imbalance under identical conditions. The findings have practical implications for predicting in-hospital patient mortality.

## Dataset Description

The dataset is downloaded in https://www.kaggle.com/datasets/mitishaagarwal/patient and organized into:

1. Raw Dataset:
   dataset.csv contains the initial unprocessed data.
2. Preprocessed Datasets:
   These datasets include training (\_train.csv) and testing (\_test.csv) splits for three different variations: data_1, data_2, and data_3.

Each variation represents different degree of imbalance.

- data_1: 10% of the patient died
  (hospital death = 1) and 90% of the patient survived (hospi-
  tal death = 0)
- data_2: 30% of the patient died
  (hospital death = 1) and 70% of the patient survived (hospi-
  tal death = 0)
- data_3: 50% of the patient died
  (hospital death = 1) and 50% of the patient survived (hospi-
  tal death = 0)

## Methodology

The methodology followed these steps:

1. Data Preprocessing:

   The raw dataset was cleaned, transformed, and split into training and testing sets.
   Three variations of preprocessing were implemented to test the sensitivity of models to data preparation.

2. Model Implementation:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

3. Model Evaluation:

   The performance of each model was assessed using AUC scores.

## Folder Structure

├── 00-raw-dataset<br>
│ └── dataset.csv # Original raw dataset<br>
├── 01-dataset-preprocessed<br>
│ ├── data_1_train.csv # Preprocessed training dataset 1<br>
│ ├── data_1_test.csv # Preprocessed testing dataset 1<br>
│ ├── data_2_train.csv # Preprocessed training dataset 2<br>
│ ├── data_2_test.csv # Preprocessed testing dataset 2<br>
│ ├── data_3_train.csv # Preprocessed training dataset 3<br>
│ └── data_3_test.csv # Preprocessed testing dataset 3<br>
├── 02-image-output<br>
│ ├── knn-combined-roc-data1.jpg # KNN combined ROC curve for dataset 1<br>
│ ├── knn-combined-roc-data2.jpg # KNN combined ROC curve for dataset 2<br>
│ ├── knn-combined-roc-data3.jpg # KNN combined ROC curve for dataset 3<br>
│ ├── logistic_regression-combined-roc-data1.jpg # Logistic Regression combined ROC curve for dataset 1<br>
│ ├── logistic_regression-combined-roc-data2.jpg # Logistic Regression combined ROC curve for dataset 2<br>
│ ├── logistic_regression-combined-roc-data3.jpg # Logistic Regression combined ROC curve for dataset 3<br>
│ ├── random_forest-combined-roc-data1.jpg # Random Forest combined ROC curve for dataset 1<br>
│ ├── random_forest-combined-roc-data2.jpg # Random Forest combined ROC curve for dataset 2<br>
│ └── random_forest-combined-roc-data3.jpg # Random Forest combined ROC curve for dataset 3<br>
├── main.py # Main entry point for the project<br>
├── model_knn.py # K-Nearest Neighbors model implementation<br>
├── model_lr.py # Logistic Regression model implementation<br>
├── model_rfc.py # Random Forest model implementation<br>
├── preprocessing.py # Data preprocessing script<br>
├── readMe.md # Project documentation<br>
└── requirements.txt # Required libraries<br>

## How to Run the Project

### Prerequisites

1. Python 3.8 or later.
2. Install the required libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the version of scikit-learn is 1.5.2

### Running the Project

1. Navigate to the project directory.

2. Run the main.py script with the desired model type:
   ```bash
   python main.py --model <model_type>
   ```
   Replace `<model_type>` with one of the following options:

- `logistic_regression`: Run the Logistic Regression model.
- `knn`: Run the K-Nearest Neighbors model.
- `random_forest`: Run the Random Forest model.
- `all`: Run all three models in sequence.

3. Example usage:

   ```bash
   python main.py --model all
   ```

   This will preprocess the data, train the all of the models, and save the combined ROC curve image.

4. Run the individual experiment:
   ```bash
   python <experiment_name>.py
   ```
   Replace `<experiment_name>` with one of the following options:
   - `preprocessing`: Run the preprocessing.
   - `model_lr`: Run the Logistic Regression model.
   - `model_knn`: Run the K-Nearest Neighbors model.
   - `model_rfc`: Run the Random Forest model.

## Expected Runtime

The total runtime for all models is approximately 20-30 minutes, depending on system performance.

### Runtime of each experiment

- Preprocessing: 10 seconds
- Logistic Regression: 10 minutes
- K-Nearest Neighbors (KNN): 1 minute
- Random Forest Classifier: 10 minutes

## Result

### Logistic Regression Model Performance

The Logistic Regression model was trained on three datasets and optimized using manual grid search with 5-fold cross-validation. The best hyperparameter combinations for each dataset showed that for imbalanced datasets, the top combination was C:1.0, solver:liblinear, max_iter:300. For Dataset1 (90:10 imbalance), the model achieved an AUC of 0.8611, which decreased when tested on more balanced datasets, reaching 0.8522 on Dataset2 and 0.8404 on Dataset3. For Dataset2 (70:30), the model performed better with an AUC of 0.8999, and 0.9003 on Dataset3. The highest performance was seen in Dataset3, where a perfectly balanced class distribution led to the highest AUC of 0.9058.

### kNN Model Performance

The kNN algorithm was evaluated on the same three datasets, with grid search used to optimize hyperparameters. For Dataset1, the model achieved an AUC of 0.7772 with 10 neighbors, using the Manhattan metric. The performance improved significantly on Dataset3, with an AUC of 0.9568 using 5 neighbors. For Dataset2, the best performance was with an AUC of 0.9682, achieved with 10 neighbors and the Manhattan metric. Smaller neighbor counts worked better on balanced datasets, while larger counts were beneficial for imbalanced ones.

### Random Forest Classifier Performance

Random Forest models were optimized for each dataset, with the best configuration for Dataset1 showing an AUC of 0.8922 with 200 estimators and a depth of 15. For Dataset2, the highest performance was 0.9744, and for Dataset3, the best AUC reached 0.9907, indicating excellent generalization across different class distributions. The models showed good performance even with extreme class imbalances, highlighting the flexibility of Random Forest in handling various dataset types.

### Key Findings

The degree of class imbalance influenced the performance of all models. Logistic Regression and kNN models showed better results on balanced datasets, while Random Forest performed well across all distributions. Proper hyperparameter tuning and dataset-specific adjustments were crucial for maximizing model accuracy.

## Report

For detailed analysis and results, refer to the project report.
