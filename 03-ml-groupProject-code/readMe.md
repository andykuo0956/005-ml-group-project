# Effect of Training and Testing Machine Learning Models on Dataset Class Imbalances for Predicting Patient Mortality

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
└── readMe.md # Project documentation<br>

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

## Expected Runtime

The total runtime for all models is approximately 25-30 minutes, depending on system performance.

### Runtime of each experiment

- Preprocessing:
- Logistic Regression:
- K-Nearest Neighbors (KNN):
- Random Forest Classifier:

## Result

## Report

For detailed analysis and results, refer to the project report located in
