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
import itertools
from typing import Tuple, List, Dict, Any

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.execution_time = 0
        
    def process_data(self) -> pd.DataFrame:
        start_time = time.time()
        
        print("Loading and cleaning dataset...")
        self.data = pd.read_csv(self.file_path)
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna(axis=1, how="all")
        print("Initial data shape:", self.data.shape)
        print("\nFirst 6 rows:")
        print(self.data.head(6))
        
        print("\nAnalyzing missing values...")
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        missing_report = pd.DataFrame({
            'Column': self.data.columns,
            'Missing Values': missing_values,
            'Missing Percentage (%)': missing_percentage
        }).sort_values(by='Missing Percentage (%)', ascending=False)
        print(missing_report[missing_report['Missing Values'] > 0])
        
        print("\nHandling missing values...")
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        self.data[numeric_cols] = self.data[numeric_cols].apply(lambda col: col.fillna(col.mean()))
        self.data[categorical_cols] = self.data[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))
        
        print("\nRemoving non-relevant features...")
        non_related_cols = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
        self.data = self.data.drop(columns=non_related_cols, errors='ignore')
        
        self.execution_time = time.time() - start_time
        print(f"\nData processing execution time: {self.execution_time:.4f} seconds")
        
        return self.data

class DatasetSplitter:
    @staticmethod
    def create_stratified_samples(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        num_samples = len(data) // 2
        class_distribution = data[target_column].value_counts(normalize=True)
        
        sampled_datasets = {}
        for i in range(2):
            sampled_data = pd.DataFrame()
            for class_label, group in data.groupby(target_column):
                samples = int(class_distribution[class_label] * num_samples)
                sampled_class_data = group.sample(n=samples, random_state=np.random.randint(1000))
                sampled_data = pd.concat([sampled_data, sampled_class_data])
            
            sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
            sampled_datasets[f"sampled_data_{i+1}"] = sampled_data
            
            print(f"\nClass distribution for sampled_data_{i+1}:")
            print(sampled_data[target_column].value_counts(normalize=True))
            print(f"Absolute count for sampled_data_{i+1}:")
            print(sampled_data[target_column].value_counts())
            
        return sampled_datasets["sampled_data_1"], sampled_datasets["sampled_data_2"]

class SMOTEHandler:
    @staticmethod
    def apply_smote(data: pd.DataFrame, target_column: str, minority_ratio: float) -> pd.DataFrame:
        categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
        
        encoder = ce.BinaryEncoder(cols=categorical_cols)
        encoded_data = encoder.fit_transform(data)
        
        X = encoded_data.drop(columns=[target_column])
        y = encoded_data[target_column]
        
        majority_count = y.value_counts().get(0, 0)
        sampling_strategy = int(majority_count * (minority_ratio / (1 - minority_ratio)))
        
        categorical_indices = [X.columns.get_loc(col) for col in X.columns 
                             if any(col.startswith(cat) for cat in categorical_cols)]
        
        smote = SMOTENC(categorical_features=categorical_indices,
                       sampling_strategy={1: sampling_strategy},
                       random_state=42)
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        result = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                          pd.DataFrame(y_resampled, columns=[target_column])], axis=1)
        
        print(f"\nSMOTE results (target ratio: {minority_ratio}):")
        print(result[target_column].value_counts(normalize=True))
        return result

class CustomKFold:
    @staticmethod
    def split(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> List[Tuple]:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        fold_size = len(indices) // n_splits
        splits = []
        
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_splits - 1 else len(indices)
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            splits.append((train_indices, val_indices))
            
        return splits

class ModelTrainer:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.execution_time = 0
        
    def custom_cross_val_score(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> List[float]:
        kf = CustomKFold()
        scores = []
        
        for train_idx, val_idx in kf.split(X, y, cv):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred_proba))
            
        return scores
    
    def grid_search(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict]:
        start_time = time.time()
        
        if self.model_type == "logistic":
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [100, 200]
            }
            base_model = LogisticRegression
            
        elif self.model_type == "random_forest":
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20]
            }
            base_model = RandomForestClassifier
            
        elif self.model_type == "knn":
            param_grid = {
                'n_neighbors': [3, 5, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'manhattan']
            }
            base_model = KNeighborsClassifier
        
        best_score = 0
        best_params = None
        best_model = None
        
        print(f"\nStarting grid search for {self.model_type}...")
        for params in self._generate_param_combinations(param_grid):
            model = base_model(**params, random_state=42 if self.model_type != "knn" else None)
            scores = self.custom_cross_val_score(model, X, y)
            mean_score = np.mean(scores)
            
            print(f"Parameters: {params}")
            print(f"Mean CV Score: {mean_score:.6f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = model
        
        self.execution_time = time.time() - start_time
        print(f"\n{self.model_type} grid search execution time: {self.execution_time:.4f} seconds")
        
        return best_model, best_params
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = []
        
        for items in itertools.product(*values):
            combinations.append(dict(zip(keys, items)))
        
        return combinations

class Visualizer:
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, 
                       title: str, filename: str) -> None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.savefig(filename, dpi=300)
        plt.close()

def main():
    total_time = 0
    
    # Data processing
    processor = DataProcessor("00-raw-dataset/dataset.csv")
    data = processor.process_data()
    total_time += processor.execution_time
    
    # Dataset splitting
    splitter = DatasetSplitter()
    sampled_data_1, sampled_data_2 = splitter.create_stratified_samples(data, "hospital_death")
    
    # SMOTE handling
    smote_handler = SMOTEHandler()
    balanced_data_1 = smote_handler.apply_smote(sampled_data_1, "hospital_death", 0.10)
    balanced_data_2 = smote_handler.apply_smote(sampled_data_2, "hospital_death", 0.50)
    
    # Model training and evaluation
    for model_type in ["logistic", "random_forest", "knn"]:
        trainer = ModelTrainer(model_type)
        
        for dataset_num, dataset in enumerate([balanced_data_1, balanced_data_2], 1):
            print(f"\nProcessing {model_type} on dataset {dataset_num}")
            
            X = dataset.drop(columns=["hospital_death"]).values
            y = dataset["hospital_death"].values
            
            best_model, best_params = trainer.grid_search(X, y)
            print(f"Best parameters: {best_params}")
            
            # Final evaluation
            y_pred_proba = best_model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc_score = auc(fpr, tpr)
            
            print(f"Final AUC score: {auc_score:.4f}")
            
            Visualizer.plot_roc_curve(
                fpr, tpr, auc_score,
                f"{model_type.title()} ROC Curve - Dataset {dataset_num}",
                f"{model_type}_data_{dataset_num}.jpg"
            )
            
            total_time += trainer.execution_time
    
    print(f"\nTotal execution time: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()