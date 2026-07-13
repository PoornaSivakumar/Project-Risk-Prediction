"""
Logistic Regression Model for Project Risk Prediction
======================================================

This script builds a logistic regression model to predict risk classification
from the risk-dataset.csv file. The model is trained on customer order data
and predicts whether an order is at risk (yes/no).

Author: Automated Model Generation
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings('ignore')


class RiskPredictionModel:
    """
    A class to handle logistic regression model for risk prediction.
    
    Attributes:
        model (LogisticRegression): The trained logistic regression model
        scaler (StandardScaler): The fitted scaler for feature normalization
        label_encoders (dict): Dictionary of label encoders for categorical features
        feature_names (list): Names of features used in the model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target values
    """
    
    def __init__(self, random_state=42):
        """Initialize the model components."""
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        
    def load_data(self, filepath):
        """
        Load data from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def preprocess_data(self, df, target_column='CLASS'):
        """
        Preprocess the data by handling missing values and encoding categorical variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X, y) preprocessed features and target
        """
        print("\nPreprocessing data...")
        
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Separate target and features
        y = df[target_column].copy()
        X = df.drop(target_column, axis=1).copy()
        
        # Encode target variable
        self.label_encoders['target'] = LabelEncoder()
        y = self.label_encoders['target'].fit_transform(y)
        
        # Handle missing values - replace '?' with NaN then drop or fill
        X = X.replace('?', np.nan)
        
        # Fill missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                # For categorical columns, fill with mode
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
            else:
                # For numeric columns, fill with median
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"Categorical columns found: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Convert all columns to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Drop any remaining NaN values
        X = X.fillna(X.mean())
        
        self.feature_names = X.columns.tolist()
        print(f"Features after preprocessing: {len(self.feature_names)}")
        print(f"Target classes: {self.label_encoders['target'].classes_}")
        
        return X, y
    
    def train(self, X_train, y_train):
        """
        Train the logistic regression model.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (pd.Series or np.ndarray): Training target
        """
        print("\nTraining model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Store training data for evaluation
        self.X_train = X_train
        self.y_train = y_train
        
        print("Model training completed!")
        print(f"Model parameters: {self.model.get_params()}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame or np.ndarray): Test features
            y_test (pd.Series or np.ndarray): Test target
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\nEvaluating model...")
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoders['target'].classes_))
        
        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features for prediction
            
        Returns:
            tuple: (predictions, probabilities)
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        print(f"\nSaving model to {filepath}...")
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print("Model saved successfully!")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a saved model from a file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            dict: Loaded model components
        """
        print(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print("Model loaded successfully!")
        return model_data
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm (np.ndarray): Confusion matrix
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoders['target'].classes_,
                   yticklabels=self.label_encoders['target'].classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_test (np.ndarray): True test labels
            y_pred_proba (np.ndarray): Predicted probabilities
            save_path (str, optional): Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"ROC curve plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to run the logistic regression model."""
    
    # Configuration
    CSV_FILE = 'risk-dataset.csv'
    MODEL_FILE = 'risk_prediction_model.pkl'
    
    # Initialize model
    model = RiskPredictionModel(random_state=42)
    
    # Load data
    df = model.load_data(CSV_FILE)
    
    # Preprocess data
    X, y = model.preprocess_data(df, target_column='CLASS')
    
    # Split data
    print("\nSplitting data into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    model.plot_confusion_matrix(metrics['confusion_matrix'], 
                               save_path='confusion_matrix.png')
    
    # Plot ROC curve
    model.plot_roc_curve(y_test, metrics['y_pred_proba'], 
                        save_path='roc_curve.png')
    
    # Save model
    model.save_model(MODEL_FILE)
    
    print("\n" + "="*50)
    print("Model training and evaluation completed!")
    print("="*50)


if __name__ == '__main__':
    main()
