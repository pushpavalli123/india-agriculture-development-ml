"""
Classification models for categorizing states by development level
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path


class DevelopmentClassifier:
    """Classification models for state development categorization"""
    
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        self.trained_models = {}
        
    def create_development_categories(self, df, target_column='HDI', n_categories=3):
        """
        Create development categories based on a target column
        
        Args:
            df: Input DataFrame
            target_column: Column to use for categorization
            n_categories: Number of categories (3: Low, Medium, High)
            
        Returns:
            Series with categories
        """
        if target_column not in df.columns:
            raise ValueError(f"Column {target_column} not found in DataFrame")
        
        if n_categories == 3:
            thresholds = [
                df[target_column].quantile(0.33),
                df[target_column].quantile(0.67)
            ]
            categories = pd.cut(
                df[target_column],
                bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
                labels=['Low', 'Medium', 'High']
            )
        elif n_categories == 2:
            threshold = df[target_column].median()
            categories = pd.cut(
                df[target_column],
                bins=[-np.inf, threshold, np.inf],
                labels=['Low', 'High']
            )
        else:
            # Equal width binning
            categories = pd.qcut(
                df[target_column],
                q=n_categories,
                labels=[f'Category_{i+1}' for i in range(n_categories)]
            )
        
        return categories
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train all classification models
        
        Args:
            X: Feature matrix
            y: Target categories
            test_size: Proportion of test set
            random_state: Random seed
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_test_pred),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred)
            }
            
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print()
        
        self.trained_models = results
        return results
    
    def get_best_model(self, metric='test_accuracy'):
        """
        Get the best performing model
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Best model name and model object
        """
        if not self.trained_models:
            raise ValueError("No models trained yet. Call train() first.")
        
        best_name = max(self.trained_models.keys(), 
                       key=lambda x: self.trained_models[x][metric])
        best_model = self.trained_models[best_name]['model']
        
        return best_name, best_model
    
    def predict(self, X, model_name=None):
        """
        Make predictions using a trained model
        
        Args:
            X: Feature matrix
            model_name: Name of model to use (None for best model)
            
        Returns:
            Predictions array
        """
        if not self.trained_models:
            raise ValueError("No models trained yet. Call train() first.")
        
        if model_name is None:
            model_name, model = self.get_best_model()
        else:
            model = self.trained_models[model_name]['model']
        
        return model.predict(X)
    
    def predict_proba(self, X, model_name=None):
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix
            model_name: Name of model to use (None for best model)
            
        Returns:
            Probability array
        """
        if not self.trained_models:
            raise ValueError("No models trained yet. Call train() first.")
        
        if model_name is None:
            model_name, model = self.get_best_model()
        else:
            model = self.trained_models[model_name]['model']
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError(f"Model {model_name} does not support predict_proba.")
    
    def get_feature_importance(self, model_name='random_forest', top_n=10):
        """
        Get feature importance from tree-based models
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models.")
        
        model = self.trained_models[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            raise ValueError(f"Model {model_name} does not support feature importance.")
    
    def save_model(self, model_name, filepath):
        """Save a trained model to disk"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models.")
        
        model = self.trained_models[model_name]['model']
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append('..')
    from data_loader import IndiaDataLoader
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    loader = IndiaDataLoader()
    preprocessor = DataPreprocessor()
    
    agri_data = loader.load_agriculture_data()
    dev_data = loader.load_development_data()
    merged_data = loader.merge_data(agri_data, dev_data)
    
    # Create development categories
    classifier = DevelopmentClassifier()
    categories = classifier.create_development_categories(merged_data, target_column='HDI')
    
    # Prepare data for ML
    X, y = preprocessor.prepare_for_ml(merged_data, target_column=None)
    y = categories[X.index] if len(categories) == len(X) else categories
    
    # Train models
    results = classifier.train(X, y)
    
    # Get best model
    best_name, best_model = classifier.get_best_model()
    print(f"\nBest model: {best_name}")

