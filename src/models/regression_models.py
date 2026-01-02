"""
Regression models for predicting agriculture and development metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from pathlib import Path


class AgricultureDevelopmentRegressor:
    """Regression models for agriculture and development prediction"""
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.model_scores = {}
        
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train all regression models
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of test set
            random_state: Random seed
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
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
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print()
        
        self.trained_models = results
        return results
    
    def get_best_model(self, metric='test_r2'):
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
        """
        Save a trained model to disk
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models.")
        
        model = self.trained_models[model_name]['model']
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        return joblib.load(filepath)


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
    
    # Prepare data for ML
    X, y = preprocessor.prepare_for_ml(merged_data, target_column='Agricultural_GDP_crores')
    
    # Train models
    regressor = AgricultureDevelopmentRegressor()
    results = regressor.train(X, y)
    
    # Get best model
    best_name, best_model = regressor.get_best_model()
    print(f"\nBest model: {best_name}")
    
    # Feature importance
    if best_name in ['random_forest', 'gradient_boosting']:
        importance = regressor.get_feature_importance(best_name)
        print("\nTop 10 Important Features:")
        print(importance)

