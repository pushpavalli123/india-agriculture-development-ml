"""
Data preprocessing utilities for India Agriculture & Development Analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Class for preprocessing agriculture and development data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = None
        
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy in ['mean', 'median']:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy=strategy)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        elif strategy == 'mode':
            for col in df_clean.columns:
                if df_clean[col].isna().any():
                    mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                    df_clean[col].fillna(mode_value, inplace=True)
        
        return df_clean
    
    def encode_categorical(self, df, columns=None):
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode (None for all categorical)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        if columns is None:
            categorical_cols = df_encoded.select_dtypes(include=['object']).columns
            # Exclude 'State' column if present
            categorical_cols = [col for col in categorical_cols if col != 'State']
        else:
            categorical_cols = columns
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, df, columns=None, method='standard'):
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            columns: List of columns to scale (None for all numerical)
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
            # Exclude 'State' column if present
            numeric_cols = [col for col in numeric_cols if col != 'State']
        else:
            numeric_cols = columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        self.feature_names = numeric_cols
        
        return df_scaled
    
    def create_features(self, df):
        """
        Create derived features from existing data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        df_features = df.copy()
        
        # Agriculture productivity features
        if 'Total_Crop_Area_ha' in df_features.columns and 'Rice_Production_tons' in df_features.columns:
            df_features['Rice_Yield_per_ha'] = (
                df_features['Rice_Production_tons'] / df_features['Total_Crop_Area_ha']
            ).replace([np.inf, -np.inf], 0)
        
        if 'Total_Crop_Area_ha' in df_features.columns and 'Wheat_Production_tons' in df_features.columns:
            df_features['Wheat_Yield_per_ha'] = (
                df_features['Wheat_Production_tons'] / df_features['Total_Crop_Area_ha']
            ).replace([np.inf, -np.inf], 0)
        
        # Irrigation efficiency
        if 'Total_Irrigation_Area_ha' in df_features.columns and 'Total_Crop_Area_ha' in df_features.columns:
            df_features['Irrigation_Coverage'] = (
                df_features['Total_Irrigation_Area_ha'] / df_features['Total_Crop_Area_ha']
            ).replace([np.inf, -np.inf], 0)
        
        # Agricultural intensity
        if 'Fertilizer_Usage_tons' in df_features.columns and 'Total_Crop_Area_ha' in df_features.columns:
            df_features['Fertilizer_Intensity'] = (
                df_features['Fertilizer_Usage_tons'] / df_features['Total_Crop_Area_ha']
            ).replace([np.inf, -np.inf], 0)
        
        # Development ratios
        if 'Agricultural_GDP_crores' in df_features.columns and 'Total_GDP_crores' in df_features.columns:
            df_features['Agriculture_GDP_Share'] = (
                df_features['Agricultural_GDP_crores'] / df_features['Total_GDP_crores']
            ).replace([np.inf, -np.inf], 0)
        
        if 'Industrial_GDP_crores' in df_features.columns and 'Total_GDP_crores' in df_features.columns:
            df_features['Industrial_GDP_Share'] = (
                df_features['Industrial_GDP_crores'] / df_features['Total_GDP_crores']
            ).replace([np.inf, -np.inf], 0)
        
        if 'Service_GDP_crores' in df_features.columns and 'Total_GDP_crores' in df_features.columns:
            df_features['Service_GDP_Share'] = (
                df_features['Service_GDP_crores'] / df_features['Total_GDP_crores']
            ).replace([np.inf, -np.inf], 0)
        
        # Per capita metrics
        if 'Rural_Population' in df_features.columns:
            if 'Agricultural_GDP_crores' in df_features.columns:
                df_features['Agri_GDP_Per_Capita'] = (
                    df_features['Agricultural_GDP_crores'] / df_features['Rural_Population']
                ).replace([np.inf, -np.inf], 0)
            
            if 'Farmers_Count' in df_features.columns:
                df_features['Farmers_Percentage'] = (
                    df_features['Farmers_Count'] / df_features['Rural_Population'] * 100
                ).replace([np.inf, -np.inf], 0)
        
        return df_features
    
    def remove_outliers(self, df, columns=None, method='iqr'):
        """
        Remove outliers from the dataset
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (None for all numerical)
            method: Outlier detection method ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        if columns is None:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
            df_clean = df_clean[(z_scores < 3).all(axis=1)]
        
        return df_clean
    
    def prepare_for_ml(self, df, target_column=None, drop_columns=None):
        """
        Prepare data for machine learning
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (if any)
            drop_columns: List of columns to drop
            
        Returns:
            X (features), y (target) if target_column provided, else just X
        """
        df_ml = df.copy()
        
        # Drop specified columns
        if drop_columns:
            df_ml = df_ml.drop(columns=drop_columns, errors='ignore')
        
        # Drop State column if present (unless it's the target)
        if 'State' in df_ml.columns and 'State' != target_column:
            df_ml = df_ml.drop(columns=['State'])
        
        # Handle missing values
        df_ml = self.handle_missing_values(df_ml)
        
        # Create features
        df_ml = self.create_features(df_ml)
        
        # Handle missing values again after feature creation
        df_ml = self.handle_missing_values(df_ml)
        
        if target_column and target_column in df_ml.columns:
            X = df_ml.drop(columns=[target_column])
            y = df_ml[target_column]
            return X, y
        
        return df_ml


if __name__ == '__main__':
    # Example usage
    from data_loader import IndiaDataLoader
    
    loader = IndiaDataLoader()
    preprocessor = DataPreprocessor()
    
    # Load and merge data
    agri_data = loader.load_agriculture_data()
    dev_data = loader.load_development_data()
    merged_data = loader.merge_data(agri_data, dev_data)
    
    # Preprocess
    processed_data = preprocessor.handle_missing_values(merged_data)
    processed_data = preprocessor.create_features(processed_data)
    
    print("Preprocessing completed!")
    print(f"Processed data shape: {processed_data.shape}")
    print("\nNew features created:")
    print(processed_data.columns.tolist())

