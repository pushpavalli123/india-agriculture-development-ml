"""
Data loading utilities for India Agriculture & Development Analysis
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


class IndiaDataLoader:
    """Class to load and manage India agriculture and development data"""
    
    def __init__(self, data_dir='data/raw'):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_agriculture_data(self, filename='agriculture_data.csv'):
        """
        Load agriculture data
        
        Args:
            filename: Name of the agriculture data file
            
        Returns:
            DataFrame with agriculture data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Creating sample structure.")
            return self._create_sample_agriculture_data()
        
        return pd.read_csv(filepath)
    
    def load_development_data(self, filename='development_data.csv'):
        """
        Load development indicators data
        
        Args:
            filename: Name of the development data file
            
        Returns:
            DataFrame with development data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Creating sample structure.")
            return self._create_sample_development_data()
        
        return pd.read_csv(filepath)
    
    def _create_sample_agriculture_data(self):
        """Create sample agriculture data structure"""
        # Indian states and union territories
        states = [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
            'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
            'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
            'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
            'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
            'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli',
            'Daman and Diu', 'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
        ]
        
        # Generate sample data
        np.random.seed(42)
        data = {
            'State': states,
            'Total_Crop_Area_ha': np.random.randint(100000, 20000000, len(states)),
            'Rice_Production_tons': np.random.randint(100000, 15000000, len(states)),
            'Wheat_Production_tons': np.random.randint(50000, 10000000, len(states)),
            'Cotton_Production_tons': np.random.randint(10000, 5000000, len(states)),
            'Sugarcane_Production_tons': np.random.randint(50000, 8000000, len(states)),
            'Total_Irrigation_Area_ha': np.random.randint(50000, 10000000, len(states)),
            'Fertilizer_Usage_tons': np.random.randint(10000, 2000000, len(states)),
            'Agricultural_GDP_crores': np.random.randint(5000, 500000, len(states)),
            'Rural_Population': np.random.randint(100000, 50000000, len(states)),
            'Farmers_Count': np.random.randint(50000, 10000000, len(states))
        }
        
        df = pd.DataFrame(data)
        return df
    
    def _create_sample_development_data(self):
        """Create sample development data structure"""
        states = [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
            'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
            'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
            'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
            'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
            'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli',
            'Daman and Diu', 'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
        ]
        
        np.random.seed(42)
        data = {
            'State': states,
            'GDP_Per_Capita': np.random.randint(50000, 300000, len(states)),
            'Literacy_Rate': np.random.uniform(60, 95, len(states)),
            'HDI': np.random.uniform(0.5, 0.8, len(states)),
            'Infrastructure_Index': np.random.uniform(0.4, 0.9, len(states)),
            'Education_Index': np.random.uniform(0.5, 0.9, len(states)),
            'Health_Index': np.random.uniform(0.5, 0.85, len(states)),
            'Urbanization_Rate': np.random.uniform(10, 50, len(states)),
            'Industrial_GDP_crores': np.random.randint(10000, 1000000, len(states)),
            'Service_GDP_crores': np.random.randint(20000, 2000000, len(states)),
            'Total_GDP_crores': np.random.randint(50000, 5000000, len(states))
        }
        
        df = pd.DataFrame(data)
        return df
    
    def merge_data(self, agriculture_df, development_df):
        """
        Merge agriculture and development data
        
        Args:
            agriculture_df: Agriculture DataFrame
            development_df: Development DataFrame
            
        Returns:
            Merged DataFrame
        """
        merged = pd.merge(
            agriculture_df,
            development_df,
            on='State',
            how='outer'
        )
        return merged
    
    def save_processed_data(self, df, filename='merged_data.csv', output_dir='data/processed'):
        """
        Save processed data to file
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")


if __name__ == '__main__':
    # Example usage
    loader = IndiaDataLoader()
    
    # Load data
    agri_data = loader.load_agriculture_data()
    dev_data = loader.load_development_data()
    
    # Merge data
    merged_data = loader.merge_data(agri_data, dev_data)
    
    # Save processed data
    loader.save_processed_data(merged_data)
    
    print("\nData loaded successfully!")
    print(f"Agriculture data shape: {agri_data.shape}")
    print(f"Development data shape: {dev_data.shape}")
    print(f"Merged data shape: {merged_data.shape}")
    print("\nFirst few rows:")
    print(merged_data.head())

