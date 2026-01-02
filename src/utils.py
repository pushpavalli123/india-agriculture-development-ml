"""
Utility functions for India Agriculture & Development Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Directories created successfully!")


def save_results(results, filename, output_dir='results'):
    """
    Save analysis results to JSON file
    
    Args:
        results: Dictionary or list to save
        filename: Output filename
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"Results saved to {filepath}")


def load_results(filename, input_dir='results'):
    """
    Load analysis results from JSON file
    
    Args:
        filename: Input filename
        input_dir: Input directory
        
    Returns:
        Loaded results
    """
    filepath = Path(input_dir) / filename
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def calculate_statistics(df, group_by=None):
    """
    Calculate comprehensive statistics
    
    Args:
        df: DataFrame
        group_by: Column to group by (optional)
        
    Returns:
        Statistics DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if group_by:
        stats = df.groupby(group_by)[numeric_cols].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ])
    else:
        stats = df[numeric_cols].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).T
    
    return stats


def get_indian_states():
    """Return list of Indian states and union territories"""
    return [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli',
        'Daman and Diu', 'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
    ]


if __name__ == '__main__':
    # Create project directories
    create_directories()
    print("\nProject structure initialized!")

