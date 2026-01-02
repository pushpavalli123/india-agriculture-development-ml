# Quick Start Guide

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the main script
```bash
python main.py
```

This will:
- Load sample data (or your data if placed in `data/raw/`)
- Preprocess the data
- Train ML models
- Generate visualizations
- Create an interactive dashboard

### Option 2: Use Jupyter Notebook
```bash
jupyter notebook notebooks/main_analysis.ipynb
```

### Option 3: Use individual modules

```python
from src.data_loader import IndiaDataLoader
from src.preprocessing import DataPreprocessor
from src.models.regression_models import AgricultureDevelopmentRegressor

# Load data
loader = IndiaDataLoader()
agri_data = loader.load_agriculture_data()
dev_data = loader.load_development_data()
merged_data = loader.merge_data(agri_data, dev_data)

# Preprocess
preprocessor = DataPreprocessor()
processed_data = preprocessor.handle_missing_values(merged_data)
processed_data = preprocessor.create_features(processed_data)

# Train models
X, y = preprocessor.prepare_for_ml(processed_data, target_column='Agricultural_GDP_crores')
regressor = AgricultureDevelopmentRegressor()
results = regressor.train(X, y)
```

## Adding Your Own Data

1. Place your agriculture data CSV file in `data/raw/agriculture_data.csv`
2. Place your development data CSV file in `data/raw/development_data.csv`

### Expected Data Format

**Agriculture Data (`agriculture_data.csv`):**
- `State`: State name
- `Total_Crop_Area_ha`: Total crop area in hectares
- `Rice_Production_tons`: Rice production in tons
- `Wheat_Production_tons`: Wheat production in tons
- `Cotton_Production_tons`: Cotton production in tons
- `Sugarcane_Production_tons`: Sugarcane production in tons
- `Total_Irrigation_Area_ha`: Irrigation area in hectares
- `Fertilizer_Usage_tons`: Fertilizer usage in tons
- `Agricultural_GDP_crores`: Agricultural GDP in crores
- `Rural_Population`: Rural population count
- `Farmers_Count`: Number of farmers

**Development Data (`development_data.csv`):**
- `State`: State name
- `GDP_Per_Capita`: GDP per capita
- `Literacy_Rate`: Literacy rate percentage
- `HDI`: Human Development Index
- `Infrastructure_Index`: Infrastructure index
- `Education_Index`: Education index
- `Health_Index`: Health index
- `Urbanization_Rate`: Urbanization rate percentage
- `Industrial_GDP_crores`: Industrial GDP in crores
- `Service_GDP_crores`: Service sector GDP in crores
- `Total_GDP_crores`: Total GDP in crores

## Output Files

After running the analysis, you'll find:

- `data/processed/merged_data.csv`: Processed and merged dataset
- `results/correlation_heatmap.png`: Correlation analysis
- `results/agri_dev_analysis.png`: Agriculture vs Development analysis
- `results/top_agri_states.png`: Top states by agricultural metrics
- `results/top_hdi_states.png`: Top states by HDI
- `results/dashboard.html`: Interactive dashboard
- `models/*.pkl`: Trained ML models

## Features

### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Classification Models
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine

### Clustering Models
- K-Means Clustering
- DBSCAN Clustering
- Hierarchical Clustering

## Troubleshooting

1. **Import errors**: Make sure you're running from the project root directory
2. **Missing data**: The project will generate sample data if your data files are missing
3. **Visualization errors**: Ensure matplotlib backend is properly configured

