# Useful Commands for India Agriculture & Development Analysis

## Basic Commands

### Run Complete Analysis
```bash
python main.py
```

### Install/Update Dependencies
```bash
pip install -r requirements.txt
```

### Check Python Version
```bash
python --version
```

## Jupyter Notebook Commands

### Start Jupyter Notebook
```bash
jupyter notebook
```

### Start Jupyter Lab
```bash
jupyter lab
```

### Open Specific Notebook
```bash
jupyter notebook notebooks/main_analysis.ipynb
```

### Convert Notebook to Python Script
```bash
jupyter nbconvert --to script notebooks/main_analysis.ipynb
```

## Data Management Commands

### View Processed Data
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/merged_data.csv'); print(df.head()); print(f'\nShape: {df.shape}')"
```

### Check Data Files
```bash
dir data\raw
dir data\processed
```

### View Sample Data Structure
```bash
python -c "from src.data_loader import IndiaDataLoader; loader = IndiaDataLoader(); agri = loader.load_agriculture_data(); print(agri.head())"
```

## Model Commands

### Load and Use Regression Model
```bash
python -c "import joblib; import pandas as pd; from src.preprocessing import DataPreprocessor; model = joblib.load('models/random_forest_regressor.pkl'); print('Model loaded successfully')"
```

### Test Model Prediction
```bash
python -c "from src.models.regression_models import AgricultureDevelopmentRegressor; from src.data_loader import IndiaDataLoader; from src.preprocessing import DataPreprocessor; loader = IndiaDataLoader(); preprocessor = DataPreprocessor(); agri = loader.load_agriculture_data(); dev = loader.load_development_data(); merged = loader.merge_data(agri, dev); processed = preprocessor.create_features(merged); X, y = preprocessor.prepare_for_ml(processed, 'Agricultural_GDP_crores'); regressor = AgricultureDevelopmentRegressor(); results = regressor.train(X, y); print('Model trained and tested')"
```

## Visualization Commands

### Open Dashboard in Browser
```bash
start results\dashboard.html
```

### View All Generated Images
```bash
dir results\*.png
```

### Generate Specific Visualization
```bash
python -c "from src.data_loader import IndiaDataLoader; from src.preprocessing import DataPreprocessor; from src.visualization import IndiaDataVisualizer; loader = IndiaDataLoader(); preprocessor = DataPreprocessor(); agri = loader.load_agriculture_data(); dev = loader.load_development_data(); merged = loader.merge_data(agri, dev); processed = preprocessor.create_features(merged); viz = IndiaDataVisualizer(); viz.plot_correlation_heatmap(processed, save_path='results/correlation_test.png'); print('Visualization created')"
```

## Analysis Commands

### Run Regression Analysis Only
```bash
python -c "from src.data_loader import IndiaDataLoader; from src.preprocessing import DataPreprocessor; from src.models.regression_models import AgricultureDevelopmentRegressor; loader = IndiaDataLoader(); preprocessor = DataPreprocessor(); agri = loader.load_agriculture_data(); dev = loader.load_development_data(); merged = loader.merge_data(agri, dev); processed = preprocessor.create_features(merged); X, y = preprocessor.prepare_for_ml(processed, 'Agricultural_GDP_crores'); regressor = AgricultureDevelopmentRegressor(); results = regressor.train(X, y); best_name, _ = regressor.get_best_model(); print(f'Best model: {best_name}')"
```

### Run Classification Analysis Only
```bash
python -c "from src.data_loader import IndiaDataLoader; from src.preprocessing import DataPreprocessor; from src.models.classification_models import DevelopmentClassifier; loader = IndiaDataLoader(); preprocessor = DataPreprocessor(); agri = loader.load_agriculture_data(); dev = loader.load_development_data(); merged = loader.merge_data(agri, dev); processed = preprocessor.create_features(merged); classifier = DevelopmentClassifier(); categories = classifier.create_development_categories(processed, 'HDI'); X_clf = preprocessor.prepare_for_ml(processed, target_column=None); y_clf = categories.loc[X_clf.index] if hasattr(categories, 'loc') else categories; results = classifier.train(X_clf, y_clf); best_name, _ = classifier.get_best_model(); print(f'Best model: {best_name}')"
```

### Run Clustering Analysis Only
```bash
python -c "from src.data_loader import IndiaDataLoader; from src.preprocessing import DataPreprocessor; from src.models.clustering_models import StateClustering; loader = IndiaDataLoader(); preprocessor = DataPreprocessor(); agri = loader.load_agriculture_data(); dev = loader.load_development_data(); merged = loader.merge_data(agri, dev); processed = preprocessor.create_features(merged); X_cluster = preprocessor.prepare_for_ml(processed, target_column=None); clusterer = StateClustering(); optimal = clusterer.find_optimal_clusters(X_cluster, max_clusters=8); print(optimal); labels = clusterer.cluster_kmeans(X_cluster, n_clusters=3); print(f'States clustered into {len(set(labels))} groups')"
```

## File Management Commands

### List All Project Files
```bash
dir /s /b *.py
```

### Count Lines of Code
```bash
findstr /R /N ".*" src\*.py | find /C ":"
```

### Check File Sizes
```bash
dir results /s
dir models /s
```

## Testing Commands

### Test Data Loader
```bash
python -c "from src.data_loader import IndiaDataLoader; loader = IndiaDataLoader(); agri = loader.load_agriculture_data(); dev = loader.load_development_data(); print(f'Agriculture: {agri.shape}, Development: {dev.shape}')"
```

### Test Preprocessing
```bash
python -c "from src.data_loader import IndiaDataLoader; from src.preprocessing import DataPreprocessor; loader = IndiaDataLoader(); preprocessor = DataPreprocessor(); agri = loader.load_agriculture_data(); dev = loader.load_development_data(); merged = loader.merge_data(agri, dev); processed = preprocessor.create_features(merged); print(f'Original: {merged.shape}, Processed: {processed.shape}')"
```

### Test Visualization
```bash
python -c "from src.visualization import IndiaDataVisualizer; viz = IndiaDataVisualizer(); print('Visualizer initialized successfully')"
```

## Quick Analysis Commands

### Get Data Summary
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/merged_data.csv'); print(df.describe())"
```

### Get Top States by Metric
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/merged_data.csv'); print(df.nlargest(5, 'Agricultural_GDP_crores')[['State', 'Agricultural_GDP_crores']])"
```

### Get Correlation Matrix
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/merged_data.csv'); numeric = df.select_dtypes(include=['float64', 'int64']); print(numeric.corr()['Agricultural_GDP_crores'].sort_values(ascending=False))"
```

## Cleanup Commands

### Remove Generated Files (Keep Data)
```bash
del results\*.png
del results\*.html
del models\*.pkl
```

### Remove All Generated Files
```bash
del /Q results\*
del /Q models\*
del /Q data\processed\*
```

## Helpful Python One-Liners

### Check Installed Packages
```bash
pip list | findstr "pandas numpy matplotlib scikit-learn"
```

### Update All Packages
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn plotly scipy xgboost lightgbm joblib openpyxl jupyter
```

### Check Project Structure
```bash
tree /F /A
```

## Windows-Specific Commands

### Open Results Folder
```bash
explorer results
```

### Open Models Folder
```bash
explorer models
```

### Open Data Folder
```bash
explorer data
```

### Open Dashboard
```bash
start results\dashboard.html
```

## Advanced Commands

### Run with Custom Parameters
```python
# Create a custom script: custom_run.py
from src.data_loader import IndiaDataLoader
from src.preprocessing import DataPreprocessor
from src.models.regression_models import AgricultureDevelopmentRegressor

loader = IndiaDataLoader()
preprocessor = DataPreprocessor()
agri = loader.load_agriculture_data()
dev = loader.load_development_data()
merged = loader.merge_data(agri, dev)
processed = preprocessor.create_features(merged)

# Custom target
X, y = preprocessor.prepare_for_ml(processed, target_column='HDI')
regressor = AgricultureDevelopmentRegressor()
results = regressor.train(X, y, test_size=0.3)  # Custom test size
```

### Export Results to Excel
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/merged_data.csv'); df.to_excel('results/merged_data.xlsx', index=False); print('Exported to Excel')"
```

### Create Summary Report
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/merged_data.csv'); summary = df.describe(); summary.to_csv('results/summary_statistics.csv'); print('Summary saved')"
```

