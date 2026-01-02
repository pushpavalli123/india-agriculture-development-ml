# India Agriculture & Development Analysis - ML Project

A comprehensive Machine Learning project analyzing agriculture and development metrics across all Indian states.

## Project Overview

This project analyzes agricultural productivity, economic development, and their interrelationships across all 28 states and 8 union territories of India using various machine learning techniques.

## Features

- **Data Analysis**: Comprehensive analysis of agriculture and development indicators
- **ML Models**: Multiple machine learning models for prediction and classification
- **Visualizations**: Interactive and static visualizations of state-wise data
- **Predictions**: Crop yield predictions, development index forecasting
- **Clustering**: State clustering based on agricultural and development patterns

## Project Structure

```
.
├── data/                   # Data files (CSV, Excel)
│   ├── raw/               # Raw data files
│   └── processed/         # Processed/cleaned data
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── data_loader.py    # Data loading utilities
│   ├── preprocessing.py  # Data preprocessing
│   ├── models/           # ML model implementations
│   ├── visualization.py  # Visualization functions
│   └── utils.py          # Utility functions
├── models/                # Saved model files
├── results/               # Analysis results and outputs
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your data files in `data/raw/` directory
2. Run the main analysis notebook:
```bash
jupyter notebook notebooks/main_analysis.ipynb
```

Or use the Python scripts:
```bash
python src/data_loader.py
python src/preprocessing.py
```

## Data Requirements

The project expects data files with the following structure:
- State-wise agricultural data (crop production, area, yield)
- Development indicators (GDP, literacy, infrastructure, etc.)
- Time series data (if available)

## Models Included

1. **Regression Models**: For predicting crop yields and development indices
2. **Classification Models**: For categorizing states by development level
3. **Clustering Models**: For grouping similar states
4. **Time Series Models**: For forecasting trends

## Results

Analysis results, visualizations, and model outputs are saved in the `results/` directory.

## Contributing

Feel free to contribute by adding new models, improving visualizations, or enhancing the analysis.

## License

MIT License

