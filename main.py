"""
Main script to run India Agriculture & Development Analysis
"""

from src.utils import create_directories
from src.data_loader import IndiaDataLoader
from src.preprocessing import DataPreprocessor
from src.visualization import IndiaDataVisualizer
from src.models.regression_models import AgricultureDevelopmentRegressor
from src.models.classification_models import DevelopmentClassifier
from src.models.clustering_models import StateClustering


def main():
    """Main analysis pipeline"""
    print("=" * 60)
    print("India Agriculture & Development Analysis")
    print("=" * 60)
    
    # Create directories
    print("\n1. Setting up project structure...")
    create_directories()
    
    # Load data
    print("\n2. Loading data...")
    loader = IndiaDataLoader()
    agri_data = loader.load_agriculture_data()
    dev_data = loader.load_development_data()
    merged_data = loader.merge_data(agri_data, dev_data)
    
    print(f"   - Agriculture data: {agri_data.shape}")
    print(f"   - Development data: {dev_data.shape}")
    print(f"   - Merged data: {merged_data.shape}")
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.handle_missing_values(merged_data)
    processed_data = preprocessor.create_features(processed_data)
    
    print(f"   - Processed data: {processed_data.shape}")
    print(f"   - New features created: {len([c for c in processed_data.columns if c not in merged_data.columns])}")
    
    # Save processed data
    loader.save_processed_data(processed_data)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    viz = IndiaDataVisualizer()
    
    try:
        viz.plot_correlation_heatmap(processed_data, save_path='results/correlation_heatmap.png')
        print("   - Correlation heatmap saved")
    except Exception as e:
        print(f"   - Error creating correlation heatmap: {e}")
    
    try:
        viz.plot_agriculture_vs_development(processed_data, save_path='results/agri_dev_analysis.png')
        print("   - Agriculture vs Development analysis saved")
    except Exception as e:
        print(f"   - Error creating agriculture vs development plot: {e}")
    
    # Regression models
    print("\n5. Training regression models...")
    try:
        X, y = preprocessor.prepare_for_ml(processed_data, target_column='Agricultural_GDP_crores')
        regressor = AgricultureDevelopmentRegressor()
        results = regressor.train(X, y)
        best_name, _ = regressor.get_best_model()
        print(f"   - Best regression model: {best_name}")
        
        # Save model
        regressor.save_model(best_name, f'models/{best_name}_regressor.pkl')
    except Exception as e:
        print(f"   - Error training regression models: {e}")
    
    # Classification models
    print("\n6. Training classification models...")
    try:
        classifier = DevelopmentClassifier()
        categories = classifier.create_development_categories(processed_data, target_column='HDI')
        X_clf = preprocessor.prepare_for_ml(processed_data, target_column=None)
        # Align categories with X_clf indices
        y_clf = categories.loc[X_clf.index] if hasattr(categories, 'loc') else categories
        
        clf_results = classifier.train(X_clf, y_clf)
        best_clf_name, _ = classifier.get_best_model()
        print(f"   - Best classification model: {best_clf_name}")
        
        # Save model
        classifier.save_model(best_clf_name, f'models/{best_clf_name}_classifier.pkl')
    except Exception as e:
        print(f"   - Error training classification models: {e}")
    
    # Clustering
    print("\n7. Performing clustering analysis...")
    try:
        X_cluster = preprocessor.prepare_for_ml(processed_data, target_column=None)
        clusterer = StateClustering()
        
        # Find optimal clusters
        optimal = clusterer.find_optimal_clusters(X_cluster, max_clusters=8)
        print(f"   - Optimal clusters analysis completed")
        
        # Perform clustering
        labels = clusterer.cluster_kmeans(X_cluster, n_clusters=3)
        clustered_df, summary = clusterer.analyze_clusters(processed_data, labels)
        print(f"   - States clustered into {len(set(labels))} groups")
    except Exception as e:
        print(f"   - Error performing clustering: {e}")
    
    # Create dashboard
    print("\n8. Creating interactive dashboard...")
    try:
        viz.create_dashboard(processed_data, save_path='results/dashboard.html')
        print("   - Dashboard saved to results/dashboard.html")
    except Exception as e:
        print(f"   - Error creating dashboard: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the results/ directory for outputs.")
    print("=" * 60)


if __name__ == '__main__':
    main()

