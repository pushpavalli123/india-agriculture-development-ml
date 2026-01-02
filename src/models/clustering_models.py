"""
Clustering models for grouping similar states
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns


class StateClustering:
    """Clustering models for grouping states by similar characteristics"""
    
    def __init__(self):
        self.models = {
            'kmeans': None,
            'dbscan': None,
            'hierarchical': None
        }
        self.scaler = StandardScaler()
        self.cluster_labels = {}
        
    def find_optimal_clusters(self, X, max_clusters=10, method='kmeans'):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X: Feature matrix
            max_clusters: Maximum number of clusters to test
            method: Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            Dictionary with metrics for different cluster numbers
        """
        X_scaled = self.scaler.fit_transform(X)
        
        results = {
            'n_clusters': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': []
        }
        
        for n in range(2, max_clusters + 1):
            if method == 'kmeans':
                model = KMeans(n_clusters=n, random_state=42, n_init=10)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n)
            else:
                raise ValueError("Method must be 'kmeans' or 'hierarchical'")
            
            labels = model.fit_predict(X_scaled)
            
            results['n_clusters'].append(n)
            if method == 'kmeans':
                results['inertia'].append(model.inertia_)
            else:
                results['inertia'].append(None)
            results['silhouette'].append(silhouette_score(X_scaled, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))
        
        return pd.DataFrame(results)
    
    def cluster_kmeans(self, X, n_clusters=3, random_state=42):
        """
        Perform K-Means clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            Cluster labels
        """
        X_scaled = self.scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        self.models['kmeans'] = kmeans
        self.cluster_labels['kmeans'] = labels
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        print(f"K-Means Clustering (n_clusters={n_clusters})")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        
        return labels
    
    def cluster_dbscan(self, X, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering
        
        Args:
            X: Feature matrix
            eps: Maximum distance between samples in the same neighborhood
            min_samples: Minimum number of samples in a neighborhood
            
        Returns:
            Cluster labels
        """
        X_scaled = self.scaler.fit_transform(X)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        self.models['dbscan'] = dbscan
        self.cluster_labels['dbscan'] = labels
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"DBSCAN Clustering")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        
        if n_clusters > 0:
            silhouette = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
            print(f"  Silhouette Score: {silhouette:.4f}")
        
        return labels
    
    def cluster_hierarchical(self, X, n_clusters=3, linkage='ward'):
        """
        Perform Hierarchical clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion
            
        Returns:
            Cluster labels
        """
        X_scaled = self.scaler.fit_transform(X)
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hierarchical.fit_predict(X_scaled)
        
        self.models['hierarchical'] = hierarchical
        self.cluster_labels['hierarchical'] = labels
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        print(f"Hierarchical Clustering (n_clusters={n_clusters}, linkage={linkage})")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        
        return labels
    
    def analyze_clusters(self, df, labels, cluster_name='Cluster'):
        """
        Analyze cluster characteristics
        
        Args:
            df: Original DataFrame with state names
            labels: Cluster labels
            cluster_name: Name for the cluster column
            
        Returns:
            DataFrame with cluster analysis
        """
        df_clustered = df.copy()
        df_clustered[cluster_name] = labels
        
        # Summary statistics by cluster
        numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns
        cluster_summary = df_clustered.groupby(cluster_name)[numeric_cols].mean()
        
        return df_clustered, cluster_summary
    
    def visualize_clusters(self, X, labels, feature_names=None, save_path=None):
        """
        Visualize clusters (2D projection using PCA)
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Names of features
            save_path: Path to save the plot
        """
        from sklearn.decomposition import PCA
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6)
        plt.colorbar(scatter)
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('State Clustering Visualization (PCA)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


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
    
    # Prepare data for clustering
    X = preprocessor.prepare_for_ml(merged_data, target_column=None)
    
    # Perform clustering
    clusterer = StateClustering()
    
    # Find optimal clusters
    optimal = clusterer.find_optimal_clusters(X, max_clusters=8)
    print("\nOptimal Clusters Analysis:")
    print(optimal)
    
    # Perform K-Means clustering
    labels = clusterer.cluster_kmeans(X, n_clusters=3)
    
    # Analyze clusters
    clustered_df, summary = clusterer.analyze_clusters(merged_data, labels)
    print("\nCluster Summary:")
    print(summary)

