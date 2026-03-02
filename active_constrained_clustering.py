import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

def get_ambiguous_points(X, cluster_centers, n_queries):
    """
    Identifies 'ambiguous' data points lying on cluster boundaries.
    Calculates the margin between the distance to the closest and
    second-closest centroids.
    """
    dist_matrix = distance.cdist(X, cluster_centers, 'euclidean')
    sorted_dist = np.sort(dist_matrix, axis=1)
    # Margin = distance to 2nd closest - distance to closest
    margin_scores = sorted_dist[:, 1] - sorted_dist[:, 0]
    # Smallest margins represent the highest ambiguity
    ambiguous_indices = np.argsort(margin_scores)[:n_queries]
    return ambiguous_indices

def simulate_oracle(X, y, query_indices):
    """
    Simulates a domain expert providing Must-Link (ML) and
    Cannot-Link (CL) constraints for the actively queried points.
    """
    ml_constraints = set()
    cl_constraints = set()
    for i in range(len(query_indices)):
        for j in range(i + 1, len(query_indices)):
            idx1, idx2 = query_indices[i], query_indices[j]
            if y[idx1] == y[idx2]:
                ml_constraints.add((idx1, idx2))
            else:
                cl_constraints.add((idx1, idx2))
                
    return list(ml_constraints), list(cl_constraints)

def constrained_kmeans_fit_predict(X, n_clusters, ml_constraints, cl_constraints, max_iter=100):
    """
    A simplified COP-KMeans algorithm that updates centroids iteratively
    while strictly enforcing Cannot-Link constraints during assignment.
    """
    n_samples, n_features = X.shape
    # Initialize centroids randomly
    np.random.seed(42)
    initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[initial_indices]
    labels = np.full(n_samples, -1)
    for iteration in range(max_iter):
        dist_matrix = distance.cdist(X, centroids, 'euclidean')
        new_labels = np.full(n_samples, -1)
        # Greedy assignment checking CL constraints
        for i in range(n_samples):
            sorted_centers = np.argsort(dist_matrix[i])
            for c in sorted_centers:
                violation = False
                for cl in cl_constraints:
                    # If assigning 'i' to cluster 'c' violates a CL constraint
                    if (i == cl[0] and new_labels[cl[1]] == c) or (i == cl[1] and new_labels[cl[0]] == c):
                        violation = True
                        break
                if not violation:
                    new_labels[i] = c
                    break
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        # Update centroids
        for c in range(n_clusters):
            cluster_points = X[labels == c]
            if len(cluster_points) > 0:
                centroids[c] = np.mean(cluster_points, axis=0)
    return labels

def evaluate_dataset(X, y, dataset_name, n_queries=30):
    """
    Runs the full pipeline on a given dataset: scaling, standard K-Means,
    active constraint generation, and constrained K-Means, followed by
    comprehensive metric calculation and visualization.
    """
    n_clusters = len(np.unique(y))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"<===================================================>")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Clusters: {n_clusters}")
    # Baseline: Standard K-Means
    kmeans_base = KMeans(n_clusters=n_clusters, random_state=42)
    labels_base = kmeans_base.fit_predict(X_scaled)
    centers_init = kmeans_base.cluster_centers_
    # Proposed: Active Constrained Clustering
    # 1. Active Selection
    ambiguous_idx = get_ambiguous_points(X_scaled, centers_init, n_queries)
    # 2. Oracle Constraint Generation
    ml, cl = simulate_oracle(X_scaled, y, ambiguous_idx)
    # 3. Constrained Assignment
    labels_constrained = constrained_kmeans_fit_predict(X_scaled, n_clusters, ml, cl)
    # Metric Calculation
    metrics = {
        'ARI_Base': adjusted_rand_score(y, labels_base),
        'ARI_Constrained': adjusted_rand_score(y, labels_constrained),
        'NMI_Base': normalized_mutual_info_score(y, labels_base),
        'NMI_Constrained': normalized_mutual_info_score(y, labels_constrained),
        'Sil_Base': silhouette_score(X_scaled, labels_base),
        'Sil_Constrained': silhouette_score(X_scaled, labels_constrained)
    }
    # Print Statistics
    print("\n--- Clustering Statistics ---")
    print(f"Queries made: {n_queries} data points")
    print(f"Constraints generated: {len(ml)} Must-Link, {len(cl)} Cannot-Link")
    print(f"\n[Adjusted Rand Index (ARI)] - Higher is better (matches ground truth)")
    print(f"  Standard K-Means : {metrics['ARI_Base']:.2%}")
    print(f"  Active Constrained : {metrics['ARI_Constrained']:.2%}")
    print(f"  Improvement      : {((metrics['ARI_Constrained'] - metrics['ARI_Base']) / abs(metrics['ARI_Base'])) * 100:.2f}%")
    print(f"\n[Normalized Mutual Info (NMI)] - Higher is better")
    print(f"  Standard K-Means : {metrics['NMI_Base']:.2%}")
    print(f"  Active Constrained : {metrics['NMI_Constrained']:.2%}")
    # PCA Visualization (2D Projection)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(18, 5))
    plt.suptitle(f"Clustering Results Analysis: {dataset_name}", fontsize=16)
    # Plot 1: Ground Truth
    plt.subplot(1, 3, 1)
    scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=30)
    plt.title("Ground Truth (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # Plot 2: Standard K-Means
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_base, cmap='tab10', alpha=0.6, s=30)
    plt.title(f"Standard K-Means\nARI: {metrics['ARI_Base']:.2f} | NMI: {metrics['NMI_Base']:.2f}")
    plt.xlabel("Principal Component 1")
    # Plot 3: Active Constrained Clustering
    plt.subplot(1, 3, 3)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_constrained, cmap='tab10', alpha=0.6, s=30)
    # Highlight Queried Points
    plt.scatter(X_pca[ambiguous_idx, 0], X_pca[ambiguous_idx, 1],
                edgecolors='red', facecolors='none', s=120, linewidth=1.5,
                label='Actively Queried Points')
    plt.title(f"Active Constrained (Proposed)\nARI: {metrics['ARI_Constrained']:.2f} | NMI: {metrics['NMI_Constrained']:.2f}")
    plt.xlabel("Principal Component 1")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test Case 1: Breast Cancer Wisconsin Diagnostic (High Dimensional Medical Data)
    cancer_data = load_breast_cancer()
    evaluate_dataset(cancer_data.data, cancer_data.target, "Breast Cancer Wisconsin (30 Features)", n_queries=25)
    # Test Case 2: Optical Recognition of Handwritten Digits (Complex Multi-Class Data)
    digits_data = load_digits()
    evaluate_dataset(digits_data.data, digits_data.target, "Handwritten Digits (64 Features)", n_queries=50)