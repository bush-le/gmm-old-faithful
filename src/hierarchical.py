import numpy as np

def fit_predict_hierarchical(X, n_clusters=2, linkage='average'):
    """
    Manual implementation of Agglomerative Hierarchical Clustering.
    """
    n_samples = X.shape[0]
    
    # Initial clusters: each point is its own cluster
    clusters = {i: [i] for i in range(n_samples)}
    
    # Compute initial pairwise distances between all points
    # shape (N, N)
    distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    np.fill_diagonal(distances, np.inf)
    
    # Maintain distances between clusters
    # Pre-allocate a large matrix to avoid reallocation
    max_clusters = 2 * n_samples
    cluster_distances = np.full((max_clusters, max_clusters), np.inf)
    cluster_distances[:n_samples, :n_samples] = distances
    
    current_cluster_id = n_samples
    active_clusters = list(range(n_samples))
    
    while len(active_clusters) > n_clusters:
        # We can extract the submatrix of active clusters
        active_indices = np.array(active_clusters)
        active_dist_matrix = cluster_distances[np.ix_(active_indices, active_indices)]
        
        i, j = np.unravel_index(np.argmin(active_dist_matrix), active_dist_matrix.shape)
        c1, c2 = active_clusters[i], active_clusters[j]
        
        # Merge c1 and c2 into a new cluster
        new_cluster = clusters[c1] + clusters[c2]
        clusters[current_cluster_id] = new_cluster
        
        active_clusters.remove(c1)
        active_clusters.remove(c2)
        active_clusters.append(current_cluster_id)
        
        # Update distances between new cluster and all other active clusters
        for c_other in active_clusters[:-1]: # exclude the new cluster itself
            pts1 = X[clusters[current_cluster_id]]
            pts2 = X[clusters[c_other]]
            
            dists = np.linalg.norm(pts1[:, np.newaxis] - pts2, axis=2)
            
            if linkage == 'single':
                dist = np.min(dists)
            else: # average
                dist = np.mean(dists)
                
            cluster_distances[current_cluster_id, c_other] = dist
            cluster_distances[c_other, current_cluster_id] = dist
            
        current_cluster_id += 1
        
    # Generate labels
    labels = np.zeros(n_samples, dtype=int)
    for label, cluster_id in enumerate(active_clusters):
        for pt_idx in clusters[cluster_id]:
            labels[pt_idx] = label
            
    return labels
