from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import numpy as np


# construct local pointclouds of a Euclidean pointcloud
def local_pointclouds(k, pointcloud) :
    kd_tree = KDTree(pointcloud, leaf_size=2)

    def neighbors_k(idx, k) :
        return kd_tree.query(np.array([pointcloud[idx]]), k = k)

    def local_pointcloud(idx, k) :
        dist, ind = neighbors_k(idx, k)
        #print(ind)
        return pointcloud[ind[0]]

    return np.array([ local_pointcloud(i,k) for i in range(len(pointcloud)) ])


# apply PCA to local pointclouds of a Euclidean pointcloud
def local_pca(k, pointcloud, n_components = 2, variance_thresh = 0.75, max_components = 5) :

    components = []
    dimss = []
    recovered_variances = []

    lp = local_pointclouds(k, pointcloud)

    max_components = min(max_components, len(pointcloud[0]))
    
    for i in range(len(pointcloud)) :
        local = lp[i]
    
        pca = PCA(n_components=max_components)
        local_dimred = pca.fit_transform(local)
    
        vs = pca.explained_variance_ratio_
        count = 0
        for dims in range(len(vs)) :
            count += vs[dims]
            if count >= variance_thresh :
                dimss.append(dims)
                break

        recovered_variances.append(np.sum(vs[:n_components]))
        
        components.append(pca.components_[:n_components])
    
    dimss = np.array(dimss)
    recovered_variances = np.array(recovered_variances)
    
    #print("To recover " + str(variance_thresh) + " of the variance, need " + str(np.average(dimss)) + " dimensions on averge.")
    print("With " + str(n_components) + " components, recover " + str(np.average(recovered_variances)) + " of the variance on averge.")

    return(np.array(components))
    