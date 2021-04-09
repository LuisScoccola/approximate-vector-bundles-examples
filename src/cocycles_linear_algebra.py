import numpy as np
from sklearn.neighbors import KDTree
from src.approx_cocycles import *


# Construct linear system to solve for 1-cocycle [cocycle] in terms of persistent cohomology generators [coh_gen]
# (Euclidean data)
def matrix_from_vertices_gen_cocycle(pointcloud, coh_gen, deaths, cocycle, co_death) :

    N = len(pointcloud)


    kd_tree = KDTree(pointcloud, leaf_size=2)
    close_neighbors = kd_tree.query_radius(pointcloud, r = co_death)
    close_neighbors = [ np.array([j for j in close_neighbors[i]
                                  if j != i and np.linalg.norm(pointcloud[i] - pointcloud[j]) < co_death])
                                  for i in range(N) ]
    close_neighbors = np.array(close_neighbors)


    # start by having as generators only the coboundaries of vertices
    gens = []

    for i in range(N) :
        gen = []
        for j in close_neighbors[i] :
            gen.append([i,j,-1])
        if len(gen) > 0 :
            gens.append(gen)

    true_coh_gen = []
    for g, d in zip(coh_gen, deaths) :
        g_ = []
        for e in g :
            i, j, v = e
            if np.linalg.norm(pointcloud[i] - pointcloud[j]) < min(co_death, d)  :
                g_.append(e)
        true_coh_gen.append(g_)

    gens = list(true_coh_gen) + list(gens) + [cocycle]


    ### rows should be indexed by pairs of ordered and distinct points within distance co_death 
    rows = [(i,j) for i in range(N) for j in close_neighbors[i] if i < j]
    edge_to_row = dict([])
    for n,p in enumerate(rows) :
        edge_to_row[p] = n


    M = np.zeros((len(rows), len(gens)), dtype=int)

    for col, g in enumerate(gens) :
        for e in g :
            i,j,v = e
            if i < j :
                M[edge_to_row[(i,j)],col] = v
            else :
                M[edge_to_row[(j,i)],col] = -v

    return M


# Construct linear system to solve for 2-cocycle [cocycle] in terms of persistent cohomology generators [coh_gen]
# (Euclidean data)
def matrix_from_edges_gen_cocycle(pointcloud, coh_gen, deaths, cocycle, co_death) :
    # assumes the cocycle doesn't have edges after co_death

    N = len(pointcloud)

    kd_tree = KDTree(pointcloud, leaf_size=2)
    close_neighbors = kd_tree.query_radius(pointcloud, r = co_death)
    close_neighbors = [ np.array([j for j in close_neighbors[i]
                                  if j != i and np.linalg.norm(pointcloud[i] - pointcloud[j]) < co_death])
                                  for i in range(N) ]
    close_neighbors = np.array(close_neighbors)

    gens = []

    for i in range(N) :
        for j in [j for j in close_neighbors[i] if j > i] :
            gen = []
            for k in close_neighbors[j] :
                if i != k and np.linalg.norm(pointcloud[i] - pointcloud[k]) < co_death:
                    # should this be 1?
                    gen.append([i,j,k,1])
            if len(gen) > 0 :
                gens.append(gen)

    true_coh_gen = []
    for g, d in zip(coh_gen, deaths) :
        g_ = []
        for e in g :
            i, j, k, v = e
            if np.linalg.norm(pointcloud[i] - pointcloud[j]) < min(co_death, d) and \
               np.linalg.norm(pointcloud[j] - pointcloud[k]) < min(co_death, d) and \
               np.linalg.norm(pointcloud[i] - pointcloud[k]) < min(co_death, d) :
                g_.append(e)
        true_coh_gen.append(g_)

    gens = list(true_coh_gen) + list(gens) + [cocycle]

    ### rows should be indexed by triples of ordered and distinct points within distance co_death 
    rows = [(i,j,k) for i in range(N) for j in close_neighbors[i] for k in close_neighbors[j] if i < j and j < k
                                                                  and np.linalg.norm(pointcloud[i] - pointcloud[k]) < co_death ]
    triangle_to_row = dict([])
    for n,p in enumerate(rows) :
        triangle_to_row[p] = n

    M = np.zeros((len(rows), len(gens)), dtype=int)

    for col, g in enumerate(gens) :
        for e in g :
            i,j,k,v = e
            i_, j_, k_ = sorted([i,j,k])
            M[triangle_to_row[(i_, j_, k_)], col] = ((-1) ** sign_perm(i,j,k)) * v

    return M


def sign_perm(i,j,k) :
    if i < j :
        if i < k :
            return int(not j < k)
        else :
            return 0
    else :
        if j < k :
            return int(i < k)
        else :
            return 1


# Construct linear system to solve for 2-cocycle [cocycle] in terms of persistent cohomology generators [coh_gen]
# (for point cloud given by distance matrix)
def matrix_from_edges_gen_cocycle_(dist_mat, coh_gen, deaths, cocycle, co_death) :

    N = len(dist_mat)

    ordered_neighbors_index = ordered_neighbors_dist_mat(dist_mat)

    close_neighbors = [ np.array([j for j in ordered_neighbors_index[i]
                                  if j != i and dist_mat[i,j] < co_death])
                                  for i in range(N) ]
    close_neighbors = np.array(close_neighbors)

    gens = []

    for i in range(N) :
        for j in [j for j in close_neighbors[i] if j > i] :
            gen = []
            for k in close_neighbors[j] :
                if i != k and dist_mat[i,k] < co_death:
                    # should this be 1?
                    gen.append([i,j,k,1])
            if len(gen) > 0 :
                gens.append(gen)

    true_coh_gen = []
    for g, d in zip(coh_gen, deaths) :
        g_ = []
        for e in g :
            i, j, k, v = e
            if dist_mat[i,j] < min(co_death, d) and \
               dist_mat[j,k] < min(co_death, d) and \
               dist_mat[i,k] < min(co_death, d) :
                g_.append(e)
        true_coh_gen.append(g_)

    gens = list(true_coh_gen) + list(gens) + [cocycle]

    ### rows should be indexed by triples of ordered and distinct points within distance co_death 
    rows = [(i,j,k) for i in range(N) for j in close_neighbors[i] for k in close_neighbors[j] if i < j and j < k and dist_mat[i,k] < co_death]
    triangle_to_row = dict([])
    for n,p in enumerate(rows) :
        triangle_to_row[p] = n

    M = np.zeros((len(rows), len(gens)), dtype=int)

    for col, g in enumerate(gens) :
        for e in g :
            i,j,k,v = e
            i_, j_, k_ = sorted([i,j,k])
            M[triangle_to_row[(i_, j_, k_)], col] = ((-1) ** sign_perm(i,j,k)) * v

    return M
