import numpy as np
import math
from src.pin import *
from sklearn.neighbors import KDTree


# Solve Procrustes problem
def best_orth_trans(sub1, sub2) :
    u, s, vh = np.linalg.svd(sub1 @ sub2.T)
    return (u @ vh)

# Compute maximum t such that the approximate cocycle associated to the approximate local
# trivialization given by [bases] is a [tolerance]-approximate cocycle
# (Euclidean data)
def approx_cocycle_vr_death(pointcloud, bases, tolerance = 0.5, max_dth = np.inf) :

    kd_tree = KDTree(pointcloud, leaf_size=2)

    death = max_dth
    
    n = len(bases)
    for i in range(n) :

        close_neighbors_i = kd_tree.query_radius(np.array([pointcloud[i]]), r = death)[0]
        for j in [j for j in close_neighbors_i if j > i] :
            ij_omega = best_orth_trans(bases[i], bases[j])

            close_neighbors_j = kd_tree.query_radius(np.array([pointcloud[j]]), r = death)[0]
            for k in [k for k in close_neighbors_j if k > j] :
                jk_omega = best_orth_trans(bases[j], bases[k])
                ik_omega = best_orth_trans(bases[i], bases[k])
    
                if np.linalg.norm(ij_omega @ jk_omega - ik_omega) >= tolerance :
                    death_candidate = max(np.linalg.norm(pointcloud[i] - pointcloud[j]),
                                          np.linalg.norm(pointcloud[j] - pointcloud[k]),
                                          np.linalg.norm(pointcloud[i] - pointcloud[k]))
                    if death_candidate < death :
                        death = death_candidate

    return death


# Compute first Stiefel--Whitney class of approximate cocycle
# (Euclidean data)
def approx_sw1_vr_from_local_bases(pointcloud, bases, max_eps) :
    cocycle = []
    n = len(pointcloud)

    kd_tree = KDTree(pointcloud, leaf_size=2)
    close_neighbors = kd_tree.query_radius(pointcloud, r = max_eps)
    # force j to be larger than i
    close_neighbors = [ np.array([j for j in close_neighbors[i]
                                  if j > i and np.linalg.norm(pointcloud[i] - pointcloud[j]) < max_eps])
                                  for i in range(n) ]
    close_neighbors = np.array(close_neighbors)

    for i in range(n) :
        for j in close_neighbors[i] :
            if np.linalg.det(best_orth_trans(bases[i],bases[j])) < 0 :
                cocycle.append([i,j,1])

    return np.array(cocycle)


# Compute Euler class of approximate cocycle with values in SO(2)
# (Euclidean data)
def approx_eu_vr_from_local_bases(pointcloud, bases, max_eps) :
    cocycle = []
    n = len(pointcloud)

    kd_tree = KDTree(pointcloud, leaf_size=2)
    close_neighbors = kd_tree.query_radius(pointcloud, r = max_eps)
    # force j to be larger than i
    close_neighbors = [ np.array([j for j in close_neighbors[i]
                                  if j > i and np.linalg.norm(pointcloud[i] - pointcloud[j]) < max_eps])
                                  for i in range(n) ]
    close_neighbors = np.array(close_neighbors)

    for i in range(n) :
        for j in close_neighbors[i] :
            ij_omega = best_orth_trans(bases[i], bases[j])
            ij_rad = math.atan2(ij_omega[1,0], ij_omega[0,0])
    
            for k in close_neighbors[j] :
                ik_close = np.linalg.norm(pointcloud[i] - pointcloud[k]) < max_eps 
                if ik_close :
                    jk_omega = best_orth_trans(bases[j], bases[k])
                    ik_omega = best_orth_trans(bases[i], bases[k])
    
                    ik_rad = math.atan2(ik_omega[1,0], ik_omega[0,0])
                    jk_rad = math.atan2(jk_omega[1,0], jk_omega[0,0])
    
                    simplex_approx_val = (ij_rad + jk_rad - ik_rad)/(2 * np.pi)
                        
                    simplex_val = int(np.rint(simplex_approx_val))
    
                    if simplex_val != 0 :
                        cocycle.append([i,j,k, simplex_val])
    
    cocycle = np.array(cocycle)

    return np.array(cocycle)


# Compute second Stiefel--Whitney class of approximate cocycle
# (Euclidean data)
def approx_sw2_vr_from_local_bases(pointcloud, bases, max_eps) :
    cocycle = []
    n = len(pointcloud)
    d = len(bases[0])
    #print("dimension is " + str(d))

    kd_tree = KDTree(pointcloud, leaf_size=2)
    close_neighbors = kd_tree.query_radius(pointcloud, r = max_eps)
    # force j to be larger than i
    close_neighbors = [ np.array([j for j in close_neighbors[i]
                                  if j > i and np.linalg.norm(pointcloud[i] - pointcloud[j]) < max_eps])
                                  for i in range(n) ]
    close_neighbors = np.array(close_neighbors)



    pin_lifts = dict([])

    for i in range(n) :
        for j in close_neighbors[i] :

            if (i,j) in pin_lifts:
                ij_pin = pin_lifts[(i,j)]
            else :
                ij_omega = best_orth_trans(bases[i], bases[j])
                ij_pin = lift_to_pin(d,ij_omega)
                pin_lifts[(i,j)] = ij_pin
    
            for k in close_neighbors[j] :
                ik_close = np.linalg.norm(pointcloud[i] - pointcloud[k]) < max_eps 
                if ik_close :

                    if (j,k) in pin_lifts:
                        jk_pin = pin_lifts[(j,k)]
                    else :
                        jk_omega = best_orth_trans(bases[j], bases[k])
                        jk_pin = lift_to_pin(d,jk_omega)
                        pin_lifts[(j,k)] = jk_pin

                    if (i,k) in pin_lifts:
                        ik_pin = pin_lifts[(i,k)]
                    else :
                        ik_omega = best_orth_trans(bases[i], bases[k])
                        ik_pin = lift_to_pin(d,ik_omega)
                        pin_lifts[(i,k)] = ik_pin

                    ki_pin = invert_pin(d, ik_pin)
    
                    simplex_approx_val = mults(d,vects_to_cliff(d,ij_pin + jk_pin + ki_pin))[0]
                    simplex_val = simplex_approx_val < 0
    
                    if simplex_val != 0 :
                        cocycle.append([i,j,k, simplex_val])
    
    cocycle = np.array(cocycle)

    return np.array(cocycle)


def ordered_neighbors_dist_mat(dist_mat) :
    return np.argsort(dist_mat, axis=1)

# Compute maximum t such that the approximate cocycle [cocycle] is a [tolerance]-approximate cocycle
# (for point cloud given by distance matrix)
def approx_cocycle_vr_death_(dist_mat, cocycle, tolerance = 0.5) :

    ordered_neighbors_index = ordered_neighbors_dist_mat(dist_mat)

    death = np.inf
    
    n = len(dist_mat)
    for i in range(n) :

        close_neighbors_i = ordered_neighbors_index[i]
        for j in close_neighbors_i :
            if j <= i :
                continue
            if dist_mat[i,j] > death :
                break
            ij_omega = cocycle[i,j]

            close_neighbors_j = ordered_neighbors_index[j]

            for k in close_neighbors_j :

                if k <= j :
                    continue
                if dist_mat[j,k] > death :
                    break

                jk_omega = cocycle[j,k]
                ik_omega = cocycle[i,k]
    
                if np.linalg.norm(ij_omega @ jk_omega - ik_omega) >= tolerance :
                    death_candidate = max(dist_mat[i,j], dist_mat[j,k], dist_mat[i,k])
                    if death_candidate < death :
                        death = death_candidate

    return death


# Compute second Stiefel--Whitney class of approximate cocycle
# (for point cloud given by distance matrix)
def approx_sw2_vr_from_local_bases_(dist_mat, rots, max_eps) :
    cocycle = []
    n = len(dist_mat)
    d = len(rots[0,0])

    ordered_neighbors_index = ordered_neighbors_dist_mat(dist_mat)
    close_neighbors = [ np.array([j for j in ordered_neighbors_index[i]
                                  if j > i and dist_mat[i,j] < max_eps])
                                  for i in range(n) ]
    close_neighbors = np.array(close_neighbors)



    pin_lifts = dict([])

    for i in range(n) :
        for j in close_neighbors[i] :

            if (i,j) in pin_lifts:
                ij_pin = pin_lifts[(i,j)]
            else :
                ij_omega = rots[i,j]
                ij_pin = lift_to_pin(d,ij_omega)
                pin_lifts[(i,j)] = ij_pin
    
            for k in close_neighbors[j] :
                ik_close = dist_mat[i,k] < max_eps 
                if ik_close :

                    if (j,k) in pin_lifts:
                        jk_pin = pin_lifts[(j,k)]
                    else :
                        jk_omega = rots[j,k]
                        jk_pin = lift_to_pin(d,jk_omega)
                        pin_lifts[(j,k)] = jk_pin

                    if (i,k) in pin_lifts:
                        ik_pin = pin_lifts[(i,k)]
                    else :
                        ik_omega = rots[i,k]
                        ik_pin = lift_to_pin(d,ik_omega)
                        pin_lifts[(i,k)] = ik_pin

                    ki_pin = invert_pin(d, ik_pin)
    
                    simplex_approx_val = mults(d,vects_to_cliff(d,ij_pin + jk_pin + ki_pin))[0]

                    simplex_val = simplex_approx_val < 0
    
                    if simplex_val != 0 :
                        cocycle.append([i,j,k, simplex_val])
    
    cocycle = np.array(cocycle)

    return np.array(cocycle)


# Compute Euler class of approximate cocycle with values in SO(2)
# (for point cloud given by distance matrix)
def approx_eu_vr_from_local_bases_(dist_mat, rots, max_eps) :
    cocycle = []
    n = len(dist_mat)
    d = len(rots[0,0])

    ordered_neighbors_index = ordered_neighbors_dist_mat(dist_mat)
    close_neighbors = [ np.array([j for j in ordered_neighbors_index[i]
                                  if j > i and dist_mat[i,j] < max_eps])
                                  for i in range(n) ]
    close_neighbors = np.array(close_neighbors)

    for i in range(n) :
        for j in close_neighbors[i] :
            ij_omega = rots[i,j]
            ij_rad = math.atan2(ij_omega[1,0], ij_omega[0,0])
    
            for k in close_neighbors[j] :
                ik_close = dist_mat[i,k] < max_eps 
                if ik_close :
                    jk_omega = rots[j,k]
                    ik_omega = rots[i,k]
    
                    ik_rad = math.atan2(ik_omega[1,0], ik_omega[0,0])
                    jk_rad = math.atan2(jk_omega[1,0], jk_omega[0,0])
    
                    simplex_approx_val = (ij_rad + jk_rad - ik_rad)/(2 * np.pi)
                        
                    simplex_val = int(np.rint(simplex_approx_val))
    
                    if simplex_val != 0 :
                        cocycle.append([i,j,k, simplex_val])
    
    cocycle = np.array(cocycle)

    return np.array(cocycle)