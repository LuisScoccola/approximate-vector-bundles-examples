import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
from src.subsampling import *



def gaussian_potential(cov) :
    return (lambda x : scipy.stats.multivariate_normal.pdf(x, cov = cov * np.identity(3) ))

def spherical_potential(radius, center = np.array([0,0,0])) :
    return (lambda x : np.where(np.linalg.norm(x-center) <= radius, 1., 0.) )

def continuous_spherical_potential(radius, center = np.array([0,0,0])) :
    return (lambda x : np.where(np.linalg.norm(x-center) <= radius, 1-(np.linalg.norm(x-center)/radius), 0.) )



def crop_potential(potential, radius) :
    return (lambda x : np.where(np.linalg.norm(x) <= radius, potential(x), 0. ) )

def spherical_mixture(radii, centers) :
    return (lambda x : np.sum( [spherical_potential(r, c)(x) for r,c in zip(radii,centers)] ) )

def continuous_spherical_mixture(radii, centers) :
    return (lambda x : np.sum( [continuous_spherical_potential(r, c)(x) for r,c in zip(radii,centers)] ) )
 
class Molecule :

    def __init__(self, potential, support_radius) :
        self.potential = potential
        self.support_radius = support_radius

    def project(self, orthonormal_frame, im_size, integral_granularity) :

        # we evaluate potential on im_size x im_size x integral_granularity many vectors
        im_step = 2 * self.support_radius / im_size 
        granularity_step = 2 * self.support_radius / integral_granularity 
        step = np.array([im_step, im_step, granularity_step])

        normalize = lambda x : x * step - self.support_radius
        normalize_rot_pot = lambda x : self.potential(orthonormal_frame.dot(normalize(x)))

        # construct the indices at which we want to evaluate
        indices = np.indices((im_size,im_size,integral_granularity)).transpose(1,2,3,0)
        # turn indices to vectors and evaluate potential
        pots = np.apply_along_axis(normalize_rot_pot, 3, indices)
        # integrate
        image = np.sum(pots, axis=2)

        return Molecule_projection(image, direction = orthonormal_frame) 

class Molecule_projection :

    def __init__(self, image, direction = None) :
        self.image = image
        if direction is not None :
            self.direction = direction


def rot_(M1, M2) :
    u, s, vh = np.linalg.svd(M1[:,0:2].T @ M2[:,0:2])
    return (u @ vh)


# find rotation from elements of SO(3)
def rot(M1, M2) :

    v1 = M1[:,2]
    v2 = M2[:,2]

    if np.allclose(v1, v2) :
        R = np.identity(3)
    elif np.allclose(v1, -v2) :
        print("Opposite viewing angles!")
        return
    else :
        cross = np.cross(v1, v2)
        dot = np.dot(v1,v2)

        sin = np.linalg.norm(cross) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))

        #print(sin)
        #print(cos)

        k = cross / np.linalg.norm(cross)
        #print(k)

        K = np.array( [[0, - k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]] )
        #print(K)

        R = np.identity(3) + sin * K + (1 - cos) * np.matmul(K, K)

    #print(R)

    ret = np.matmul(np.matmul(M2.transpose(),R),M1)
    #print(ret)
    
    ret = ret[0:2,0:2]
    #print(ret)

    return ret

# compute distance between two images
def image_dif(img1, img2, smooth_param = 5) :

    t = 1 - np.abs(np.linspace(-1, 1, smooth_param))
    kernel = t.reshape(smooth_param, 1) * t.reshape(1, smooth_param)
    kernel /= kernel.sum()

    s1 = scipy.signal.convolve2d(img1, kernel, mode='same')
    s2 = scipy.signal.convolve2d(img2, kernel, mode='same')

    #plt.figure(figsize=(2,2))
    #plt.imshow(s1, cmap='gray')
    #plt.show()

    #plt.figure(figsize=(2,2))
    #plt.imshow(img1, cmap='gray')
    #plt.show()

    return np.linalg.norm(s1-s2)

# rotate a square image using a rotation matrix
def rotate_image(rot, image) :
    # only takes square images

    size = image.shape[0]
    s = size//2

    def at(i,j) :
        if i >= size or i < 0 or j >= size or j < 0 :
            return 0
        else :
            return image[i,j]

    ret = np.empty((size,size))

    for i in range(size) :
        for j in range(size) :
            a = np.floor(rot.dot(np.array([i-s, j-s])) + np.array([s,s])).astype(int)
            ret[i,j] = at(a[0],a[1])

    return ret



def distances(mats, prs, smooth) :
    how_many = len(mats)
    dists = np.zeros((how_many,how_many))

    for n1 in range(how_many) :
        for n2 in range(n1+1, how_many) :
            r = rot(mats[n1],mats[n2])

            dists[n1,n2] = image_dif(prs[n1].image, rotate_image(r, prs[n2].image), smooth_param = smooth)

    return(dists + dists.transpose())

def distances_rots(mats, prs, rots, smooth) :
    how_many = len(mats)
    dists = np.zeros((how_many,how_many))

    for n1 in range(how_many) :
        for n2 in range(n1+1, how_many) :
            r = rots[n1,n2]

            dists[n1,n2] = image_dif(prs[n1].image, rotate_image(r, prs[n2].image), smooth_param = smooth)

    return(dists + dists.transpose())





def projections(mats, M, image_size = 50, image_depth = 10 ) :

    prs = np.array([ M.project(mat, image_size, image_depth) for mat in mats ])

    return(mats,prs)


def random_projections(M, how_many = 4, tot = 40, image_size = 50, image_depth = 10, seed = 0) :
    np.random.seed(seed)

    mats = np.array([ special_ortho_group.rvs(3) for i in range(tot) ])

    mats = mats[getGreedyPerm(mats[:,:,2], how_many)['perm']]

    return projections(mats, M, image_size = image_size, image_depth = image_depth)
