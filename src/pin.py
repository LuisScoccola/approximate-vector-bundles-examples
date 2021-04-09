import numpy as np
from gmpy2 import popcount


# multiplication in Clifford algebra
def sign(d, n, m) :
    c = 0
    for i in range(d-1) :
        c += popcount(m<<(d-1-i) & n)
    return ((-1)**c)

def mult(d, x, y) :
    ret = np.zeros(2**d)
    for n in range(2**d) :
        for m in range(2**d) :
            ret[ n ^ m ] += sign(d, n, m) * x[n] * y[m]
    return ret

def mults(d, xs) :
    #print(type(xs))
    if len(xs) == 1 :
        return xs[0]
    else :
        return mult(d, xs[0], mults(d, xs[1:]))


# write a vector in canonical basis of Cliff
def vect_to_cliff(d, v) :
    ret = np.zeros(2**d)
    for i in range(d) :
        ret[1<<i] = v[i]
    return ret

def vects_to_cliff(d, L) :
    return list(map( lambda v : vect_to_cliff(d,v), L))

def invert_pin(d, L) :
    return L[::-1]


# Householder reflection of a vector
def refl_along(d, v) :
    return (np.identity(d) - 2 * np.matmul(np.array([v]).T, np.array([v]))/np.linalg.norm(v)**2 )


# lift from O(d) to Pin(d)
def lift_to_pin(d, M) :
    vs, coef = np.linalg.qr(M, mode='raw')
    vs = vs.T
    R = np.triu(vs)
    refls = (vs - R + np.identity(d)).T
    refls = [ref/np.linalg.norm(ref) for ref,c in zip(refls,coef) if not np.isclose(c,0)]
    missing = [ np.eye(1, d, i)[0] for i in range(d) if np.isclose(R[i,i], -1) ]
    refls = refls + missing

    return refls