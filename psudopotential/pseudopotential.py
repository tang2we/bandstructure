import sys
import numpy as np
from scipy import constants,linalg
from numba import jit, njit, prange, set_num_threads
import time
from matplotlib import pyplot as plt

set_num_threads(8)
np.set_printoptions(threshold=sys.maxsize)

def potential_matrix(H,K,L,coefficient,reciprocal_basis):
    n=(2*H+1)*(2*K+1)*(2*L+1)
    pseudomatrix=pseudomatrix_kernal(H,K,L,n)
    pseudomatrix=pseudomatrix.transpose((1,0,2))-pseudomatrix
    result_matrix = result_matrix_kernal(pseudomatrix,H,K,L,n,reciprocal_basis)
    return result_matrix*coefficient


def potential_coefficient(a,charges,parameter):
    return -parameter*charges*constants.elementary_charge**2/(a**2*constants.pi*constants.epsilon_0)

@njit(parallel=True)
def pseudomatrix_kernal(H,K,L,n):
    pseudomatrix = np.zeros((n,n,3))
    for h in prange(-H,H+1):
        for k in prange(-K,K+1):
            for l in prange(-L,L+1):
                for j in prange(n):
                    pseudomatrix[(h+H)*((2*K+1)*(2*L+1))+(k+K)*(2*L+1)+(l+L),j]=np.array([h,k,l])
    return pseudomatrix

@njit(parallel=True)
def result_matrix_kernal(pseudomatrix,H,K,L,n,reciprocal_basis):
    result_matrix = np.zeros((n,n))
    for h in prange(-H,H+1):
        for k in prange(-K,K+1):
            for l in prange(-L,L+1):
                index = (h+H)*((2*K+1)*(2*L+1))+(k+K)*(2*L+1)+(l+L)
                for j in prange(n):
                    if j!=index:
                        result_matrix[index,j]=1/np.sum((pseudomatrix[index][j]*reciprocal_basis)**2)
    return result_matrix

def halmitonian(potential_matrix,k_vertor,H,K,L,reciprocal_basis):
    n=(2*H+1)*(2*K+1)*(2*L+1)
    halmitonian_matrix = halmitonian_kernal(k_vertor,H,K,L,reciprocal_basis,n)
    return halmitonian_matrix+potential_matrix


@njit(parallel=True)
def halmitonian_kernal(k_vector,H,K,L,reciprocal_basis,n):
    halmitonian_matrix = np.zeros((n,n))
    for h in prange(-H,H+1):
        for k in prange(-K,K+1):
            for l in prange(-L,L+1):
                index = (h+H)*((2*K+1)*(2*L+1))+(k+K)*(2*L+1)+(l+L)
                halmitonian_matrix[index,index]=np.sum((k_vector-(h*reciprocal_basis[0]+k*reciprocal_basis[1]+l*reciprocal_basis[2]))**2)
                
    return halmitonian_matrix

def states(N):
    num = np.floor(np.power(N,1.0/3))
    H = num//2
    K = num//2
    L = np.floor(N/((2*H+1)*(2*K+1))//2)
    return int(H),int(K),int(L)


def path_initialize(k_start,k_end,N=50):
    N = int(N)
    k_1 = np.linspace(k_start[0],k_end[0],endpoint=True,num=N)
    k_2 = np.linspace(k_start[1],k_end[1],endpoint=True,num=N)
    k_3 = np.linspace(k_start[2],k_end[2],endpoint=True,num=N)
    return np.array([k_1,k_2,k_3])


def band_structure(H,K,L,radius,charges,parameter,reciprocal_basis,Path):
    start = time.time()
    potentialmatrix=potential_matrix(H,K,L,potential_coefficient(radius,charges,parameter),reciprocal_basis)
    E_0 = constants.hbar**2*(2*constants.pi)**2/(2*radius**2*constants.electron_mass)
    potentialmatrix/=E_0
    bands = []
    for path in Path:
        for i in np.arange(len(path[0])):
            Ham = halmitonian(potentialmatrix,np.array([path[0][i],path[1][i],path[2][i]]),H,K,L,reciprocal_basis)
            eigen_value= linalg.eigvalsh(Ham)
            bands.append(np.sort(eigen_value.real)*E_0)
            
    end = time.time()
    print(f"total times{end-start}")
    return np.stack(bands, axis=-1)

RECIPROCAL_BASIS = np.array([
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
])

# sample points per k-path
n = 100

# symmetry points in the Brillouin zone
G = np.array([0, 0, 0])
L = np.array([1/2, 1/2, 1/2])
K = np.array([3/4, 3/4, 0])
X = np.array([0, 0, 1])
W = np.array([1, 1/2, 0])
U = np.array([1/4, 1/4, 1])

# k-paths
lambd = path_initialize(L, G, n)
delta = path_initialize(G, X, n)
x_uk = path_initialize(X, U, n / 4)
sigma = path_initialize(K, G, n)

H,K,L=10,10,10

SIMPLE_CUBIC = np.array([[1,0,0],[0,1,0],[0,0,1]])

GG = np.array([0,0,0])
XX = np.array([0,0.5,0])
MM = np.array([0.5,0.5,0])
RR = np.array([0.5,0.5,0,5])

GX = path_initialize(GG,XX,100)
XM = path_initialize(XX,MM,100)
MR = path_initialize(MM,RR,100)
RG = path_initialize(RR,GG,100)

bands_simple = band_structure(H,K,L,1e-10,1,0,SIMPLE_CUBIC,[GX,XM,MR,RG])

for i in bands_simple[:16]:
    plt.plot(i)