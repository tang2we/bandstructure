import bandstructure as bs
import numpy as np

# parameters Es, Ep, Vss, Vsp, Vxx, Vxy
# Ep - Es = 7.20
args = (-4.03, 3.17, -8.13, 5.88, 3.17, 7.51)

# k-points per path
n = 1000

# lattice constant
a = 1

# nearest neighbor distance
d = a / 4 * np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
])

# sysmmetry points in the BZ
G = 2 * np.pi / a * np.array([0, 0, 0])
K = 2 * np.pi / a * np.array([3/4, 3/4, 0])
L = 2 * np.pi / a * np.array([1/2, 1/2, 1/2])
U = 2 * np.pi / a * np.array([1/4, 1/4, 1])
W = 2 * np.pi / a * np.array([1, 1/2, 0])
X = 2 * np.pi / a * np.array([0, 0, 1])

# tretrahedral
t = bs.tetrahedral.Tetrahedral()

# k-paths
lambd = t.compute_path(L, G, n, endpoint=False)
delta = t.compute_path(G, X, n, endpoint=False)
x_uk = t.compute_path(X, U, n, endpoint=False)
sigma = t.compute_path(K, G, n, endpoint=True)
path = np.concatenate((lambd, delta, x_uk, sigma), axis=0)

# band structure
bands = t.band_structure(d, path, *args)
print(np.shape(bands))
# plot
t.plot_band_structure(bands, path)



