import numpy as np
from scipy import constants as c
import functools
import itertools
from matplotlib import pyplot as plt

KINETIC_CONSTANT = c.hbar**2 / (2 * c.m_e * c.e)

def coefficients(m, states):
    n = (states**3) // 2
    s = m + n
    floor = states // 2

    h = s // states**2 - floor
    k = s % states**2 // states - floor
    l = s % states - floor

    return h, k, l

def kinetic(k, g):
    v = k + g
    return KINETIC_CONSTANT * v @ v

def potential(g, tau, sym, asym=0):
    return sym * np.cos(2 * np.pi * g @ tau) # + 1j * asym * np.sin(2 * np.pi * g @ tau)

def hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states):
    a = lattice_constant
    ff = form_factors
    basis = reciprocal_basis
    
    kinetic_c = (2 * np.pi / a)**2
    offset = 1 / 8 * np.ones(3)
    
    n = states**3
    
    @functools.lru_cache(maxsize=n)
    def cached_coefficients(m):
        return coefficients(m, states)
    
    h = np.empty(shape=(n, n))
    
    for row, col in itertools.product(range(n), repeat=2):
        if row == col:
            g = cached_coefficients(row - n // 2) @ basis
            h[row][col] = kinetic_c * kinetic(k, g)
        else:
            g = cached_coefficients(row - col) @ basis
            factors = ff.get(g @ g)
            h[row][col] = potential(g, offset, *factors) if factors else 0
    
    return h

def band_structure(lattice_constant, form_factors, reciprocal_basis, states, path):
    bands = []
    
    for k in np.vstack(path):
        h = hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states)
        eigvals = np.linalg.eigvals(h)
        eigvals.sort()
        # bands.append(eigvals[:8])
        bands.append(eigvals)
    
    return np.stack(bands, axis=-1)

def linpath(a, b, n=50, endpoint=True):
    spacings = [np.linspace(start, end, num=n, endpoint=endpoint) for start, end in zip(a, b)]
    return np.stack(spacings, axis=-1)

A = 5.43e-10

rytoev = lambda *i: np.array(i) * 13.6059

FORM_FACTORS = {
    3.0: rytoev(-0.21),
    8.0: rytoev(0.04),
   11.0: rytoev(0.08)
}

RECIPROCAL_BASIS = np.array([
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
])

n = 100

G = np.array([0, 0, 0])
L = np.array([1/2, 1/2, 1/2])
K = np.array([3/4, 3/4, 0])
X = np.array([0, 0, 1])
W = np.array([1, 1/2, 0])
U = np.array([1/4, 1/4, 1])

lambd = linpath(L, G, n, endpoint=False)
delta = linpath(G, X, n, endpoint=False)
x_uk = linpath(X, U, n // 4, endpoint=False)
sigma = linpath(K, G, n, endpoint=True)

bands = band_structure(A, FORM_FACTORS, RECIPROCAL_BASIS, states=10, path=[lambd, delta, x_uk, sigma])

bands -= max(bands[3])

plt.figure(figsize=(15, 9))

ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.xlim(0, len(bands))
# plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)
plt.ylim(min(bands[0]) - 1, max(bands[10]) + 1)

xticks = n * np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])
plt.xticks(xticks, ('$L$', '$\Lambda$', '$\Gamma$', '$\Delta$', '$X$', '$U,K$', '$\Sigma$', '$\Gamma$'), fontsize=18)
plt.yticks(fontsize=18)

for y in np.arange(-25, 25, 2.5):
    plt.axhline(y, ls='--', lw=0.3, color='black', alpha=0.3)

plt.tick_params(axis='both', which='both',
                top='off', bottom='off', left='off', right='off',
                labelbottom='on', labelleft='on', pad=5)

plt.xlabel('k-Path', fontsize=20)
plt.ylabel('E(k) (eV)', fontsize=20)

plt.text(135, -18, 'Fig. 1. Band structure of Si.', fontsize=12)

colors = 1 / 255 * np.array([
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207]
])

for band, color in zip(bands, colors):
    plt.plot(band, lw=2.0, color=color)

plt.show()
