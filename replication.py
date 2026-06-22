import numpy as np

def phi(h):
    return np.tanh(h) # activation function

def make_layer(n_pre, n_out, sw, sb): # Wx+b
    W = np.random.randn(n_out, n_pre) * sw / np.sqrt(n_pre)
    b = np.random.randn(n_out, 1) * sb
    return W, b

def forward_layer (x, W, b):
    h = W @ x+b
    x_next = phi(h)
    return h, x_next

def compute_q (h):
    return np.mean(h**2)

def forward_network (x0, depth, width, sw, sb):
    x = np.asarray(x0)
    if x.ndim == 1:
        x = x[:, None]
    
    hs = []
    qs = []
    for _ in range (depth):
        W, b = make_layer(x.shape[0], width, sw, sb)
        h, x = forward_layer(x, W, b)
        hs.append(h)
        qs.append(compute_q(h))
    
    return hs, qs

from numpy.polynomial.hermite_e import hermegauss
_X, _W = hermegauss(80)

def gaussian_average (values):
    return _W @ values / np.sqrt(2 * np.pi)

def length_integrand (z,q):
    h = np.sqrt(q) * z
    return phi(h)**2

def length_map (q, sw, sb):
    values = length_integrand(_X, q)
    average = gaussian_average(values)
    return sw**2*average + sb**2

def theory_qs(q0, depth, sw, sb):
    qs = []
    q = sw ** 2 * q0 + sb ** 2
    qs.append(q)
    for _ in range(1, depth):
        q = length_map(q, sw, sb)
        qs.append(q)

    return qs


