import numpy as np
from tqdm import tqdm


def loss(S, A, V, U, W, lambda_coefficient):
    mask_S = S
    mask_A = A
    network_loss = np.linalg.norm(np.multiply(mask_S, S - V @ V.T), "fro") ** 2
    attributes_loss = np.linalg.norm(np.multiply(mask_A, A @ W - V @ U.T), "fro") ** 2
    total_loss = network_loss + lambda_coefficient * attributes_loss
    return total_loss


def normalize_W(W):
    return W / np.trace(W)


def normalize_S(S):
    return S / S.sum()


def normalize_A(A):
    return A / A.sum()


def initial_V(n, k):
    np.random.seed(0)
    _V = np.random.standard_normal((n, k))
    _V[_V <= 0] = 0
    return _V


def initial_U(m, k):
    np.random.seed(0)
    _U = np.random.standard_normal((m, k))
    _U[_U <= 0] = 0
    return _U


def initial_W(m):
    np.random.seed(0)
    _W = np.diag(np.random.standard_normal(m))
    _W[_W <= 0] = 0
    return normalize_W(_W)


def initial(m, n, k):
    return initial_V(n, k), initial_U(m, k), initial_W(m)


def update_V(S, A, V, U, W, lambda_coefficient):
    numerator = S @ V + S.T @ V + lambda_coefficient * A @ W @ U
    denominator = (
        2 * V @ V.T @ V + lambda_coefficient * V @ U.T @ U + np.finfo(float).eps
    )
    return np.multiply(V, numerator / denominator)


def update_U(S, A, V, U, W, lambda_coefficient):
    numerator = W @ A.T @ V
    denominator = U @ V.T @ V + np.finfo(float).eps
    return np.multiply(U, numerator / denominator)


def update_W(S, A, V, U, W, lambda_coefficient):
    numerator = A.T @ V @ U.T
    denominator = A.T @ A @ W + np.finfo(float).eps
    return normalize_W(np.multiply(W, numerator / denominator))


def update(S, A, V, U, W, lambda_coefficient):
    _V = update_V(S, A, V, U, W, lambda_coefficient)
    _U = update_U(S, A, V, U, W, lambda_coefficient)
    _W = update_W(S, A, V, U, W, lambda_coefficient)
    if _V[_V < 0] or _U[_U < 0] or _W[_W < 0]:
        print("Error in negative value of factorized matrices")
    return _V, _U, _W


def train(S, A, m, n, k, lambda_coefficient=0.1, max_iterations=1000, epsilon=1e-6):
    assert S.shape == (n, n)
    assert A.shape == (n, m)
    # S = normalize_S(S)
    # A = normalize_A(A)
    V, U, W = initial(m, n, k)
    losses = []
    best_index = 0
    best_loss = np.inf
    best_parameters = None
    with tqdm(range(max_iterations)) as pbar:
        _loss = loss(S, A, V, U, W, lambda_coefficient)
        pbar.set_description(f"Loss: {_loss:.8f}")
        for i in pbar:
            V, U, W = update(S, A, V, U, W, lambda_coefficient)
            _loss = loss(S, A, V, U, W, lambda_coefficient)
            losses.append(_loss)
            pbar.set_description(f"Loss: {_loss:.8f}")
            pbar.update(1)
            if _loss < best_loss and i > 100:
                best_index = i
                best_loss = _loss
                best_parameters = V, U, W
    V, U, W = best_parameters
    return V, U, W, losses, best_index
