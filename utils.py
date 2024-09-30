import torch
import numpy as np
import math
import hashlib
from omegaconf import DictConfig
import base64


def generate_isometric_input(seq_length=10, gamma=1):
    """
    Generate some data X_0 of shape (T, d) such that X_0 @ X_0^T = I
    where seq_length = gamma * d_in
    """
    assert gamma <= 1, "gamma must be smaller than 1"
    assert gamma > 0, "gamma must be positive"

    d_in = int(1 / gamma * seq_length)
    # Let us generate X_0 such that X_0 @ X_0^T =
    a = np.random.randn(d_in, d_in)
    # Compute the qr factorization
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    ph = np.diag(d / np.abs(d))
    X = (
        q @ ph @ q
    )  # to enforce uniqueness of QR decomposition and get uniformly sampled matrices
    X_0 = X[:seq_length, :d_in]

    return torch.tensor(X_0, dtype=torch.float32)


def generate_isometric_dataset(n_data, seq_length, gamma):
    data = torch.zeros((n_data, seq_length, int(1 / gamma * seq_length)))
    for i in range(n_data):
        data[i] = generate_isometric_input(seq_length=seq_length, gamma=gamma)
    return data


## chafai matrices
def generate_chafai(n):
    G_unnorm = math.sqrt(math.log(2)) * (torch.randn(n, n) - math.log(2) / 2)
    G_exp = torch.exp(G_unnorm)
    G = G_exp / (torch.sum(G_exp, axis=1)[:, None])
    return G


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))
    ax.legend()


def _aux_explore_params_recursive(parent_name, element, result):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig):
                _aux_explore_params_recursive(f"{parent_name}.{k}", v, result)
            else:
                result[f"{parent_name}.{k}"] = v
    else:
        result[parent_name] = element


def explore_params_recursive(params):
    result = {}
    for k, v in params.items():
        _aux_explore_params_recursive(k, v, result)
    return result


def params2experiment_name(params):
    def _make_hashable(d):
        out = tuple(sorted(d.items()))
        return out

    def _make_hash(t):
        # Use hashlib because the Python hash() function
        # will not provide hashes that are consistent across restarts
        # https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
        hasher = hashlib.sha256()
        hasher.update(repr(t).encode())
        hash_code = base64.b64encode(hasher.digest()).decode()
        # Replace '/' from hash code to be able to use it as folder name
        hash_code = hash_code.replace("/", "0")
        return hash_code

    result = explore_params_recursive(params)
    result_hashable = _make_hashable(result)
    experiment_name = _make_hash(result_hashable)
    return experiment_name
