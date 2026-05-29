""""
The original code comes from https://github.com/xunzheng/notears/blob/master/notears/utils.py.
"""

import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import pandas as pd


def linear_sem(dag: pd.DataFrame, effect_ranges=((-1.5, -0.5), (0.5, 1.5)), sample_size=1000, **kwargs) -> pd.DataFrame:
    """Simulate samples from linear SEM."""

    def simulate_parameter(B, w_ranges):
        """Simulate SCM parameters for a DAG."""
        W = np.zeros(B.shape)
        S = np.random.randint(len(w_ranges), size=B.shape)
        for i, (low, high) in enumerate(w_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            W += B * (S == i) * U
        return W

    def _simulate_single_equation(Pa_X, w, scale, scm_type, n):
        """
        Pa_X: np.array, rows = samples, cols = parents of x
        w: np.array, weight vector
        scale: float, scale of noise
        n: int, num of samples
        """
        if scm_type == 'gaussian':
            z = np.random.normal(scale=scale, size=n)
            x = Pa_X @ w + z
        elif scm_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = Pa_X @ w + z
        elif scm_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = Pa_X @ w + z
        elif scm_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = Pa_X @ w + z
        elif scm_type == 'logistic':
            x = np.random.binomial(1, sigmoid(Pa_X @ w)) * 1.0
        elif scm_type == 'poisson':
            x = np.random.poisson(np.exp(Pa_X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    n = sample_size
    adj_matrix = dag.to_numpy().astype(int)
    G = ig.Graph.Adjacency(adj_matrix.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == adj_matrix.shape[0]

    Weight_matrix = simulate_parameter(adj_matrix, w_ranges=effect_ranges)
    data_matrix = np.zeros([n, adj_matrix.shape[0]])
    for i in ordered_vertices:
        parents = G.neighbors(i, mode=ig.IN)
        data_matrix[:, i] = _simulate_single_equation(
            data_matrix[:, parents],
            Weight_matrix[parents, i],
            kwargs.get('noise_scale', 1.0),
            kwargs.get('scm_type', 'gaussian'),
            n
        )
    return pd.DataFrame(data_matrix, columns=dag.columns)


