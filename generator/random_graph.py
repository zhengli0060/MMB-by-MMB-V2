import numpy as np
import igraph as ig
import random
import pandas as pd

def simulate_dag(
    num_nodes: int,
    expected_degree: int = None,
    p_edge: float = None,
    graph_type: str = 'ER'
) -> pd.DataFrame:
    """Simulate a random DAG.

    Args:
        num_nodes (int): Number of nodes.
        expected_degree (int, optional): Expected degree of each node. The value
            provided must be greater or equal than 1.
        p_edge (float, optional): Probability of edge between each pair of nodes.
        graph_type (str): ER, SF, BP.

    Returns:
        B (pd.DataFrame): [num_nodes, num_nodes] binary adj matrix of DAG.
        [i, j] == 1 means i -> j, and 0 means no edge.

    """
    if num_nodes < 1:
        raise ValueError('`num_nodes` must be greater or equal than 1.')
    if (expected_degree is None) == (p_edge is None):
        raise ValueError('Provide exactly one of `expected_degree` or `p_edge`.')
    if expected_degree is not None and expected_degree < 1:
        raise ValueError('`expected_degree` must be greater or equal than 1.')
    if p_edge is not None and not 0 <= p_edge <= 1:
        raise ValueError('`p_edge` must be between 0 and 1.')

    num_edges = (
        int(round(expected_degree * num_nodes * 0.5))
        if expected_degree is not None
        else int(round(p_edge * num_nodes * (num_nodes - 1) * 0.5))
    )
    max_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = min(num_edges, max_edges)

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        if p_edge is not None:
            G_und = ig.Graph.Erdos_Renyi(n=num_nodes, p=p_edge)
        else:
            G_und = ig.Graph.Erdos_Renyi(n=num_nodes, m=num_edges)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=num_nodes, m=int(round(num_edges / num_nodes)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * num_nodes)
        G = ig.Graph.Random_Bipartite(top, num_nodes - top, m=num_edges, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    # topological order
    topological_sorting = ig.Graph.Adjacency(B_perm.tolist()).topological_sorting()
    B_perm = B_perm[topological_sorting, :][:, topological_sorting]
    B_perm = pd.DataFrame(B_perm, index=[f'V{i+1}' for i in range(num_nodes)], columns=[f'V{i+1}' for i in range(num_nodes)])
    return B_perm



def set_latent_nodes(adj_matrix: pd.DataFrame, num_latent: int, debug: bool=False) -> tuple[pd.DataFrame, list[str]]:
    """
    Set the latent nodes in the graph.
    """
    
    # Define Cand_latent as nodes with more than two children and no parents
    Cand_latent = set(adj_matrix.columns[adj_matrix.sum(axis=1) >= 2]) - set(adj_matrix.columns[adj_matrix.sum(axis=0) >= 1])
    if debug:
        print(f"Init Candidates for latent nodes: {Cand_latent}")

    if num_latent is None:
        # Determine the number of latent nodes to set
        num_latent = max(1, int(len(Cand_latent) * latent_rate))
    if debug:
        print(f"Number of latent nodes to set: {num_latent}")
        print(f"Candidates for latent nodes: {Cand_latent}")

    latent_nodes = []
    if len(Cand_latent) >= num_latent:
        # Randomly select latent nodes from the candidates
            latent_nodes = random.sample(sorted(Cand_latent), num_latent)

    if debug:
        print(f"Selected latent nodes: {latent_nodes}")

    # Set the latent nodes in the graph
    for node in latent_nodes:
        # Rename the row and column labels from 'Vi' to 'Li'
        new_label = f"L{node[1:]}"  # Replace 'V' with 'L' while keeping the numeric part
        adj_matrix.rename(index={node: new_label}, columns={node: new_label}, inplace=True)

    latent_nodes = [f"L{node[1:]}" for node in latent_nodes]  # Update latent node names to match the new labels
    return adj_matrix, latent_nodes