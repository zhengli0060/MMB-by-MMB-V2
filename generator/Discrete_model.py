import numpy as np
import pandas as pd
import pyagrum as gum


def discrete_model(dag: pd.DataFrame, num_categories=2, sample_size=1000) -> pd.DataFrame:
    """Simulate samples from a discrete model defined by a DAG.
        Args:
            dag (pd.DataFrame): [d, d] binary adjacency matrix of a DAG.
                                dag.loc[parent, child] == 1 means parent -> child.
            num_categories (int): number of categories for each variable.
            sample_size (int): number of samples to generate.
        Returns:
            pd.DataFrame: A DataFrame containing the generated data, where columns correspond to node names
    """

    bn = random_CPT(dag, num_categories=num_categories)
    data = generate_data(bn, samples=sample_size)

    return data


def random_CPT(dag: pd.DataFrame, num_categories: int = 2) -> dict:
    """Generate random Conditional Probability Tables (CPTs) for each node in the DAG.

    Args:
        dag (pd.DataFrame): [d, d] binary adjacency matrix of a DAG.
                            dag.loc[parent, child] == 1 means parent -> child.
        num_categories (int): number of categories for each variable.

    Returns:
        gum.BayesNet: A Bayesian network with randomly generated CPTs for each node.
    """


    if num_categories < 2:
        raise ValueError("num_categories must be at least 2")

    nodes = list(dag.columns)

    # Build a Bayesian network from the adjacency matrix
    bn = gum.BayesNet("random_bn")
    name_to_id = {}
    for node in nodes:
        var = gum.LabelizedVariable(node, node, num_categories)
        name_to_id[node] = bn.add(var)

    for parent in dag.index:
        for child in dag.columns:
            if dag.loc[parent, child] == 1:
                bn.addArc(name_to_id[parent], name_to_id[child])

    bn.generateCPTs()

    return bn

def generate_data(bn: gum.BayesNet, samples: int) -> pd.DataFrame:
    """Generate synthetic data from a given Bayesian network.

    Args:
        bn (gum.BayesNet): A Bayesian network from which to generate data.
        samples (int): The number of samples to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the generated data, where columns correspond to node names.
    """
    # Generate samples using pyAgrum's sampling method
    generator = gum.BNDatabaseGenerator(bn)
    generator.drawSamples(samples)
    data = generator.to_pandas()

    converted_columns = {}
    for column in data.columns:
        numeric_series = pd.to_numeric(data[column], errors='coerce')
        if numeric_series.notna().all() and np.allclose(numeric_series, numeric_series.astype(int)):
            converted_columns[column] = numeric_series.astype(int)
        else:
            converted_columns[column] = pd.Series(
                pd.Categorical(data[column]).codes.astype(int),
                index=data.index
            )

    data = pd.DataFrame(converted_columns, index=data.index)
    # assert that no nan values are generated
    if data.isnull().values.any():
        raise ValueError("Generated data contains NaN values.")
    return data