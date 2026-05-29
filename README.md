# MMB-by-MMB-V2

MMB-by-MMB-V2 is the second version of the MMB-by-MMB algorithm for local causal structure learning and evaluation. It provides a complete pipeline for random DAG generation, latent-node assignment, DAG-to-PAG conversion, continuous or discrete data simulation, MMB-by-MMB inference, and local-structure evaluation. A complete end-to-end example is available in `example.ipynb`.

## 😀 Main Components

- `simulate_dag`: randomly generates a DAG in `generator/random_graph.py`.
- `set_latent_nodes`: assigns latent nodes in `generator/random_graph.py`.
- `dag2pag`: converts a DAG to a PAG in `dagtopag/dag2.py`.
- `linear_sem`: simulates continuous data from a linear SEM in `generator/Continuous_model.py`.
- `discrete_model`: simulates discrete data in `generator/Discrete_model.py`.
- `data_MMB_by_MMB` / `oracle_MMB_by_MMB`: main entry points for the data-based and oracle versions of MMB-by-MMB in `mmb_by_mmb.py`.
- `local_mark_evaluation`: evaluates learned local structures in `Utils/util_tools.py`.
- `generator/Example_Read_bif_and_to_topological_DAG.ipynb`: shows how to read a BIF file from `https://www.bnlearn.com/bnrepository/` and convert it into a topological DAG, using examples such as Alarm, Mildew, and Andes.

## 🗒️ Typical Workflow

A typical workflow is to generate or load a DAG, optionally assign latent nodes and convert the DAG to a PAG, then simulate continuous or discrete data, run MMB-by-MMB on a target node, and finally evaluate the learned local structure.

## 🛠️ Environment

Python version in the current environment:

- Python 3.11.15

Third-party packages used by the codebase and their versions in the current environment:

| Package | Version in current environment |
| --- | --- |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| networkx | 3.6.1 |
| scipy | 1.17.1 |
| causal-learn | 0.1.4.5 |
| igraph | 1.0.0 |
| pyAgrum | 2.3.2 |

Note: standard-library modules such as `logging`, `typing`, `itertools`, `time`, `collections`, `dataclasses`, `enum`, `warnings`, `os`, `sys`, and `random` are also used, but they are included with Python and do not require separate installation.

## 🚀 Evaluation

Let $\mathrm{PAG}$ and $\widehat{\mathrm{PAG}}$ denote the true PAG and the learned PAG. The function `local_mark_evaluation` evaluates the local structure around a target variable $T$ by comparing the target row and target column of the two PAG matrices, excluding the diagonal entry $(T,T)$. Formally, it considers

$$
\Omega_T
=
\{(T,V):V\in \mathbf{O}\setminus\{T\}\}
\cup
\{(V,T):V\in \mathbf{O}\setminus\{T\}\},
$$

where $\mathbf{O}$ is the set of observed variables. In other words, $\Omega_T$ contains all matrix entries corresponding to endpoint marks of edges incident to $T$.

It reports four target-specific metrics:

- `Mark-Precision`, `Mark-Recall`, and `Mark-F1`, which measure exact recovery of edge marks around the target.
- `Local-SHD`, which counts structural mismatches around the target.

The mark-level true positives are defined as

$$
\mathrm{TP}_{\mathrm{mark}}
=
\sum_{(i,j)\in\Omega_T}
\mathbb{I}
\left[
\widehat{\mathrm{PAG}}(i,j)\neq 0
\ \land\
\widehat{\mathrm{PAG}}(i,j)=\mathrm{PAG}(i,j)
\right].
$$

The metrics are then defined as

$$
\mathrm{Mark-Precision}
=
\frac{\mathrm{TP}_{\mathrm{mark}}}{\sum_{(i,j)\in\Omega_T}\mathbb{I}[\widehat{\mathrm{PAG}}(i,j)\neq 0]},
$$

$$
\mathrm{Mark-Recall}
=
\frac{\mathrm{TP}_{\mathrm{mark}}}{\sum_{(i,j)\in\Omega_T}\mathbb{I}[\mathrm{PAG}(i,j)\neq 0]},
$$

$$
\mathrm{Mark-F1}
=
\frac{2\cdot \mathrm{Mark-Precision}\cdot \mathrm{Mark-Recall}}{\mathrm{Mark-Precision}+\mathrm{Mark-Recall}},
$$

and

$$
\mathrm{Local-SHD}
=
\sum_{(i,j)\in\Omega_T}
\mathbb{I}\left[\widehat{\mathrm{PAG}}(i,j)\neq \mathrm{PAG}(i,j)\right].
$$

This evaluation is strict: a predicted mark is counted as correct only if it is nonzero and exactly matches the ground-truth PAG mark. Higher mark-level scores and smaller `Local-SHD` indicate more accurate recovery of the target's local causal structure.


## 🤝 Contributing & Contact

If you have questions or suggestions, feel free to open an issue or contact the author directly.

Email: [zhengli0060(at)gmail(dot)com]