import numpy as np
from typing import Union, Set, Tuple, Dict, List
import pandas as pd
import networkx as nx
import time
from typing import Union, List, Optional, Dict, Tuple
from causallearn.utils.cit import CIT



class CI_test:
    """
    A lightweight wrapper for causal-learn CIT that only accepts pd.DataFrame.

    Supported methods:
    - "fisherz" / "mv_fisherz" / "kci" : continuous data
                
    - "gsq" / "chisq"                  : discrete data

    Usage:
        cit = CI_test(df, method="fisherz", alpha=0.01)
        CI, p = cit("X", "Y", ["Z1", "Z2"])
    """

    DISCRETE_METHODS = {"gsq", "chisq"}
    CONTINUOUS_METHODS = {"fisherz", "mv_fisherz", "kci"}
    ORACLE_METHODS = {"d_sep"}
    SUPPORTED_METHODS = DISCRETE_METHODS | CONTINUOUS_METHODS | ORACLE_METHODS

    def __init__(
        self,
        data: pd.DataFrame,
        method: str = "fisherz",
        alpha: float = 0.05,
        Max_time: Optional[float] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("CausalLearnCITest only accepts pd.DataFrame.")

        if method == "Fisher_Z":
            method = "fisherz"
        elif method == "D_sep":
            method = "d_sep"
        elif method == "G_sq":
            method = "gsq"

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods are: {sorted(self.SUPPORTED_METHODS)}"
            )

        if data.columns.duplicated().any():
            raise ValueError("DataFrame columns must be unique.")

        self.alpha = float(alpha)
        self.method = method
        self.Max_time = Max_time
        self.start_time = time.process_time()
        self.use_cache = bool(use_cache)
        self.max_cache_size = kwargs.get("max_cache_size", 10000)
        self._cache: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}
        self._ci_num = 0

        self.df = data.copy()
        self.columns = list(map(str, self.df.columns))
        self.df.columns = self.columns
        self.col_to_idx = {col: idx for idx, col in enumerate(self.columns)}

        self.data = self._prepare_data(self.df)

        self.latent_nodes = set(kwargs.get("latent_nodes") or [])
        self.selection_bias_nodes = set(kwargs.get("selection_bias_nodes") or [])
        self.for_bidden_nodes = self.latent_nodes | self.selection_bias_nodes



        if self.method in self.ORACLE_METHODS:
            self.G = nx.DiGraph(self.data)
            selection_bias_idx = set()
            if self.selection_bias_nodes:
                selection_bias_idx = set(self.col_to_idx[node] for node in self.selection_bias_nodes)
            if nx.__version__ > "3.4.2":
                self.cit_obj = lambda X, Y, S: 1 if nx.is_d_separator(self.G, X, Y, set(S)|selection_bias_idx) else 0
            else:
                self.cit_obj = lambda X, Y, S: 1 if nx.d_separated(self.G, {X}, {Y}, set(S)|selection_bias_idx) else 0
        else:
            self.cit_obj = CIT(self.data, self.method, **kwargs)



    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        if self.method in self.DISCRETE_METHODS:
            if df.isna().any().any():
                raise ValueError(f"Method '{self.method}' does not allow missing values.")

            return np.column_stack([
                pd.factorize(df[col], sort=True)[0].astype(np.int64, copy=False)
                for col in df.columns
            ])

        try:
            data = df.to_numpy(dtype=float, copy=True)
        except ValueError as exc:
            raise TypeError(
                f"Method '{self.method}' requires numeric continuous data."
            ) from exc

        if np.isinf(data).any():
            raise ValueError("Input data contains Inf.")

        if self.method != "mv_fisherz" and np.isnan(data).any():
            raise ValueError(
                f"Method '{self.method}' does not allow NaN. "
                f"Use 'mv_fisherz' if missing values are present."
            )

        return data

    def _to_idx(self, value: Union[int, str]) -> int:
        if isinstance(value, str):
            if value not in self.col_to_idx:
                raise ValueError(f"Unknown variable name: {value}")
            return self.col_to_idx[value]
        return int(value)


    def _format_query(
        self,
        X: Union[int, str],
        Y: Union[int, str],
        S: Optional[List[Union[int, str]]] = None,
    ) -> Tuple[int, int, List[int]]:
        assert X not in self.for_bidden_nodes, f"Variable '{X}' is a latent or selection bias node."
        assert Y not in self.for_bidden_nodes, f"Variable '{Y}' is a latent or selection bias node."
        if S is not None:
            assert set(S).isdisjoint(self.for_bidden_nodes), f"Conditioning set S contains latent or selection bias nodes: {set(S) & self.for_bidden_nodes}"

        S = [] if S is None else sorted({self._to_idx(s) for s in S})
        X = self._to_idx(X)
        Y = self._to_idx(Y)

        if X == Y:
            raise ValueError("X and Y must be different.")
        if X in S or Y in S:
            raise ValueError("Conditioning set S must not contain X or Y.")

        return X, Y, S

    def _cache_key(self, X: int, Y: int, S: List[int]) -> Tuple[int, int, Tuple[int, ...]]:
        a, b = sorted((X, Y))
        return a, b, tuple(S)

    def get_ci_num(self) -> int:
        return self._ci_num

    def __call__(
        self,
        X: Union[int, str],
        Y: Union[int, str],
        S: Optional[List[Union[int, str]]] = None,
        **kwargs,
    ) -> Tuple[bool, float]:
        if self.Max_time is not None and time.process_time() - self.start_time > self.Max_time:
            raise TimeoutError(
                f"The simulation exceeded maximum time limit of {self.Max_time} seconds."
            )

        X, Y, S = self._format_query(X, Y, S)
        key = self._cache_key(X, Y, S)

        if self.use_cache and key in self._cache:
            p_value = self._cache[key]
        else:
            p_value = float(self.cit_obj(X, Y, S))
            if self.use_cache:
                if len(self._cache) >= self.max_cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = p_value
            self._ci_num += 1

        CI = p_value > self.alpha
        dep = -np.log(max(p_value, 1e-300))

        if kwargs.get("output_dict", False):
            return {"CI": CI, "p_value": p_value, "dep": dep}
        return CI, p_value
    


    


