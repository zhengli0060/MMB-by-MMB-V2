import sys
import os


import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Sequence
from Utils.CI_test import CI_test
import time




class MarkovBlanketLearner(ABC):
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], ci_test: CI_test = None, alpha: float = 0.05, **kwargs):
        """
        Initialize the Markov Blanket Learner.

        :param data: Input dataset (pd.DataFrame or np.ndarray).
        :param ci_test: Instance of CI_test for conditional independence testing.
        :param alpha: Significance level for CI tests.
        """
        self._validate_data(data)
        self.data = data
        if ci_test is None:
            ci_test_type = kwargs.get("ci_test_type")
            if ci_test_type is None:
                raise ValueError("ci_test_type must be provided if ci_test(a CI_test) is None.")
            self.ci_test = CI_test(data, method=ci_test_type, alpha=alpha, **kwargs)
            self.flag_ci_test = True  ## This flag indicates that ci_test is new created from the data
        else:
            if not isinstance(ci_test, CI_test):
                raise TypeError("ci_test must be an instance of CI_test.")
            self.ci_test = ci_test
            self.flag_ci_test = False  ## This flag indicates that ci_test is passed in by the user
        
        self.num_nodes = self.data.shape[1]
        self.mb_set: Dict[Union[str, int], list[Union[str, int]]] = {}
        

        self.latent_variables = kwargs.get("latent_nodes", None) ## for oracle test the Markov blanket learning in the presence of latent variables
        self.selection_bias_nodes = kwargs.get("selection_bias_nodes", None)  ## for oracle test the Markov blanket learning in the presence of selection bias nodes

        if self.latent_variables is not None or self.selection_bias_nodes is not None:
            if str(self.ci_test.method).lower() != "d_sep":
                raise ValueError("Only D_sep method supports latent variables or selection bias.")
            
            if self.latent_variables is not None and not isinstance(self.latent_variables, list):
                raise TypeError("latent_variables must be a list of column labels or indices.")
            if self.selection_bias_nodes is not None and not isinstance(self.selection_bias_nodes, list):
                raise TypeError("selection_bias_nodes must be a list of column labels or indices.")
            
            if self.latent_variables is not None and not all(var in self.column_names for var in self.latent_variables):
                raise ValueError("Some latent variables are not found in data columns.")
            if self.selection_bias_nodes is not None and not all(var in self.column_names for var in self.selection_bias_nodes):
                raise ValueError("Some selection bias nodes are not found in data columns.")
            if self.latent_variables is not None and self.selection_bias_nodes is not None:
                assert not (set(self.latent_variables) & set(self.selection_bias_nodes)), "latent_variables and selection_bias_nodes have overlapping elements"
            if self.latent_variables is not None:
                self.observed_variables = list(set(self.column_names) - set(self.latent_variables))
            if self.selection_bias_nodes is not None:
                self.observed_variables = list(set(self.observed_variables) - set(self.selection_bias_nodes))
        else:
            self.observed_variables = list(self.column_names)

        self.bool_mb_df = pd.DataFrame(
            0, index=self.observed_variables, columns=self.observed_variables
        )  ## 0 means not tested, 1 means is MB, -1 means not MB

    def _validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Validate and format the input data.

        :param data: Input dataset (pd.DataFrame or np.ndarray).
        :return: Formatted data as a NumPy array.
        """
        if len(data.shape) != 2:
            raise ValueError("data must be a 2D array or DataFrame.")

        if isinstance(data, pd.DataFrame):
            self.column_names = list(data.columns)

        elif isinstance(data, np.ndarray):
            self.column_names = list(range(data.shape[1]))

        else:
            raise TypeError("Data must be a pandas DataFrame or a NumPy array.")

    def _format_target(self, target: Union[int, str]) -> Union[int, str]:
        """
        Format the target node to its column index.

        :param target: Target node (column label or index).
        :return: Target label if input data is a DataFrame, Target index if input data is a NumPy array.
        """
        if isinstance(target, str):
            if target not in self.column_names:
                raise ValueError(f"Target '{target}' not found in data columns.")
            return target
        elif isinstance(target, int):
            if target < 0 or target >= self.num_nodes:
                raise ValueError(f"Target index {target} is out of bounds.")
            return target
        else:
            raise TypeError("Target must be a string (column label) or an integer (column index).")

    @abstractmethod
    def __call__(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Abstract method to get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        raise NotImplementedError

    @abstractmethod
    def get_markov_blanket(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Abstract method to get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        raise NotImplementedError
    
    def get_ci_num(self) -> int:
        """
        Get the number of conditional independence tests performed.

        :return: Number of CI tests.
        """
        if self.flag_ci_test:
            return self.ci_test.get_ci_num()  ## if there is a new CI_test created
        else:
            return 0  ## if the ci_test is passed in by the user

    def update_bool_mb_df(self, target: Union[int, str], mb: List[Union[int, str]]) -> None:
        """
        Update the bool_mb_df DataFrame with the Markov blanket information.

        :param target: Target node (column label or index).
        :param mb: List of nodes in the Markov blanket.
        """
        for Y in self.observed_variables:
            if Y == target:
                continue
            if Y in mb:
                self.bool_mb_df.loc[target, Y] = 1
                self.bool_mb_df.loc[Y, target] = 1
            else:
                self.bool_mb_df.loc[target, Y] = -1
                self.bool_mb_df.loc[Y, target] = -1
    
    def get_whitelisted(self, target: Union[int, str]) -> Sequence[Union[int, str]]:
        """
        Get the whitelisted variables for the target node.
        NOTE: whitelisted variables can be replaced with background knowledge, such as some variables that are explicitly known to be target's Markov blanket, or variables that are adjacent to the target in the causal structure.

        :param target: Target node (column label or index).
        :return: List of whitelisted variables.
        """
        whitelisted = [col for col in self.bool_mb_df.columns if self.bool_mb_df.loc[target, col] == 1]
        return whitelisted

    def max_mb_size(self) -> int:
        """
        This method returns the maximum size of the Markov blanket across all target nodes in self.mb_set.
        return: Maximum size of the Markov blanket.
        """
        
        # If all number in self.bool_mb_df is not 0
        if (self.bool_mb_df.values == 0).any():
            return self.bool_mb_df.shape[0]
        
        max_size = 0
        for target, mb in self.mb_set.items():
            if len(mb) > max_size:
                max_size = len(mb)
        
        return max_size

class TC_learn(MarkovBlanketLearner):
    """
    References:
        - Pellet J P, Elisseeff A. Using markov blankets for causal structure learning[J]. Journal of Machine Learning Research, 2008, 9(7).
    """
    def __init__(self, data, ci_test = None, alpha = 0.05, **kwargs):
        super().__init__(data, ci_test, alpha, **kwargs)

    def __call__(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        mb = self.get_markov_blanket(target, **kwargs)
        return mb
    
    def get_markov_blanket(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        target = self._format_target(target)
        if target in self.mb_set:
            return self.mb_set[target]
        
        """ Get the Markov blanket of the target node using the CI test. """
        
        Candidate = self.observed_variables.copy()
        if target in Candidate:
            Candidate.remove(target)
        else:
            raise ValueError(f"Target {target} is not in the observed variables{self.observed_variables}.")
        
        # whitelisted includes the has been determined MB variables.
        whitelisted = self.get_whitelisted(target)
        mb = []
        mb.extend(whitelisted)  # Add whitelisted variables to MB. Note: ['X1'].extend(['X1', 'X2']) = ['X1', 'X1', 'X2']
        for Y in Candidate:
            if Y in mb or self.bool_mb_df.loc[target, Y] == -1:  ## if the CI test has been done before
                continue

            S = [var for var in Candidate if var != Y]
            if not self.ci_test(target, Y, S)[0]:
                mb.append(Y)
        self.mb_set[target] = mb

        # Update the bool_mb_df DataFrame
        self.update_bool_mb_df(target, mb)
        return mb
    
"""
Note: the GSMB algorithm and its variants (IAMB, inter-IAMB, fast-IAMB) require the number of instances to be exponential to the size of the MB.
"""


class Grow_Shrink_learn(MarkovBlanketLearner):
    """
    References:
        - Margaritis D, Thrun S. Bayesian network induction via local neighborhoods[J]. Advances in neural information processing systems, 1999, 12.
        url{https://github.com/cran/bnlearn/blob/c335d14a311ebc00bf482133987f0b40d8073474/R/grow-shrink.R#L30}
    """
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], ci_test: CI_test = None, alpha: float = 0.05, **kwargs):
        super().__init__(data, ci_test, alpha, **kwargs)
    
    def __call__(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        mb = self.get_markov_blanket(target, **kwargs)
        return mb
    
    def get_markov_blanket(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:

        target = self._format_target(target)
        if target in self.mb_set:
            return self.mb_set[target]
        
        """ Get the Markov blanket of the target node using the CI test. """
        
        Candidate = self.observed_variables.copy()
        if target in Candidate:
            Candidate.remove(target)
        else:
            raise ValueError(f"Target {target} is not in the observed variables.")
        
        # whitelisted includes the has been determined MB variables.
        whitelisted = self.get_whitelisted(target)
        mb = []
        mb.extend(whitelisted)  # Add whitelisted variables to MB. Note: ['X1'].extend(['X1', 'X2']) = ['X1', 'X1', 'X2']
        # Remove whitelisted variables from Candidate
        Candidate = [col for col in Candidate if col not in mb]

        # Growing phase
        Flag = True
        while Flag:
            Flag = False
            for Y in Candidate:
                # if len(mb) > max_k: continue  # avoid testing with large conditioning sets.

                if not self.ci_test(target, Y, mb)[0]:
                    mb.append(Y)
                    Candidate.remove(Y)
                    Flag = True
                    break

        # Shrinking phase
        Flag = True
        while Flag:
            Flag = False
            mb_temp = mb.copy()
            for Y in mb_temp:
                S = [var for var in mb if var != Y]  # Create a copy of MB and remove Y from it
                if self.ci_test(target, Y, S)[0]:
                    mb.remove(Y)
                    Flag = True
                    break
                

        self.mb_set[target] = mb
        return mb
    

class IAMB_learn(MarkovBlanketLearner):
    """
    Implementation of the IAMB (Incremental Association Markov Blanket) algorithm.
    References:
        - Tsamardinos I, Aliferis C F, Statnikov A. Algorithms for large scale Markov blanket discovery[C]//Proceedings of the sixteenth international Florida artificial intelligence research society conference. 2003: 376-381.
        url{https://github.com/cran/bnlearn/blob/c335d14a311ebc00bf482133987f0b40d8073474/R/incremental-association.R#L8}
    """
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], ci_test: CI_test = None, alpha: float = 0.05, **kwargs):
        super().__init__(data, ci_test, alpha, **kwargs)

    def __call__(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        mb = self.get_markov_blanket(target, **kwargs)
        return mb

    def get_markov_blanket(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node using the IAMB algorithm.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        target = self._format_target(target)
        if target in self.mb_set:
            return self.mb_set[target]
        
        Candidate = self.observed_variables.copy()
        if target in Candidate:
            Candidate.remove(target)
        else:
            raise ValueError(f"Target {target} is not in the observed variables.")

        # whitelisted includes the has been determined MB variables
        whitelisted = self.get_whitelisted(target)
        mb = []
        mb.extend(whitelisted)  # Add whitelisted variables to MB. Note: ['X1'].extend(['X1', 'X2']) = ['X1', 'X1', 'X2']
        # Remove whitelisted variables from Candidate
        Candidate = [col for col in Candidate if col not in mb]

        # Step 1: Forward phase
        include_flag = True  # Flag to indicate if any variable was added in the last iteration
        while include_flag:
            include_flag = False   # stop if there are no candidates for inclusion.
            temp_p_value = float("inf")   # get the one which maximizes the association measure. 
            best_candidate = None

            # if len(mb) > max_k: break  # stop if the conditioning set has grown too large.

            for Y in Candidate:
                # Perform CI test
                S = mb.copy()
                independent, p_value = self.ci_test(target, Y, S)
                if not independent and p_value < temp_p_value:  # The smaller the p-value, the stronger the dependency.
                    temp_p_value = p_value
                    best_candidate = Y

            if best_candidate is not None:
                mb.append(best_candidate)
                Candidate.remove(best_candidate)
                include_flag = True

        # Step 2: Backward phase
        mb_temp = mb.copy()
        for Y in mb_temp:
            S = [var for var in mb if var != Y]
            if self.ci_test(target, Y, S)[0]:
                mb.remove(Y)

        self.mb_set[target] = mb

        # Update the bool_mb_df DataFrame
        self.update_bool_mb_df(target, mb)

        return mb
    


class IAMB_forward_learn(MarkovBlanketLearner):
    """
    Implementation of the IAMB (Incremental Association Markov Blanket) algorithm without the backward step of deleting variables.
    References:
        - Tsamardinos I, Aliferis C F, Statnikov A. Algorithms for large scale Markov blanket discovery[C]//Proceedings of the sixteenth international Florida artificial intelligence research society conference. 2003: 376-381.
    """
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], ci_test: CI_test = None, alpha: float = 0.05, **kwargs):
        super().__init__(data, ci_test, alpha, **kwargs)

    def __call__(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        mb = self.get_markov_blanket(target, **kwargs)
        return mb
    
    def get_markov_blanket(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node using the IAMB algorithm.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        target = self._format_target(target)
        if target in self.mb_set:
            return self.mb_set[target]
        
        Candidate = self.observed_variables.copy()
        if target in Candidate:
            Candidate.remove(target)
        else:
            raise ValueError(f"Target {target} is not in the observed variables.")

        # whitelisted includes the has been determined MB variables
        whitelisted = self.get_whitelisted(target)
        mb = []
        mb.extend(whitelisted)  # Add whitelisted variables to MB. Note: ['X1'].extend(['X1', 'X2']) = ['X1', 'X1', 'X2']
        # Remove whitelisted variables from Candidate
        Candidate = [col for col in Candidate if col not in mb]

        # Step 1: Forward phase
        include_flag = True  # Flag to indicate if any variable was added in the last iteration
        while include_flag:
            include_flag = False   # stop if there are no candidates for inclusion.
            temp_p_value = float("inf")   # get the one which maximizes the association measure. 
            best_candidate = None

            # if len(mb) > max_k: break  # stop if the conditioning set has grown too large.

            for Y in Candidate:
                # Perform CI test
                S = mb.copy()
                independent, p_value = self.ci_test(target, Y, S)
                if not independent and p_value < temp_p_value:  # The smaller the p-value, the stronger the dependency.
                    temp_p_value = p_value
                    best_candidate = Y

            if best_candidate is not None:
                mb.append(best_candidate)
                Candidate.remove(best_candidate)
                include_flag = True


        self.mb_set[target] = mb
        # Update the bool_mb_df DataFrame
        self.update_bool_mb_df(target, mb)

        return mb
    


class gaussian_MB_learn:
    """
    Implementation of the Gaussian Markov Blanket algorithm.
    References:
        - https://github.com/ban-epfl/rcd/blob/main/rcd/utilities/utils.py#L101
    """
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], alpha: float = None, **kwargs):
        from scipy import stats
        self.num_samples, self.num_nodes = data.shape
        if self.num_samples <= self.num_nodes:
            raise ValueError("Number of samples must be greater than number of nodes.")
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
            self.column_names = list(data.columns)

        elif isinstance(data, np.ndarray):
            self.data = data
            self.column_names = list(range(data.shape[1]))
        else:
            raise ValueError("Unsupported data type. Please provide a pandas DataFrame or a numpy array.")
        
        crr = np.corrcoef(self.data, rowvar=False)
        prec = np.linalg.pinv(crr)
        norm_vec = np.sqrt(np.diag(prec))
        mb_mat = np.abs(prec / norm_vec[:, None] / norm_vec[None, :])

        sig_level = 1 / self.num_nodes ** 2 if alpha is None else alpha

        thresh = np.tanh(stats.norm.ppf(1 - sig_level / 2) / np.sqrt(self.num_samples - self.num_nodes - 1))

        mb_mat = np.where(mb_mat > thresh, 1, -1)  # 1 means MB, -1 means not MB
        # set diagonal to 0
        np.fill_diagonal(mb_mat, 0)

        self.bool_mb_df = pd.DataFrame(
            mb_mat, index=self.column_names, columns=self.column_names
        )
        # print(f"bool_mb_df: {self.bool_mb_df}")

    def __call__(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        mb = [col for col in self.bool_mb_df.columns if self.bool_mb_df.loc[target, col] == 1]
        return mb


class FastIAMB_learn(MarkovBlanketLearner):
    """
    Implementation of the Fast-IAMB (Speculative IAMB) Markov Blanket learning algorithm.

    Unlike IAMB (which adds one variable per outer iteration), Fast-IAMB speculatively
    adds ALL currently-dependent candidates in one Grow phase (sorted by association
    strength, i.e. ascending p-value), then relies on the Shrink phase to remove false
    positives. This typically reduces the number of outer iterations.

    The `_enough_data` guard prevents CI tests whose conditioning sets are too large
    relative to the available samples:
      - For continuous data (default): n > min_samples_per_cond * (|S| + 2)
      - For discrete data:             n >= min_expected_per_cell * 4 * 2^|S|
    If `max_k` is provided, conditioning sets larger than max_k are also skipped.

    References:
        - Yaramakala S, Margaritis D. Speculative Markov blanket discovery for optimal
          feature selection[C]// Fifth IEEE International Conference on Data Mining
          (ICDM'05). IEEE, 2005.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        ci_test: CI_test = None,
        alpha: float = 0.05,
        **kwargs,
    ):
        super().__init__(data, ci_test, alpha, **kwargs)
        self.alpha = alpha
        self.max_k = kwargs.get("max_k", None)          # max conditioning-set size; None = no limit
        self.max_iter = kwargs.get("max_iter", 100)
        self.n_samples = self.data.shape[0]
        # "discrete" or "continuous" – controls the enough_data heuristic
        self.data_type = kwargs.get("data_type", "continuous")
        # Continuous: require n > min_samples_per_cond * (|S| + 2)
        self.min_samples_per_cond = kwargs.get("min_samples_per_cond", 100)
        # Discrete: require n >= min_expected_per_cell * 4 * 2^|S| (assumes ~binary vars)
        self.min_expected_per_cell = kwargs.get("min_expected_per_cell", 1)
        self.skip_shrink = kwargs.get("skip_shrink", False)

    def _enough_data(self, cond: List[Union[int, str]]) -> bool:
        """
        Heuristic: is there enough data for a reliable CI test given conditioning set `cond`?

        Always returns False if max_k is set and |cond| > max_k.
        For continuous data uses a Fisher-z rule of thumb.
        For discrete data uses an average-cell-count rule of thumb.
        """
        if self.max_k is not None and len(cond) > self.max_k:
            return False
        if self.data_type == "discrete":
            # rough lower bound: assume ~2 categories per variable
            return self.n_samples >= self.min_expected_per_cell * 4 * (2 ** len(cond))
        else:
            # Fisher's z: z = atanh(r) * sqrt(n - |S| - 3) ~ N(0,1)
            # needs n > |S| + 3 at minimum; use a generous multiplier for reliability
            return self.n_samples > self.min_samples_per_cond * (len(cond) + 2)

    def __call__(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        return self.get_markov_blanket(target, **kwargs)

    def get_markov_blanket(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Learn the Markov blanket of `target` using Fast-IAMB.
        """
        target = self._format_target(target)
        if target in self.mb_set:
            return self.mb_set[target]

        Candidate = self.observed_variables.copy()
        if target in Candidate:
            Candidate.remove(target)
        else:
            raise ValueError(
                f"Target {target} is not in the observed variables {self.observed_variables}."
            )

        whitelisted = self.get_whitelisted(target)
        mb: List[Union[int, str]] = list(whitelisted)
        Candidate = [col for col in Candidate if col not in mb]

        def get_sorted_candidates():
            """
            Test every candidate conditioned on current MB and return those
            that are NOT independent, sorted ascending by p-value (strongest
            association first).
            """
            S = mb.copy()
            if not self._enough_data(S):
                return []
            result = []
            for Y in Candidate:
                independent, p_value = self.ci_test(target, Y, S)
                # if not independent:
                if p_value < self.alpha:
                    result.append((p_value, Y))
            result.sort(key=lambda x: x[0])
            return result

        candidates = get_sorted_candidates()
        n_iter = 0

        while candidates and n_iter < self.max_iter:
            n_iter += 1
            insufficient_data = False

            # ===== Grow phase =====
            # All candidates were evaluated conditioning on mb at the *start* of this
            # iteration (stored in `candidates`). Speculatively add them all without
            # re-testing – the Shrink phase will remove any false positives.
            for _p, Y in candidates:
                if not self._enough_data(mb):  # stop if MB has grown too large
                    insufficient_data = True
                    break
                if Y not in mb and Y in Candidate:
                    mb.append(Y)
                    Candidate.remove(Y)

            # ===== Shrink phase =====
            removed_any = False
            if not self.skip_shrink:
                for Y in mb.copy():
                    S = [var for var in mb if var != Y]
                    if not self._enough_data(S):
                        # conditioning set too large / sparse – skip this variable
                        continue
                    independent, _ = self.ci_test(target, Y, S)
                    if independent:
                        mb.remove(Y)
                        Candidate.append(Y)
                        removed_any = True

            # If we hit a data-insufficiency wall and nothing was removed,
            # further iterations won't help.
            if insufficient_data and (self.skip_shrink or not removed_any):
                break

            candidates = get_sorted_candidates()

        self.mb_set[target] = mb
        self.update_bool_mb_df(target, mb)
        return mb


def MB_learn(data: Union[pd.DataFrame, np.ndarray], ci_test: CI_test = None, alpha: float = None, **kwargs) -> MarkovBlanketLearner:
    """
    Factory function to create a MarkovBlanketLearner instance based on the specified method.

    :param data: Input dataset (pd.DataFrame or np.ndarray).
    :param ci_test: Instance of CI_test for conditional independence testing.
    :param alpha: Significance level for CI tests.
    :param kwargs: Additional arguments for specific learner methods.
    :return: An instance of a MarkovBlanketLearner subclass.
    """
    mb_method_type = kwargs.get("mb_method_type", "TC")

    if mb_method_type == "TC":
        return TC_learn(data, ci_test, alpha, **kwargs)
    elif mb_method_type == "Grow_Shrink":
        return Grow_Shrink_learn(data, ci_test, alpha, **kwargs)
    elif mb_method_type == "IAMB":
        return IAMB_learn(data, ci_test, alpha, **kwargs)
    elif mb_method_type == "IAMB_forward":
        return IAMB_forward_learn(data, ci_test, alpha, **kwargs)
    elif mb_method_type == "gaussian_MB":
        return gaussian_MB_learn(data, alpha, **kwargs)
    elif mb_method_type == "FastIAMB":
        return FastIAMB_learn(data, ci_test, alpha, **kwargs)
    else:
        raise ValueError(f"Unknown method type: {mb_method_type}")
    




