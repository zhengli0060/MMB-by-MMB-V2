import logging
import time
from typing import List, Optional, Union
from Utils.PartMixGraph import PartMixGraph, Mark, Node, SeparationSet
from Utils.CI_test import CI_test
from itertools import combinations
import networkx as nx
import pandas as pd
from Utils.Markov_Blanket_Learner import MB_learn
from Utils.util_tools import sanitize_data

"""
NOTE: The 'Node' only be used in  the Graph learning process, we don't use it in the other process, e.g., MB learning.
"""
logger = logging.getLogger("Local_Learner") 
logger.setLevel(logging.INFO)


def oracle_MMB_by_MMB(data:pd.DataFrame, target: str, latent_nodes: list[str]=[]) -> PartMixGraph:
    """
    Oracle version of MMB-by-MMB algorithm, which uses the true Markov Blanket and D-separation test.
    """
    time_start = time.time()

    ci_test_obj = CI_test(data, method="d_sep", latent_nodes=latent_nodes)
    local_learner = Local_Learner(data, target=target, ci_test_obj=ci_test_obj, version='oracle', latent_nodes=latent_nodes)
    local_pag = local_learner.oracle_local_learner()
    ci_num = local_learner.get_ci_test_number()
    time_end = time.time()
    return {'PAG.DataFrame': local_pag._to_pandas_adjacency(), 'PAG.PartMixGraph': local_pag, 'CI_num': ci_num, "runtime_sec": time_end - time_start}



def data_MMB_by_MMB(data:pd.DataFrame, target: str, ci_method: str="fisherz", mb_method: str="gaussian_MB", max_depth: int=100, alpha: float=0.01, max_K: int=3, **kwargs) -> PartMixGraph:
    """
    Data version of MMB-by-MMB algorithm, which uses the observed data and Fisher's Z test.
    """
    time_start = time.time()
    ci_test_obj = CI_test(data, method=ci_method, alpha=alpha, **kwargs)
    local_learner = Local_Learner(data, target=target, ci_test_obj=ci_test_obj, version='data', mb_method=mb_method, max_K=max_K, whether_fast=kwargs.get("whether_fast", True))
    local_pag = local_learner.local_structure_learner(max_depth=max_depth)
    ci_num = local_learner.get_ci_test_number()
    time_end = time.time()
    return {'PAG.DataFrame': local_pag._to_pandas_adjacency(), 'CI_num': ci_num, "runtime_sec": time_end - time_start} 




class Local_Learner:

    def __init__(self, data, target: str, ci_test_obj: CI_test, version: str='oracle', **kwargs):
        
        data = sanitize_data(data)
        self.all_vars = data.columns.to_list()
        self.ci_test = ci_test_obj
        self.target = target

        
        if version == 'oracle':  # For the orcale case, ci_type should be "d_sep"
            assert ci_test_obj.method == "d_sep", "For oracle version, ci_test_obj must be a d_sep test object."
            self.dag = nx.DiGraph(data)  # Create a directed graph from the adjacency matrix
            self.latent_nodes = kwargs.get("latent_nodes", [])  # List of latent nodes, default is None

            self.observed_vars = [var for var in self.all_vars if var not in self.latent_nodes]
            # For oracle case, we can directly use the true Markov Blanket
            self.mb_learner = MB_learn(data, ci_test=self.ci_test, mb_method_type='TC',**kwargs)

            self.ancList = {}
            for var in self.all_vars:
                self.ancList[var] = set(nx.ancestors(self.dag, var))
        else:    
            self.observed_vars = self.all_vars
            mb_method = kwargs.get("mb_method", None)
            assert mb_method is not None, "mb_method must be specified in kwargs, e.g., mb_method='gaussian_MB' for continuous data."
            mb_alpha = kwargs.get("mb_alpha", 1 / len(self.observed_vars)**2)  # Significance level for MB learning
            self.whether_fast = kwargs.get("whether_fast", True)  # Whether to use the fast version of local structure learning, True means using the fast version, False means using the complete version.
            self.mb_learner = MB_learn(data, ci_test=self.ci_test, alpha=mb_alpha, mb_method_type=mb_method,**kwargs) 
        
        self.max_K = kwargs.get("max_K", len(self.observed_vars)-2)  # Maximum size of the separation set, default is 100
        self.sepsets = SeparationSet(set(self.observed_vars))
        self.pag = PartMixGraph(incoming_graph_data=self.observed_vars)
        self.target_Node = self.pag._str_Node[self.target] 
        self._cache_mb = dict()  # Cache for Markov Blankets
        self.rank_collider = 1

    def get_mb(self, node: Node) -> set[Node]:
        """
        Get the Markov Blanket of the target node.
        """
        if node in self._cache_mb:
            return self._cache_mb[node]

        mb = self.mb_learner(node.name)
        for n in self.observed_vars:
            if n not in mb and n != node:
                if self.sepsets.has_sepset(node, n):
                    continue
                self.sepsets.add_sepset(node, n, set(mb))
                self.ci_test._ci_num += 1


        mb_Nodes = {self.pag._str_Node[name] for name in mb}
        self._cache_mb[node] = mb_Nodes
        return mb_Nodes

    def get_ci_test_number(self) -> int:
        """
        Get the number of CI tests performed.
        """
        return self.ci_test.get_ci_num()  # including the number of CI tests performed in the MB learning process and skeleton learning process
    
    def oracle_skeleton_learning(self, sub_Nodes: list[Node]) -> PartMixGraph:
        """
        Oracle skeleton learning using D-separation.
        """
        sub_vars = {node.name for node in sub_Nodes}
        sub_graph = PartMixGraph(incoming_graph_data=sub_Nodes)
        sub_graph._init_complete_graph()
        sepsets = SeparationSet(sub_vars)
        logger.info(f'Initial graph information: {sub_graph}')
        for x_node, y_node in sub_graph.edges():
            logger.info(f'Checking edge {x_node} -- {y_node} for oracle skeleton learning')
            sepset_candidates = self.ancList[x_node.name] | self.ancList[y_node.name]

            sepset_candidates = sepset_candidates - {x_node.name, y_node.name}
            sepset = sepset_candidates & sub_vars  ## consider sub_vars is the observed variables in the current sub_graph

            if self.ci_test(x_node.name, y_node.name, list(sepset))[0]:
                sepsets.add_sepset(x_node, y_node, sepset)
                self.sepsets.add_sepset(x_node, y_node, sepset) # store the sepset in the global sepsets
                sub_graph.remove_edge(x_node, y_node) #
                logger.info(f'remove {x_node} -- {y_node} via oracle skeleton learning')

        self.orient_collider(sub_graph, sepsets)
        self.orient_rules(sub_graph, sepsets)

        return sub_graph 

    def update_subpag_to_pag(self, sub_pag: PartMixGraph, target_node: Node):
        """
        Update the information in sub_pag that is guaranteed to be correct into self.pag.
        """
        # Update all edges incident to the target node.
        for node in sub_pag.get_adj_nodes(target_node):
            edge = sub_pag.get_Edge(target_node, node)
            if not self.pag.has_edge(target_node, node):
                self.pag._insert_edge(target_node, node, edge.lmark, edge.rmark)
            else:
                self.pag.update_edge(node1=target_node, 
                                     lmark=edge.lmark if edge.lmark != Mark.CIRCLE else None,
                                     rmark=edge.rmark if edge.rmark != Mark.CIRCLE else None, node2=node)

        # Update all V-structures involving the target node.
        collider_paths = sub_pag.get_all_uncovered_collider_paths_from_target(target_node)
        for path in collider_paths:
            for i in range(len(path) - 1):
                edge = sub_pag.get_Edge(path[i], path[i + 1])
                if not self.pag.has_edge(path[i], path[i + 1]):
                    self.pag._insert_edge(path[i], path[i + 1], edge.lmark, edge.rmark)
                else:
                    self.pag.update_edge(node1=path[i], 
                                         lmark=edge.lmark if edge.lmark != Mark.CIRCLE else None,
                                         rmark=edge.rmark if edge.rmark != Mark.CIRCLE else None, node2=path[i + 1])
                    
        self.orient_collider(self.pag, self.sepsets)
        self.orient_rules(self.pag, self.sepsets)

    def update_subpag_to_pag_data(self, sub_pag: PartMixGraph, target_node: Node, sepsets: SeparationSet):
        """
        Update the information in sub_pag that is guaranteed to be correct into self.pag.
        """
        # Update all edges incident to the target node.
        for node in sub_pag.get_adj_nodes(target_node):
            edge = sub_pag.get_Edge(target_node, node)
            if not self.pag.has_edge(target_node, node):
                self.pag._insert_edge(target_node, node, edge.lmark, edge.rmark)
            else:
                self.pag.update_edge(node1=target_node, 
                                     lmark=edge.lmark if edge.lmark != Mark.CIRCLE else None,
                                     rmark=edge.rmark if edge.rmark != Mark.CIRCLE else None, node2=node)

        # Update all V-structures involving the target node.
        collider_paths = sub_pag.get_all_uncovered_collider_paths_from_target(target_node)
        for path in collider_paths:
            for i in range(1, len(path) - 1):
                if len(sepsets.get_sepset(path[i-1], path[i+1])) <= self.rank_collider:  # At this point, treat this V-structure as reliable.
                    edge = sub_pag.get_Edge(path[i], path[i + 1])
                    if not self.pag.has_edge(path[i], path[i + 1]):
                        self.pag._insert_edge(path[i], path[i + 1], edge.lmark, edge.rmark)
                    else:
                        self.pag.update_edge(node1=path[i], 
                                            lmark=edge.lmark if edge.lmark != Mark.CIRCLE else None,
                                            rmark=edge.rmark if edge.rmark != Mark.CIRCLE else None, node2=path[i + 1])
                else:
                    break  # If the current V-structure is not reliable, longer ones are not reliable either.
                    
        self.orient_collider(self.pag, self.sepsets)
        self.orient_rules(self.pag, self.sepsets)


    def update_waitlist(self, max_depth: int, donelist: list[Node]):
        waitlist = []
        visited = {self.target_Node}
        current_level = [self.target_Node]
        
        for depth in range(max_depth):
            next_level = []
            for node in current_level:
                # Get nodes connected by an undirected path.
                neighbors = self.pag.get_adj_nodes(node)   #   
                for neighbor in neighbors:
                    if self.pag.has_star_arrow_edge(node, neighbor):
                        continue
                    if self.pag.has_definite_edge(node, neighbor):
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        waitlist.append(neighbor)
            
            if not next_level:
                break  # Stop early when there are no new nodes to explore.
            current_level = next_level

        waitlist = [node for node in waitlist if node not in donelist]
        # Check whether the waitlist contains duplicate nodes.
        if len(waitlist) != len(set(waitlist)):
            raise ValueError("Waitlist contains duplicate nodes, which should not happen.")
        return waitlist



    def oracle_local_learner(self, max_depth: int=10**18) -> PartMixGraph:
        
        waitlist = [self.target_Node]
        donelist = []
        while (waitlist):
            node = waitlist.pop(0)
            mbplus_node_list = list(self.get_mb(node) | {node})
            local_pag = self.oracle_skeleton_learning(mbplus_node_list)
            self.update_subpag_to_pag(local_pag, node)
            donelist.append(node)
            waitlist = self.update_waitlist(max_depth, donelist)
        
        return self.pag

    def learn_skeleton_base_data(self, sub_Nodes: list[Node]):
        """
        Learn the skeleton of the graph based on the provided subset of all nodes.
        """
        sub_vars = {node.name for node in sub_Nodes}
        sub_graph = PartMixGraph(incoming_graph_data=sub_Nodes)
        logger.info(f'Initial graph information: {sub_graph}')
        sub_graph._init_complete_graph()
        sepsets = SeparationSet(sub_vars)

        for x, y in combinations(sub_Nodes, 2):
            if self.sepsets.has_sepset(x, y):
                try:
                    sepsets.add_sepset(x, y, self.sepsets.get_sepset(x, y))
                    sub_graph.remove_edge(x, y)
                    logger.info(f'remove {x} -- {y} via has sepset')
                except ValueError:
                    continue

        sep_size: int = 0
        while sep_size <= self.max_K and (sub_graph.max_degree() - 1 >= sep_size):
            for x_node in sub_graph.Node_list:
                adj_x_nodes = sub_graph.get_adj_nodes(x_node)
                for y_node in adj_x_nodes:
                    adj_node_x_noy = adj_x_nodes - {y_node}
                    adj_node_x_noy = set(node.name for node in adj_node_x_noy)
                    if len(adj_node_x_noy) < sep_size:
                        continue
                    logger.info(f'Checking edge {x_node} -- {y_node} for sep_size={sep_size}')
                    if self.pag.has_edge(x_node, y_node):    # We can add some background knowledge here
                        logger.info(f'skip {x_node} -- {y_node} because cpdag has this edge')
                        continue
                    for sep in combinations(adj_node_x_noy, sep_size):
                        """
                        The CI test not accept the Node type, so we need to convert it to string.
                        """
                        if self.ci_test(x_node.name, y_node.name, list(sep))[0]:
                            sepsets.add_sepset(x_node, y_node, set(sep))
                            self.sepsets.add_sepset(x_node, y_node, set(sep))
                            sub_graph.remove_edge(x_node, y_node) # 
                            break
            sep_size += 1

        self.orient_collider(sub_graph, sepsets)

        if not self.whether_fast:
            for x_node in sub_graph.Node_list:
                pds_set_x = sub_graph.get_possible_d_sep(x_node, maxlength=self.max_K)
                adj_list_x = list(sub_graph.get_adj_nodes(x_node))
                for y_node in adj_list_x:
                    if self.pag.has_edge(x_node, y_node):    # We can add some background knowledge here
                        logger.info(f'skip {x_node} -- {y_node} because pag has this edge')
                        continue
                    sep_size = 0
                    while sep_size <= self.max_K and (len(pds_set_x) - 1 >= sep_size):
                        logger.info(f'Checking edge {x_node} -- {y_node} for sep_size={sep_size} in pds')
                        pds_node_x_noy = pds_set_x - {y_node}
                        pds_node_x_noy = set(node.name for node in pds_node_x_noy)
                        for sepset in combinations(pds_node_x_noy, sep_size):
                            """
                            The CI test not accept the Node type, so we need to convert it to string.
                            """
                            if self.ci_test(x_node.name, y_node.name, list(sepset))[0]:
                                sepsets.add_sepset(x_node, y_node, set(sepset))
                                self.sepsets.add_sepset(x_node, y_node, set(sepset))
                                sub_graph.remove_edge(x_node, y_node) #
                                break
                        if not sub_graph.has_edge(x_node, y_node):
                            break  # Edge already removed, no need to check larger sep_size
                        sep_size += 1        
            sub_graph.clear_all_orientations()


        return sub_graph, sepsets


    def local_structure_learner(self, max_depth: int=None) -> PartMixGraph:
        if max_depth is None:
            max_depth = int(len(self.observed_vars)/10)
        waitlist = [self.target_Node]
        donelist = []
        while (waitlist):
            node = waitlist.pop(0)
            mbplus_node_list = list(self.get_mb(node) | {node})
            local_pag, sepsets = self.learn_skeleton_base_data(mbplus_node_list)
            self.update_subpag_to_pag_data(local_pag, node, sepsets)
            donelist.append(node)
            waitlist = self.update_waitlist(max_depth, donelist)
        
        return self.pag
  


    def orient_collider(self, undirected_graph: PartMixGraph, sepsets: SeparationSet):
        
        Cand_triplets = undirected_graph.find_unique_triplets()
        for (z, y, x) in Cand_triplets:
            if sepsets.has_sepset(x, z): # in local causal discovery, we check the existence of sepset of x and z
                if not sepsets.is_in_sepset(target=y, node1=x, node2=z):
                    undirected_graph.update_edge(node1=x, lmark=None, rmark=Mark.ARROW, node2=y)
                    undirected_graph.update_edge(node1=z, lmark=None, rmark=Mark.ARROW, node2=y)
                    logger.info(f"Orienting collider: {x} *-> {y} <-* {z}")

    
    def orient_rules(self, pag: PartMixGraph, sepsets: SeparationSet):

        """
        Apply orientation rules to the PAG.
        Reference: Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias, Artificial Intelligence 172 (16-17), 1873-1896.
        The logical structure is inspired by the pcalg implementation, but all rules are re-implemented here.
        """

        
        update_flag = True
        while update_flag:  # Continue until no changes are made
            update_flag = False
            pag, update_flag = self.Rule_1(pag, sepsets, update_flag)
            pag, update_flag = self.Rule_2(pag, sepsets, update_flag)
            pag, update_flag = self.Rule_3(pag, sepsets, update_flag)
            pag, update_flag = self.Rule_4(pag, sepsets, update_flag)

        update_flag = True
        while update_flag:  # Continue until no changes are made
            update_flag = False
            pag, update_flag = self.Rule_8(pag, sepsets, update_flag)
            pag, update_flag = self.Rule_9(pag, sepsets, update_flag)
            pag, update_flag = self.Rule_10(pag, sepsets, update_flag)



    def Rule_1(self, pag: PartMixGraph, sepsets: SeparationSet, update_flag: bool) -> PartMixGraph:
        """
        If a *-> b o-* r, and a and r are not adjacent, then orient the triple as a *-> b -> r.
        """
        for b, r in pag.get_circ_star_edge(): # Get edges of the form b o-* r
            for a in pag.get_into_nodes(b):
                if sepsets.has_sepset(a, r) and \
                    sepsets.is_in_sepset(target=b, node1=a, node2=r):
                        pag.update_edge(node1=b, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                        update_flag = True
                        logger.info(f"Orienting Rule 1: {b} o-* {r} to {b} --> {r}")
                        break # No need to check other As for this b, r pair

        return pag, update_flag        

    def Rule_2(self, pag: PartMixGraph, sepsets: SeparationSet, update_flag: bool) -> PartMixGraph:
        """
        If a -> b *-> r or a *-> b -> r, and a *–○ r, then orient a *–○ r as a *-> r.
        """
        for r, a in pag.get_circ_star_edge():  # edges of the form a *-o r
            for b in pag.get_into_nodes(r):  # b *-> r
                if pag.has_directed_edge(a, b) or \
                    (pag.has_into_edge(a, b) and pag.has_out_edge(b, r)):
                    pag.update_edge(node1=a, lmark=None, rmark=Mark.ARROW, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 2: {a} *-o {r} to {a} *-> {r}")
                    break
        return pag, update_flag

    def Rule_3(self, pag: PartMixGraph, sepsets: SeparationSet, update_flag: bool) -> PartMixGraph:
        """
        If a *-> b <-* r, a *-o t o-* r, a and r are not adjacent, and t *-o b,
        then orient t *-o b as t *-> b.
        """
        for b, t in pag.get_circ_star_edge():  # Get edges of the form t *-o b
            Cand_ar = pag.get_into_nodes(b)   # a or r *-> b
            Cand_ar = {a for a in Cand_ar if pag.has_into_edge(a, b)}  # Only consider a *-> b
            if len(Cand_ar) >= 2:
                for a, r in combinations(Cand_ar, 2):
                    if sepsets.has_sepset(a, r) and \
                        sepsets.is_in_sepset(target=t, node1=a, node2=r):  # otherwise t is a collider on <a, t, r>
                            pag.update_edge(node1=t, lmark=None, rmark=Mark.ARROW, node2=b)
                            update_flag = True
                            logger.info(f"Orienting Rule 3: {t} *-o {b} to {t} *-> {b}")
                            break
        return pag, update_flag

    def updateList(self, path, new_ts, old_path_list):  # arguments are all lists
        """
        Update the list of paths by adding new paths formed with elements from the given set.
        """
        return old_path_list + [path + [t] for t in new_ts]

    def minDiscrPath(self, a: Node, b: Node, r: Node, pag: PartMixGraph, sepsets: SeparationSet) -> Optional[List[Node]]:
        """
        Find the minimal discriminating path between two nodes given a third node.
        We had a path a <-* b o-* r and a -> r, then we need to find the minDiscrPath for Rule4 in Zhang 2008.
        Parameters:
        - a: The first node.
        - b: The second node.
        - r: The third node.


        Returns:
        - A list of nodes representing the minimal discriminating path, or None if no such path exists.
        """
        Cand_ts = pag.get_into_nodes(a)  # Get all nodes that point to a 
        visited = {a, b, r}  # Nodes already visited
        Cand_ts = Cand_ts - visited  # Remove visited nodes
        if len(Cand_ts) == 0:
            return None
        
        list_paths = self.updateList([a], Cand_ts, [])  # Initialize paths with a and candidates

        while list_paths:
            path = list_paths.pop(0)
            cand_t = path[-1]  # Last node in the current path

            if sepsets.has_sepset(cand_t, r):  # not adjacent
                return path[:: -1] + [b, r]  # t *-> a <-* b o-* r

            pred_t = path[-2]
            visited.add(cand_t)  # Mark the current node as visited

            if pag.has_directed_edge(cand_t, r) and pag.has_into_edge(pred_t, cand_t): #  t <-> a <-* b o-* r and t -> r
                Cand_ts = pag.get_into_nodes(cand_t) - visited  # Get unvisited nodes that point to t
                if len(Cand_ts) > 0:
                    list_paths = self.updateList(path, Cand_ts, list_paths)

        return None

    def Rule_4(self, pag: PartMixGraph, sepsets: SeparationSet, update_flag: bool) -> PartMixGraph:
        """
        If u = <t, ..., a, b, r> is a discriminating path between t and r for b, and b o-* r;
        then if b ∈ SepSet(t, r), orient b o-* r as b -> r; otherwise orient the triple <a, b, r> as a <-> b <-> r.
        """
        for b, r in pag.get_circ_star_edge():  # Get edges of the form b o-* r
            # a -> r and b *-> a
            Cand_as = pag.get_parents(r)
            Cand_as = {a for a in Cand_as if pag.has_into_edge(b, a)}
            while len(Cand_as) > 0:
                a = Cand_as.pop()  # Take one candidate a
                md_path = self.minDiscrPath(a, b, r, pag, sepsets)  
                if md_path is not None:
                    t = md_path[0]
                    if sepsets.is_in_sepset(target=b, node1=t, node2=r):
                        pag.update_edge(node1=b, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                        
                        logger.info(f"Orienting Rule 4: {b} o-* {r} to {b} -> {r}")
                    else:
                        pag.update_edge(node1=a, lmark=Mark.ARROW, rmark=Mark.ARROW, node2=b)
                        pag.update_edge(node1=b, lmark=Mark.ARROW, rmark=Mark.ARROW, node2=r)
                        logger.info(f"Orienting Rule 4: {a} <-> {b} <-> {r}")
                    update_flag = True
                    break  # No need to check other As for this b, r pair
        return pag, update_flag

   
        
    def Rule_8(self, pag: PartMixGraph, sepsets: SeparationSet, update_flag: bool) -> PartMixGraph:
        """
        If a -> b -> r or a -o b -> r, and a o-> r, orient a o-> r as a -> r.
        """
        for a, r in pag.get_circ_arrow_edge():
            for b in pag.get_parents(r):
                if pag.has_directed_edge(a, b) or pag.has_tail_circ_edge(a, b):
                    pag.update_edge(node1=a, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 8: {a} o-> {r} to {a} -> {r}")
                    break
        return pag, update_flag
    
    def is_uncovered_path(self, path: List[Node], sepsets: SeparationSet) -> bool:
        """
        Check if the given path is uncovered by sepsets for 'minUncovPdPath' and 'minUncovCircPath' in orient rules.
        In local learning, direct adjacency between nodes may not be fully determined. Therefore, we use self.is_uncovered_path (which relies on separation sets) to check for uncovered paths in orientation rules, rather than PartMixGraph.is_uncovered_path, to ensure correctness in these cases.
        """
        for i in range(1, len(path) - 1):
            if not (sepsets.has_sepset(path[i - 1], path[i + 1]) and sepsets.is_in_sepset(target=path[i], node1=path[i - 1], node2=path[i + 1])):
                return False
        return True

    def minUncovPdPath(self, a: Node, b: Node, r: Node, pag: PartMixGraph, sepsets: SeparationSet) -> Optional[List[Node]]:
        """
            Find a minimal uncovered pd path from initial (a,b,r)
        """
        Cand_ts = pag.get_pd_path_nodes(b) 
        visited = {a, b, r}  # Nodes already visited
        Cand_ts = Cand_ts - visited  # Remove visited nodes
        if len(Cand_ts) == 0:
            return None

        list_paths = self.updateList([b], Cand_ts, [])  # Initialize paths with b and candidates

        while list_paths:
            path = list_paths.pop(0)
            cand_t = path[-1]  # Last node in the current path
            visited.add(cand_t)  # Mark the current node as visited
            tr_edge = pag.get_Edge(cand_t, r)
            if tr_edge is not None and \
                tr_edge.lmark != Mark.ARROW and tr_edge.rmark != Mark.TAIL:
                    mpath = [a] + path + [r] # <a, b, ..., t ,r>
                    if self.is_uncovered_path(mpath, sepsets):
                        return mpath

            else:
                Cand_tis = pag.get_pd_path_nodes(cand_t)
                Cand_tis = Cand_tis - visited  # Remove visited nodes
                if len(Cand_tis) > 0:
                    list_paths = self.updateList(path, Cand_tis, list_paths)

        return None

    def Rule_9(self, pag: PartMixGraph, sepsets: SeparationSet, update_flag: bool) -> PartMixGraph:
        """
        If a o-> r, and p = <a, b, ..., r> is an uncovered potentially directed path from a to r such that r and b are not adjacent, then orient a o-> r as a -> r.
        """
        for a, r in pag.get_circ_arrow_edge():
            ## find all b s.t. a (o-)--(o>) b and b and r are not connected
            Cand_bs = pag.get_pd_path_nodes(a) # a (o-)--(o>) b
            Cand_bs = {b for b in Cand_bs if not pag.has_edge(b, r)}  # Remove candidates that are adjacent to r
            Cand_bs.remove(r)  # Remove r from candidates
            while len(Cand_bs) > 0:
                b = Cand_bs.pop()
                if not (sepsets.has_sepset(b, r) and \
                    sepsets.is_in_sepset(target=a, node1=b, node2=r)): 
                    # Verify whether the separation set of b and r has been identified, and check if a is included in the separation set of b and r. Note that a is defined not to be a collider in the triple <b, a, r>.
                    continue
                upd_path = self.minUncovPdPath(a, b, r, pag, sepsets)
                if upd_path is not None:
                    pag.update_edge(node1=a, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 9: {a} o-> {r} to {a} -> {r} via {b}")
                    break

        return pag, update_flag

    def Rule_10(self, pag: PartMixGraph, sepsets: SeparationSet, update_flag: bool) -> PartMixGraph:

        """
        R10: Suppose a o-> r, b -> r <- t, p1 is an uncovered p.d. path from a to b, and p2 is an uncovered p.d. path from a to t.
        Let u be the vertex adjacent to a on p1 (u could be b), and w be the vertex adjacent to a on p2 (w could be t).
        If u and w are distinct, and are not adjacent, then orient a o-> r as a -> r.
        """
        for a, r in pag.get_circ_arrow_edge():
            pa_r = pag.get_parents(r)  # Get all parents of r
            if len(pa_r) < 2:
                continue
            for b, t in combinations(pa_r, 2):
                ## this is the easiest one
                if pag.has_pd_edge(a, b) and pag.has_pd_edge(a, t) and \
                    sepsets.has_sepset(b, t) and \
                    sepsets.is_in_sepset(target=a, node1=b, node2=t):
                    pag.update_edge(node1=a, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 10: {a} o-> {r} to {a} -> {r} via {b} and {t}")
                    break  # No need to check other pairs of b and t for this a, r pair
                else:
                    Cand_uw = pag.get_pd_path_nodes(a)  # Get all nodes that a (o-)--(o>) can reach
                    Cand_uw.remove(r) # Remove r from candidates
                    if len(Cand_uw) < 2:
                        continue
                    for u, w in combinations(Cand_uw, 2):
                        # for speed
                        if pag.has_edge(u, w):
                            continue  # u and w are adjacent, skip this pair

                        if u == b:
                            p1 = [a, b]
                        else:
                            p1 = self.minUncovPdPath(a, u, b, pag, sepsets)
                        if p1 is not None:
                            if w == t:
                                p2 = [a, t]
                            else:
                                p2 = self.minUncovPdPath(a, w, t, pag, sepsets)

                            if p2 is not None and u!=w and \
                                sepsets.has_sepset(u, w) and \
                                sepsets.is_in_sepset(target=a, node1=u, node2=w):
                                    pag.update_edge(node1=a, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                                    update_flag = True
                                    logger.info(f"Orienting Rule 10: {a} o-> {r} to {a} -> {r} via {b} and {t}")
                                    break
                    if pag.has_directed_edge(a, r):
                        break  # No need to check other pairs of b and t for this a, r pair
        return pag, update_flag

    




