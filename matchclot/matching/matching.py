import ot
import numpy as np
import scipy
import networkx


def OT_matching(sim_matrix, entropy_reg=0.01):
    weights = sim_matrix

    # Negative weights are interpreted as costs, scale them to be in [0, 1]
    C = -weights + 1
    C -= C.min()
    C += 1
    C = C / C.max()
    del weights

    # Uniform distributions on cell profiles
    mod1_distr = np.ones(sim_matrix.shape[0]) / sim_matrix.shape[0]
    mod2_distr = np.ones(sim_matrix.shape[1]) / sim_matrix.shape[1]

    if entropy_reg > 0:
        # Entropy-regularized OT
        bipartite_matching_adjacency = ot.sinkhorn(
            mod1_distr, mod2_distr, reg=0.01, M=C, numItermax=5000, verbose=True
        )
    else:
        # Exact OT
        bipartite_matching_adjacency = ot.emd(
            mod1_distr, mod2_distr, M=C, numItermax=10000000
        )

    return bipartite_matching_adjacency


def MWB_matching(raw_logits, threshold_quantile=0.99):
    while threshold_quantile >= 0:
        try:
            weights = raw_logits.copy()
            # Discard weights smaller than the threshold quantile
            quantile_row = np.quantile(
                weights, threshold_quantile, axis=0, keepdims=True
            )
            quantile_col = np.quantile(
                weights, threshold_quantile, axis=1, keepdims=True
            )
            mask_ = weights < quantile_row
            mask_ = np.logical_and(mask_, (weights < quantile_col), out=mask_)
            weights[mask_] = 0
            weights_sparse = scipy.sparse.csr_matrix(-weights)
            del weights
            graph = networkx.algorithms.bipartite.matrix.from_biadjacency_matrix(
                weights_sparse
            )
            u = [n for n in graph.nodes if graph.nodes[n]["bipartite"] == 0]
            matches = (
                networkx.algorithms.bipartite.matching.minimum_weight_full_matching(
                    graph, top_nodes=u
                )
            )
            best_matches = np.array([matches[x] - len(u) for x in u])
            bipartite_matching_adjacency = np.zeros(raw_logits.shape)
            bipartite_matching_adjacency[
                np.arange(raw_logits.shape[0]), best_matches
            ] = 1
            return bipartite_matching_adjacency
        except networkx.exception.NetworkXException:
            threshold_quantile -= 0.01
            print(
                "Discarding too many edges, problem infeasible, lowering threshold to {}".format(
                    threshold_quantile
                )
            )
    raise RuntimeError("No feasible solution found")
