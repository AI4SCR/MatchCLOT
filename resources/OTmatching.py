import ot
import numpy as np

def get_OT_bipartite_matching_adjacency_matrix(sim_matrix, epsilon=0.01):
    weights = sim_matrix

    # Negative weights are interpreted as costs, scale them to be in [0, 1]
    C = -weights + 1
    C -= C.min()
    C += 1
    C = C/C.max()
    del weights

    # Uniform distributions on cell profiles
    mod1_distr = np.ones(sim_matrix.shape[0]) / sim_matrix.shape[0]
    mod2_distr = np.ones(sim_matrix.shape[1]) / sim_matrix.shape[1]

    if epsilon > 0:
        # Entropy-regularized OT
        bipartite_matching_adjacency = ot.sinkhorn(mod1_distr, mod2_distr, reg=0.01, M=C, numItermax=5000, verbose=True)
    else:
        # Exact OT
        bipartite_matching_adjacency = ot.emd(mod1_distr, mod2_distr, M=C, numItermax=10000000)

    return bipartite_matching_adjacency