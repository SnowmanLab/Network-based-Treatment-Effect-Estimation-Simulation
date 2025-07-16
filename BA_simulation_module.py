import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import norm
from tqdm import trange
import os
import pickle


# ------------------------
# Generate BA Network and Adjacency Matrix
# ------------------------
def generate_ba_adjacency_matrix(n_nodes=300, m_edges=3):
    G = nx.barabasi_albert_graph(n=n_nodes, m=m_edges)
    A = nx.to_numpy_array(G)
    return A, G


# ------------------------
# Normalize Adjacency Matrix
# ------------------------
def normalize_adjacency(A):
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return A / row_sums


# ------------------------
# Fraction of treated neighbors
# ------------------------
def get_A_tilde(A, neighbor_degree=1):
    if neighbor_degree == 1:
        A_prod = A.copy()
    elif neighbor_degree == 2:
        A_prod = A @ A
    elif neighbor_degree == 3:
        A_prod = A @ A @ A
    else:
        raise ValueError("neighbor_degree must be 1, 2, or 3")
    np.fill_diagonal(A_prod, 0)
    return normalize_adjacency(A_prod)


# ------------------------
# Generate Confounders and Treatment
# ------------------------
def generate_confounders(n, sigma_C=np.sqrt(1/12), dist="uniform"):
    if dist == "uniform":
        return np.random.uniform(0, np.sqrt(12 * sigma_C**2), n)
    elif dist == "normal":
        return np.random.normal(0, sigma_C, n)
    else:
        raise ValueError("Unsupported distribution")


def h_foo(C):
    return (
        0.15 * (C < 0.33) +
        0.51 * ((0.33 <= C) & (C < 0.66)) +
        0.85 * (0.66 <= C)
    )


def generate_treatment(C):
    probs = h_foo(C)
    return np.random.binomial(1, probs)


# ------------------------
# Generate X-features and Outcome
# ------------------------
def feat_X1_C(A_tilde, W, C):
    W_signed = np.where(W == 0, -1, 1)
    return A_tilde @ (W_signed * C)


def g1(X, C):
    return (
        1.5 * ((X >= 0.5) & (C >= -0.2) & (X < 0.7)) +
        4.0 * ((X >= 0.5) & (C >= -0.2) & (X >= 0.7)) +
        0.5 * ((X >= 0.5) & (C < -0.2)) +
        3.5 * ((X < 0.5) & (C >= -0.2)) +
        2.5 * ((X < 0.5) & (C < -0.2))
    )


def g0(X, C):
    return (
        0.5 * ((X >= 0.4) & (C >= 0.2)) -
        0.75 * ((X >= 0.4) & (C < 0.2)) +
        0.25 * ((X < 0.4) & (C >= 0.2)) -
        0.5 * ((X < 0.4) & (C < 0.2))
    )


def get_error_Y(n, error_type="runif", sigma_Y=0.1):
    if error_type == "runif":
        a = np.sqrt(12 * sigma_Y**2) / 2
        return np.random.uniform(low=-a, high=a, size=n)
    else:
        raise ValueError(f"Unsupported error_type: {error_type}")


def generate_outcomes(W, X, C, sigma_Y=0.1, error_type="runif"):
    n = W.shape[0]
    error_Y = get_error_Y(n, error_type, sigma_Y)
    g1_vals = np.array([g1(x, c) for x, c in zip(X, C)])
    g0_vals = np.array([g0(x, c) for x, c in zip(X, C)])
    Y = W * g1_vals + (1 - W) * g0_vals + error_Y
    theta = np.mean(g1_vals) - np.mean(g0_vals)
    return Y, theta, error_Y


# ------------------------
# Create full data
# ------------------------
def generate_full_data(n, m, sigma_C=np.sqrt(1/12), sigma_Y=0.1, error_type="runif", seed=None):
    if seed is not None:
        np.random.seed(seed)
    A, G = generate_ba_adjacency_matrix(n_nodes=n, m_edges=m)
    A_tilde_deg1 = get_A_tilde(A, neighbor_degree=1)
    C = generate_confounders(n, sigma_C=sigma_C)
    W = generate_treatment(C)
    X = feat_X1_C(A_tilde_deg1, W, C)
    Y, theta, error_Y = generate_outcomes(W, X, C, sigma_Y=sigma_Y, error_type=error_type)
    data = {
        "A": A,
        "G": G,
        "C": C,
        "W": W,
        "X": X,
        "Y": Y,
        "theta": theta,
        "error_Y": error_Y,
        "A_tilde_deg1": A_tilde_deg1
    }
    return data


def generate_dependency_graph(A, model=1):
    n = A.shape[0]
    identity = np.eye(n)
    if model == 1:
        A_dep = A @ (A + identity)
    elif model == 2:
        A_dep = A @ (A @ (A @ (A + identity) + identity) + identity)
    else:
        raise ValueError("Model must be 1 or 2")
    np.fill_diagonal(A_dep, 0)
    A_dep[A_dep > 0] = 1
    return A_dep


def get_conditioning_sets(A, A_dep):
    n = A.shape[0]
    inds_others_A = [np.where(A[i] >= 1)[0].tolist() for i in range(n)]
    inds_depgr_neighb = [np.where(A_dep[i] >= 1)[0].tolist() for i in range(n)]
    return inds_others_A, inds_depgr_neighb


def compute_B_matrix(inds_others_A):
    n = len(inds_others_A)
    B = np.zeros((n, n))
    for i in range(n - 1):
        Xi = set(inds_others_A[i] + [i])
        for j in range(i + 1, n):
            Xj = set(inds_others_A[j] + [j])
            B[i, j] = len(Xi.intersection(Xj))
    degrees_B = np.unique(B[np.triu_indices(n, k=1)])
    degrees_B_indices = {
        l: np.argwhere(np.triu(B, k=1) == l) for l in degrees_B
    }
    return B, degrees_B, degrees_B_indices


def compute_degrees(inds_others_A):
    D = np.array([len(neigh) for neigh in inds_others_A])
    degrees_D = np.unique(D)
    return D, degrees_D, len(degrees_D)


def get_sigma2(pred_g1, pred_g0, pred_h, W, Y, D, degrees_D, len_D, B, degrees_B, degrees_B_indices):
    n = len(W)
    psi = pred_g1 - pred_g0
    psi[W == 1] += (Y[W == 1] - pred_g1[W == 1]) / pred_h[W == 1]
    psi[W == 0] -= (Y[W == 0] - pred_g0[W == 0]) / (1 - pred_h[W == 0])
    psi[~np.isfinite(psi)] = 0.0

    theta_D = {deg: psi[D == deg].mean() for deg in degrees_D}
    Epsi2 = 0.0
    psi_centered = psi.copy()
    for deg in degrees_D:
        inds = np.where(D == deg)[0]
        psi_centered[inds] -= theta_D[deg]
        Epsi2 += np.sum(psi_centered[inds] ** 2) / n

    cov_hat = 0.0
    for l, indices in degrees_B_indices.items():
        for i in range(len_D - 1):
            for j in range(i + 1, len_D):
                di = degrees_D[i]
                dj = degrees_D[j]
                mask_i = (D[indices[:, 0]] == di) | (D[indices[:, 0]] == dj)
                mask_j = (D[indices[:, 1]] == di) | (D[indices[:, 1]] == dj)
                valid = mask_i & mask_j
                idx_i = indices[valid, 0]
                idx_j = indices[valid, 1]
                cov_hat += np.sum(psi_centered[idx_i] * psi_centered[idx_j]) / n

    var_est = (Epsi2 + 2 * cov_hat) / n
    return var_est


def partition_data_with_dependency(A_dep, K, seed=None):
    """
    Partition nodes into K dependency-aware folds.
    Greedily assigns nodes such that within each fold nodes are connected.
    """
    np.random.seed(seed)
    n = A_dep.shape[0]
    all_indices = set(range(n))
    remaining = set(range(n))
    partitions = [[] for _ in range(K)]

    while remaining:
        for k in range(K):
            if not remaining:
                break
            # pick a seed node
            node = remaining.pop()
            current_partition = set([node])
            neighbors = set(np.where(A_dep[node] == 1)[0])
            neighbors &= remaining
            current_partition |= neighbors
            partitions[k].extend(current_partition)
            remaining -= current_partition
    return partitions


def partition_data_uniform(n, K, seed=None):
    np.random.seed(seed)
    indices = np.random.permutation(n)
    return [list(indices[i::K]) for i in range(K)]


def get_independent_complement(I_k, A_dep):
    """
    Returns indices that are not in I_k and also not neighbors of any node in I_k.
    """
    I_k_set = set(I_k)
    n = A_dep.shape[0]
    neighbors = set()
    for i in I_k:
        neighbors.update(np.where(A_dep[i] == 1)[0])
    forbidden = I_k_set.union(neighbors)
    return list(set(range(n)) - forbidden)


def compute_ipw_estimator(Y, W, pred_h_all, Ik_list):
    """
    Compute IPW estimator with dependency-aware cross-fitting (per paper equation).

    Args:
        Y (np.ndarray): Outcome vector of shape (n,)
        W (np.ndarray): Treatment vector of shape (n,)
        pred_h_all (np.ndarray): Estimated propensity scores of shape (n,)
        Ik_list (List[List[int]]): List of K index sets (cross-fitting folds)

    Returns:
        float: Estimated IPW value
    """
    K = len(Ik_list)
    ipw_estimates = []

    for k in range(K):
        Ik = Ik_list[k]
        W_k = W[Ik]
        Y_k = Y[Ik]
        h_k = pred_h_all[Ik]

        # Prevent division by zero
        h_k = np.clip(h_k, 1e-6, 1 - 1e-6)

        term1 = W_k * Y_k / h_k
        term0 = (1 - W_k) * Y_k / (1 - h_k)
        ipw_k = np.mean(term1 - term0)
        ipw_estimates.append(ipw_k)

    return np.mean(ipw_estimates)

# def compute_ipw_variance(Y, W, pred_h, partition):
#     infl_list = []
#     for k in range(len(partition)):
#         Ik = partition[k]
#         h_k = pred_h[Ik]
#         w_k = W[Ik]
#         y_k = Y[Ik]
#         ipw_term = (w_k * y_k / h_k) - ((1 - w_k) * y_k / (1 - h_k))
#         infl_list.append(ipw_term)
#     infl_all = np.concatenate(infl_list)
#     infl_all = infl_all[np.isfinite(infl_all)]
#     return np.var(infl_all) / len(infl_all)



def compute_hajek_estimator(Y, W, pred_h_all, Ik_list):
    """
    Compute Hajek estimator with dependency-aware cross-fitting.

    Args:
        Y (np.ndarray): Outcome vector of shape (n,)
        W (np.ndarray): Treatment vector of shape (n,)
        pred_h_all (np.ndarray): Estimated propensity scores of shape (n,)
        Ik_list (List[List[int]]): List of K index sets (cross-fitting folds)

    Returns:
        float: Estimated Hajek value
    """
    K = len(Ik_list)
    hajek_estimates = []

    for k in range(K):
        Ik = Ik_list[k]
        W_k = W[Ik]
        Y_k = Y[Ik]
        h_k = pred_h_all[Ik]

        # Prevent division by zero
        h_k = np.clip(h_k, 1e-6, 1 - 1e-6)

        w_numerator = np.sum(W_k * Y_k / h_k)
        w_denominator = np.sum(W_k / h_k)

        c_numerator = np.sum((1 - W_k) * Y_k / (1 - h_k))
        c_denominator = np.sum((1 - W_k) / (1 - h_k))

        w_mean = w_numerator / w_denominator if w_denominator != 0 else 0.0
        c_mean = c_numerator / c_denominator if c_denominator != 0 else 0.0

        hajek_k = w_mean - c_mean
        hajek_estimates.append(hajek_k)

    return np.mean(hajek_estimates)

# def compute_hajek_variance(Y, W, pred_h):
#     w = W
#     y = Y
#     h = pred_h

#     infl = (w * y / h) - ((1 - w) * y / (1 - h))
#     infl = infl[np.isfinite(infl)]
#     return np.var(infl) / len(infl)


def run_single_simulation(n, m=2, sigma_C=np.sqrt(1/12), sigma_Y=0.1, error_type="runif",
                          model=1, K=10, S=1, n_bootstrap=300, seed=None, n_estimators=500):
    """
    Full simulation logic that performs S-fold cross-fitting using dependency-aware partitions.
    Calculates netAIPW (with sigma2 and bootstrap), IPW (averaged across folds), and Hajek.
    """
    data = generate_full_data(n=n, m=m, sigma_C=sigma_C, sigma_Y=sigma_Y,
                              error_type=error_type, seed=seed)

    np.random.seed(seed)
    A = data['A']
    # A_tilde = data['A_tilde_deg1']
    W = data['W']
    Y = data['Y']
    C = data['C']
    X = data['X']
    theta_true = data['theta']
    n = len(Y)

    # Dependency graph
    A_dep = generate_dependency_graph(A, model=model)
    inds_others_A, _ = get_conditioning_sets(A, A_dep)
    B, degrees_B, degrees_B_indices = compute_B_matrix(inds_others_A)
    D, degrees_D, len_D = compute_degrees(inds_others_A)

    # Store per-s run
    theta_netAIPW_list = []
    sigma2_list = []
    theta_IPW_list = []
    no_data = 0


    for s in range(S):
        partition = partition_data_uniform(n, K, seed)
        # partition = partition_data_with_dependency(A_dep, K, seed=seed + s if seed is not None else None)
        pred_g1 = np.zeros(n)
        pred_g0 = np.zeros(n)
        pred_h = np.zeros(n)

        for k in range(K):
            Ik = partition[k]
            Ikc = get_independent_complement(Ik, A_dep)
            n_min = 30  
            if len(Ikc) < n_min:
                print(f"For sample size {n}, Fold {k} has too few independent samples ({len(Ikc)}). Skipping.")
                no_data += 1
                continue
            df_train = pd.DataFrame({
                'X': X[Ikc],
                'C': C[Ikc],
                'Y': Y[Ikc],
                'W': W[Ikc]
            })
            df_test = pd.DataFrame({'X': X[Ik], 'C': C[Ik]})

            rf1 = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5)
            rf0 = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5)
            clf_h = RandomForestClassifier(n_estimators=n_estimators, max_depth=2)
            
            rf1.fit(df_train[df_train.W == 1][['X', 'C']], df_train[df_train.W == 1]['Y'])
            rf0.fit(df_train[df_train.W == 0][['X', 'C']], df_train[df_train.W == 0]['Y'])
            clf_h.fit(df_train[['X', 'C']], df_train['W'])

            pred_g1[Ik] = rf1.predict(df_test)
            pred_g0[Ik] = rf0.predict(df_test)
            pred_h[Ik] = clf_h.predict_proba(df_test)[:, 1]

        # netAIPW estimate
        part1 = W * (Y - pred_g1) / pred_h
        part0 = (1 - W) * (Y - pred_g0) / (1 - pred_h)
        infl = pred_g1 - pred_g0 + part1 - part0
        theta_s = np.mean(infl[np.isfinite(infl)])
        theta_netAIPW_list.append(theta_s)

        # IPW estimator from this fold
        theta_IPW_s = compute_ipw_estimator(Y, W, pred_h, partition)
        theta_IPW_list.append(theta_IPW_s)

        # Variance estimation
        sigma2_s = get_sigma2(pred_g1, pred_g0, pred_h, W, Y,
                              D, degrees_D, len_D, B, degrees_B, degrees_B_indices)
        sigma2_list.append(sigma2_s)

    # Aggregate
    theta_netAIPW = np.median(theta_netAIPW_list)
    var_netAIPW = np.median(np.array(sigma2_list) + (np.array(theta_netAIPW_list) - theta_netAIPW)**2)
    
    theta_IPW = np.mean(theta_IPW_list)
    # var_IPW = compute_ipw_variance(Y, W, pred_h, partition)
    
    # Global propensity estimation (for Hajek)
    features_df = pd.DataFrame({'X': X, 'C': C})
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=2)
    clf.fit(features_df, W)
    pred_h_all = clf.predict_proba(features_df)[:, 1]

    # Hajek estimator (single computation)
    theta_Hajek = compute_hajek_estimator(Y, W, pred_h_all, [np.arange(n)])
    # var_Hajek = compute_hajek_variance(Y, W, pred_h_all)


    # Bootstrap variance
    theta_bootstrap = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(n, n, replace=True)
        part1_b = W[sample_idx] * (Y[sample_idx] - pred_g1[sample_idx]) / pred_h[sample_idx]
        part0_b = (1 - W[sample_idx]) * (Y[sample_idx] - pred_g0[sample_idx]) / (1 - pred_h[sample_idx])
        infl_b = pred_g1[sample_idx] - pred_g0[sample_idx] + part1_b - part0_b
        infl_b[~np.isfinite(infl_b)] = 0
        theta_bootstrap.append(np.mean(infl_b))
    var_boot = np.var(theta_bootstrap)

    return {
        "theta_true": theta_true,
        "theta_netAIPW": theta_netAIPW,
        "theta_IPW": theta_IPW,
        "theta_Hajek": theta_Hajek,
        "var_netAIPW": var_netAIPW,
        "var_netAIPW_boot": var_boot,
        "no_data" : no_data,
    }


def save_partial_results(results_list, save_dir, n, suffix="partial"):
    """
    Save intermediate simulation results to file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"simulation_results_n{n}_{suffix}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results_list, f)


def load_partial_results(save_dir, n, suffix="partial"):
    """
    Load previously saved results.
    """
    save_path = os.path.join(save_dir, f"simulation_results_n{n}_{suffix}.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)
    else:
        return []


def run_simulations_over_n(n_list, num_simulations=1000, save_dir="simulation_outputs", seed_offset=0):
    """
    Run simulations for different sample sizes n and save results periodically.
    """
    all_results = {}
    for n in n_list:
        print(f"\nRunning simulations for n = {n}")
        partial_results = load_partial_results(save_dir, n)
        start_idx = len(partial_results)
        print(f"Starting from simulation #{start_idx + 1}")

        for i in tqdm(range(start_idx, num_simulations), desc=f"Simulating n={n}"):
            seed = seed_offset + i
            
            # result = run_single_simulation(n=n, seed=seed)
            # partial_results.append(result)

            # # Save after each iteration
            # save_partial_results(partial_results, save_dir, n)
            
            try:
                result = run_single_simulation(n=n, seed=seed)
                partial_results.append(result)

                # Save after each iteration
                save_partial_results(partial_results, save_dir, n)

            except Exception as e:
                print(f"Simulation {i} failed for n = {n}: {e}")

        all_results[n] = partial_results

    return all_results

