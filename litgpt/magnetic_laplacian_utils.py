import numpy as np
import networkx as nx
from litgpt.positional_encodings_config import magentic_laplace_encodings_q

from scipy.sparse import csr_matrix


def new_magnetic_laplacian(G, q):
    """
    Computes the Magnetic Laplacian of a directed graph G with potential q.
    
    Parameters:
    G : networkx.DiGraph
        The directed graph for which the magnetic Laplacian is computed.
    q : float
        The magnetic potential, q >= 0.
        
    Returns:
    L_q : np.ndarray
        The Magnetic Laplacian matrix.
    """
    
    # Get the number of nodes in the graph
    n = len(G.nodes)
    
    # Initialize the degree matrix D_s and adjacency matrix A_s
    D_s = np.zeros((n, n))  # Degree matrix (symmetrized)
    A_s = np.zeros((n, n))  # Adjacency matrix (symmetrized)
    
    # Map nodes to indices
    node_list = sorted(G.nodes)
    node_index = {node: i for i, node in enumerate(node_list)}
    
    # Fill the adjacency matrix and degree matrix
    for u, v in G.edges():
        i = node_index[u]
        j = node_index[v]
        
        # Adjacency matrix A_s (symmetrized)
        A_s[i, j] += 1  # Default weight 1 if not provided
        A_s[j, i] += 1  # Symmetrize
        
        # Degree matrix D_s
        D_s[i, i] += 1
        D_s[j, j] += 1
    
    # Compute Theta_q matrix
    Theta_q = np.zeros_like(A_s, dtype=complex)
    for u, v in G.edges():
        i = node_index[u]
        j = node_index[v]
        
        # The potential term Theta_q
        if A_s[i, j] != A_s[j, i]:  # Only apply for directed edges
            Theta_q[i, j] = 2 * np.pi * q * (A_s[i, j] - A_s[j, i])
    
    # Compute the magnetic Laplacian L_q = D_s - A_s * exp(i * Theta_q)
    L_q = D_s - A_s * np.exp(1j * Theta_q)
    
    return L_q

def magnetic_laplacian(G, q):
    """
    Compute the magnetic Laplacian for a directed graph `g`.
    
    Parameters:
    - g: networkx.DiGraph
        A directed graph.
    - q: float
        Magnetic flux parameter (normalized, typically in [0, 1]).

    Returns:
    - Magnetic Laplacian as a sparse matrix.
    """
    def exp_theta_i(A, q):
        return np.exp(2 * np.pi * q * 1j * (A - A.T))
    nodelist = sorted(G.nodes())
    A_directed = nx.to_pandas_adjacency(G,nodelist=nodelist).to_numpy()
    A_symmetric = A_directed + A_directed.T
    assert np.allclose(A_symmetric, A_symmetric.T)
    D_s = np.diag(np.sum(A_symmetric, axis=1))
    asymmetric_element = exp_theta_i(A_directed, q)
    laplacian = D_s - np.multiply(asymmetric_element, A_symmetric)
    assert np.allclose(laplacian, laplacian.conj().T)
    return laplacian


def magL_eigenvectors(MagL):
    _, eig_vecs = np.linalg.eigh(MagL)
    return eig_vecs

#TODO: change 1 to 1/1000
def magnetic_laplacian_eigenvectors(g, max_seq_len, q=magentic_laplace_encodings_q):
    MagL = magnetic_laplacian(G=g, q = q)
    vec = magL_eigenvectors(MagL)
    vec = pad_and_concat_eigenvectors(vec, max_seq_len)
    return vec


def pad_and_concat_eigenvectors(eigenvectors, max_seq_len):
    """
    Pad and concatenate the real and imaginary parts of eigenvectors.

    Parameters:
    eigenvectors (list of list of complex): A square matrix (m by m) of complex-valued eigenvectors.
    max_seq_len (int): The maximum sequence length.

    Returns:
    np.ndarray: A vector with real parts from 0 to max_seq_len - 1 and imaginary parts from max_seq_len to max_seq_len * 2 - 1.
    """
    m = eigenvectors.shape[1]

    # Initialize the padded vector with complex zeros
    padded_vector = np.zeros(
        (
            eigenvectors.shape[0],
            max_seq_len * 2,
        ),
        dtype=float,
    )

    for i in range(len(eigenvectors)):
        real_part = np.array([e.real for e in eigenvectors[i]])
        imag_part = np.array([e.imag for e in eigenvectors[i]])

        # Place real part in the first half
        padded_vector[i][:m] = real_part
        # Place imaginary part in the second half
        padded_vector[i][max_seq_len : max_seq_len + m] = imag_part
    return padded_vector.astype(np.float32)
