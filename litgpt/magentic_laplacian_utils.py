import numpy as np
import networkx as nx
from litgpt.positional_encodings_config import magentic_laplace_encodings_q


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
    A_directed = nx.to_pandas_adjacency(G, nodelist=nodelist).to_numpy()
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


def magnetic_laplacian_eigenvectors(g, max_seq_len, num_of_eigenvecs,q=magentic_laplace_encodings_q):
    MagL = magnetic_laplacian(G=g, q=q)
    vec = magL_eigenvectors(MagL)
    if num_of_eigenvecs > 0 and num_of_eigenvecs < vec.shape[1]:
        vec = vec[:, :num_of_eigenvecs]
    vec = pad_and_concat_eigenvectors(vec, max_seq_len = max_seq_len if num_of_eigenvecs == -1 else num_of_eigenvecs) 
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
    return padded_vector
