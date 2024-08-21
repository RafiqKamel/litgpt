import numpy as np
import networkx as nx
from litgpt.positional_encodings_config import magentic_laplace_encodings_q

def magnetic_laplacian(g, tolerance, q=magentic_laplace_encodings_q):
    def exp_theta_i(A, q):
        return np.exp(2 * np.pi * q * 1j * (A - A.T))

    A_symmetric = nx.adjacency_matrix(g.to_undirected()).toarray()
    A_directed = nx.adjacency_matrix(g).toarray()
    D_s = np.diag(np.sum(A_symmetric, axis=1))
    asymmetric_element = exp_theta_i(A_directed, q)
    laplacian = D_s - np.multiply(asymmetric_element, A_symmetric)

    # Check if the real part is very close to zero and set it to zero
    laplacian_real = np.real(laplacian)
    laplacian_real[np.abs(laplacian_real) < tolerance] = 0

    # Check if the imaginary part is very close to zero and set it to zero
    laplacian_img = np.imag(laplacian)
    laplacian_img[np.abs(laplacian_img) < tolerance] = 0  
      
    # Reconstruct the complex array with the updated real part and original imaginary part
    laplacian = laplacian_real + 1j * laplacian_img
    return laplacian





def magL_eigenvectors(MagL):
    _, eig_vecs = np.linalg.eig(MagL)
    return eig_vecs


def magnetic_laplacian_eigenvectors(g, max_seq_len, q=0.25, tolerance=1e-5):
    MagL = magnetic_laplacian(g=g, q = q, tolerance= tolerance)
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
