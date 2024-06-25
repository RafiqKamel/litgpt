import numpy as np
import networkx as nx


def magnetic_laplacian(g, q=0.25, tolerance=1e-10):
    def exp_theta_i(A, q=0.25):
        return np.exp(2 * np.pi * q * 1j * (A - A.T))

    A_symmetric = nx.adjacency_matrix(g.to_undirected()).toarray()
    A_directed = nx.adjacency_matrix(g).toarray()
    D_s = np.diag(np.sum(A_symmetric, axis=1))
    asymmetric_element = exp_theta_i(A_directed, q)
    laplacian = D_s - np.multiply(asymmetric_element, A_symmetric)

    # Check if the real part is very close to zero and set it to zero
    laplacian_real = np.real(laplacian)
    laplacian_real[np.abs(laplacian_real) < tolerance] = 0
    # Reconstruct the complex array with the updated real part and original imaginary part
    laplacian = laplacian_real + 1j * laplacian.imag
    return laplacian


def convert_complex_to_real(eig_vecs):
    """
    Convert complex eigenvectors to real by concatenating the real and imaginary parts.

    Parameters:
    eig_vecs (numpy.ndarray): Complex eigenvectors.

    Returns:
    numpy.ndarray: Real-valued vectors.
    """
    real_part = np.real(eig_vecs)
    imag_part = np.imag(eig_vecs)
    real_valued_vecs = np.concatenate((real_part, imag_part), axis=-1)
    return real_valued_vecs


def magL_eigenvectors(MagL):
    _, eig_vecs = np.linalg.eig(MagL)
    return convert_complex_to_real(eig_vecs)


def magnetic_laplacian_eigenvectors(g, max_seq_len, q=0.25, tolerance=1e-10):
    MagL = magnetic_laplacian(g, q, tolerance)
    vec = magL_eigenvectors(MagL)
    return pad_and_concat_eigenvectors(vec, max_seq_len)


def pad_and_concat_eigenvectors(eigenvectors, max_seq_len):
    """
    Pad and concatenate the real and imaginary parts of eigenvectors.

    Parameters:
    eigenvectors (list of list of complex): A square matrix (m by m) of complex-valued eigenvectors.
    max_seq_len (int): The maximum sequence length.

    Returns:
    np.ndarray: A vector with real parts from 0 to max_seq_len - 1 and imaginary parts from max_seq_len to max_seq_len * 2 - 1.
    """
    m = len(eigenvectors)

    # Initialize the padded vector with complex zeros
    padded_vector = np.zeros(
        (
            max_seq_len,
            max_seq_len * 2,
        ),
        dtype=complex,
    )

    for i in range(m):
        real_part = [e.real for e in eigenvectors[i]]
        imag_part = [e.imag for e in eigenvectors[i]]

        # Place real part in the first half
        padded_vector[i][:m] = real_part
        # Place imaginary part in the second half
        padded_vector[i][max_seq_len : max_seq_len + m] = imag_part

    return padded_vector.astype(np.float32)
