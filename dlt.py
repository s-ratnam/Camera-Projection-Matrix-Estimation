"""Functions to compute projection matrix using DLT."""
import numpy as np


def generate_homogenous_system(pts2d: np.ndarray,
                               pts3d: np.ndarray) -> np.ndarray:
    """Generate a matrix A s.t. Ap=0. Follow the convention in the jupyter
    notebook and process the rows in the same order as the input, i.e. the
    0th row of input should go to 0^th and 1^st row in output.

    Note: remember that input is not in homogenous coordinates. Hence you need 
    to append w=1 for all 2D inputs, and the same thing for 3D inputs.

    Args:
        pts2d: numpy array of shape nx2, with the 2D measurements.
        pts3d: numpy array of shape nx3, with the actual 3D coordinates.
    """

    n = pts2d.shape[0]

    A = np.zeros((2*n, 12))

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    ones = np.ones((n, 1))
    # append w = 1 for 2d and 3d inputs 
    pts2d = np.hstack((pts2d, ones))
    pts3d = np.hstack((pts3d, ones))
    # generate 2n x 12 matrix A
    for i in range(0, 2*n, 2):
        j = i//2
        # First row 
        A[i, 0:4] = 0.
        A[i, 4:8] = -pts2d[j, 2]*pts3d[j,:]
        A[i, 8:12] = pts2d[j, 1]*pts3d[j,:]
        # second row 
        A[i+1, 0:4] = pts2d[j, 2]*pts3d[j,:]
        A[i+1, 4:8] = 0.
        A[i+1, 8:12] = -pts2d[j, 0]*pts3d[j,:]


    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return A


def get_eigenvector_with_smallest_eigenvector(A: np.ndarray) -> np.ndarray:
    """Get the unit normalized eigenvector corresponding to the minimum 
    eigenvalue of A.

    Hints: you may want to use np.linalg.svd.

    Note: please work out carefully if you need to access a row or a column to 
    get the required eigenvector from the SVD results.

    Args:
        A: the numpy array of shape pxq, for which the eigenvector is to be computed.

    Returns:
        eigenvec: the numpy array of shape (q,), the computed eigenvector of the minimum eigenvalue, 
        (note: just a single eigenvector).
    """

    eigenvec = np.empty(A.shape[0])

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    # singular value decomposition 
    u, s, vh = np.linalg.svd(A)
    # last row of vh of returned component from SVD
    eigenvec = vh[-1,:]
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return eigenvec


def estimate_projection_matrix_dlt(pts2d: np.ndarray,
                                   pts3d: np.ndarray) -> np.ndarray:
    """Estimate the projection matrix using DLT.

    Note: 
    1. Scale your projection matrix estimate such that the last entry is 1.

    Args:
        pts2d: numpy array of shape nx2, with the 2D measurements.
        pts3d: numpy array of shape nx3, with the actual 3D coordinates.

    Returns:
        estimated projection matrix of shape 3x4.
    """

    assert pts2d.shape[0] >= 6

    A = generate_homogenous_system(pts2d, pts3d)

    eigenvec = get_eigenvector_with_smallest_eigenvector(A)

    P = eigenvec.reshape(3, 4)

    # scaling P so that the 12th entry is 1
    P = P/P[2, 3]

    return P
