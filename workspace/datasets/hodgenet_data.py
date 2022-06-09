import numpy as np
import scipy
import scipy.io


def flip(f, ev, ef, bdry=False):
    """Compute vertices of flipped edges."""
    # glue together the triangle vertices adjacent to each edge
    duplicate = f[ef].reshape(ef.shape[0], -1)
    duplicate[(duplicate == ev[:, 0, None])
              | (duplicate == ev[:, 1, None])] = -1  # remove edge vertices

    # find the two remaining verts (not -1) in an orientation-preserving way
    idxs = (-duplicate).argsort(1)[:, :(1 if bdry else 2)]
    idxs.sort(1)  # preserve orientation by doing them in index order
    result = np.take_along_axis(duplicate, idxs, axis=1)

    return result


def random_scale_matrix(scale):
    """Generate a random 3D anisotropic scaling matrix."""
    return np.diag(1 + (np.random.rand(3) * 2 - 1) * scale)


def d01(v, e):
    """Compute d01 operator from 0-froms to 1-forms."""
    row = np.tile(np.arange(e.shape[0]), 2)
    col = e.T.flatten()
    data = np.concatenate([np.ones(e.shape[0]), -np.ones(e.shape[0])], axis=0)
    d = scipy.sparse.csr_matrix(
        (data, (row, col)), dtype=np.double, shape=(e.shape[0], v.shape[0]))
    return d

