# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# License: 3-clause BSD.

# TODO see if we can leverage the ArrayBuilder from sklearn.utils here; that
# would probably need a .pxd for an extra speed boost.

from libc.stdlib cimport abs
cimport numpy as np
from cpython cimport array

#from ..utils import murmurhash3_32
from sklearn.utils.murmurhash cimport murmurhash3_bytes_s32

def resize_arrays(arrays, size):
    for arr in arrays:
        array.resize(arr, size)

def transform(raw_X, Py_ssize_t n_features):
    """Guts of FeatureHasher.transform.

    Returns
    -------
    n_samples : integer
    i_ind, j_ind, values : arrays
        For constructing a scipy.sparse.coo_matrix.

    """
    cdef np.int32_t h
    cdef Py_ssize_t i
    cdef int value

    cdef int size = 10000
    cdef array.array i_ind = array.array('i')
    cdef array.array j_ind = array.array('i')
    cdef array.array values = array.array('i')
    resize_arrays((i_ind, j_ind, values), size)

    cdef int idx = -1
    i = -1
    for i, x in enumerate(raw_X):
        for f in x:
            idx += 1
            if isinstance(f, unicode):
                f = f.encode("utf-8")
            # Need explicit type check because Murmurhash does not propagate
            # all exceptions. Add "except *" there?
            elif not isinstance(f, bytes):
                raise TypeError("feature names must be strings")
            h = murmurhash3_bytes_s32(f, 0)

            # grow arrays as needed
            if idx >= size:
                size *= 2
                resize_arrays((i_ind, j_ind, values), size)

            i_ind[idx] = i
            j_ind[idx] = abs(h) % n_features
            value = (h >= 0) * 2 - 1
            values[idx] = value

    resize_arrays((i_ind, j_ind, values), idx + 1)
    return i + 1, i_ind, j_ind, values
