# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# License: 3-clause BSD.

# The handling of hash functions is a bit tricky. We assume that hash functions
# return something convertible to Py_ssize_t, which seems to be true for the
# built-in function hash(), even though there seems to be no guarantee to this
# effect in any of
# * http://docs.python.org/library/functions.html#hash
# * http://docs.python.org/dev/library/functions.html#hash
# * http://docs.python.org/reference/datamodel.html#object.__hash__
# * http://docs.python.org/dev/reference/datamodel.html#object.__hash__
#
# long (or libc.stdint.int32_t) is too small with 64-bit Python:
#
#   >>> hash(2 ** 34) == 2 ** 34
#   True
#
# On the bright side, this means we're forward-compatible with a 64-bit clean
# scipy.sparse, if that's ever implemented.

# TODO see if we can leverage the ArrayBuilder from sklearn.utils here; that
# would probably need a .pxd for an extra speed boost.

cimport numpy as np


def transform(raw_X, hashfn, Py_ssize_t n_features):
    """Guts of FeatureHasher.transform.

    Returns
    -------
    n_samples : integer
    i_ind, j_ind, values : lists
        For constructing a scipy.sparse.coo_matrix.

    """
    cdef Py_ssize_t h
    cdef Py_ssize_t i
    cdef int sign

    i_ind = []
    j_ind = []
    values = []

    i = -1
    for i, x in enumerate(raw_X):
        for f in x:
            h = hashfn(f)
            sign = (h & 1) * 2 - 1
            h = <uint32_t>(h)
            h >>= 1

            i_ind.append(i)
            j_ind.append(h % n_features)
            values.append(sign)

    return i + 1, i_ind, j_ind, values
