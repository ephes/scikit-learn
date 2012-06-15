import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin


class FeatureHasher(BaseEstimator, TransformerMixin):
    """Implements feature hashing, aka the hashing trick.

    This class turns sequences of symbolic feature names into scipy.sparse
    matrices, using a hash function to compute the matrix column corresponding
    to a name.

    The types of "names" supported depend on the hash function being used;
    use Python string to "play safe".

    This class is a low-memory alternative to DictVectorizer and
    CountVectorizer, intended for large-scale (online) learning and situations
    where memory is tight, e.g. when running prediction code on small
    computers.

    Parameters
    ----------
    n_features : integer
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.
    dtype : NumPy type, optional
        The type of feature values. Passed to scipy.sparse matrix constructors
        as the dtype argument. Do not set this to bool, np.boolean or any
        unsigned integer type.
    non_negative : boolean, optional
        Whether output matrices should contain non-negative values only;
        effectively calls abs on the matrix prior to returning it.
        When True, output values will be multinomially distributed.
        When False, output values will be normally distributed (Gaussian) with
        mean 0, assuming a good hash function.

    """

    def __init__(self, n_features, dtype=np.float64, non_negative=False):
        if not isinstance(n_features, (int, np.integer)):
            raise TypeError("n_features must be integral, got %r (%s)"
                            % (n_features, type(n_features)))
        elif n_features < 1:
            raise ValueError("invalid number of features (%d)" % n_features)

        # TODO support murmurhash and arbitrary callables
        self._hash = np.vectorize(hash)
        self.dtype = dtype
        self.n_features = n_features
        self.non_negative = non_negative

    def fit(self, X=None, y=None):
        """No-op.

        This method doesn't do anything. It exists purely for compatibility
        with the scikit-learn transformer API.

        Returns
        -------
        self : FeatureHasher

        """
        return self

    def transform(self, raw_X, y=None):
        """Transform a sequence of instances to a scipy.sparse matrix.

        Parameters
        ----------
        raw_X : iterable over iterable over feature names, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.
            See the class docstring for allowable feature name types.
        y : (ignored)

        Returns
        -------
        X : scipy.sparse matrix, shape = (n_samples, self.n_features)
            Feature matrix, for use with estimators or further transformers.

        """
        i_ind = []
        j_ind = []
        values = []

        i = -1
        for i, x in enumerate(raw_X):
            hashes = self._hash(x)
            signs = (hashes & 1) * 2 - 1
            hashes >>= 1

            for j, h in enumerate(hashes):
                i_ind.append(i)
                j_ind.append(h % self.n_features)
                values.append(signs[j])

        if i < 0:
            raise ValueError("cannot vectorize empty sequence")

        X = sp.coo_matrix((values, (i_ind, j_ind)), dtype=self.dtype,
                          shape=(i + 1, self.n_features))
        if self.non_negative:
            X = X.tocsr()
            np.abs(X.data, X.data)
        return X
