"""Scores."""

import typing as ty

import numpy as np
import scipy.sparse as ss
from joblib import Parallel, delayed
from unipy import linalg, to_arraytype


class Scores:
    """Scores."""

    def __init__(
        self,
        reference: ty.Any,
        factored: ty.Any,
    ):
        self.reference = reference
        self.factored = factored
        pass

    def densify(self, s):
        return np.array(s.todense())

    def score_peak_selection(self, nr_peaks: int = 100) -> ty.Any:
        """Calculate score for peak selection."""
        total_ion_sum = np.array(self.reference.sum(axis=1))
        sort = np.argsort(total_ion_sum.T)
        peak_selection = (sort[0])[-nr_peaks:]

        selection1 = self.densify(self.reference[peak_selection, :])
        selection2 = self.factored[0][peak_selection, :] @ self.factored[1]
        selection1[selection1 < 0.1] = np.nan

        score = np.linalg.norm(selection1 - selection2, 2, axis=0) / np.linalg.norm(selection1, 2, axis=0) * 100
        return score

    def _peak_match(self, i, nr_peaks):
        spectrum = self.densify(self.reference[:, [i]])[:, 0]
        compare = to_arraytype(self.factored[0] @ self.factored[1][:, [i]], "numpy")[:, 0]
        idx = np.argpartition(spectrum, -nr_peaks)[-nr_peaks:]
        score = np.linalg.norm(compare[idx] - spectrum[idx], 2) / np.linalg.norm(spectrum[idx], 2) * 100

        return (i, score)

    def peak_match_per_spectrum(
        self,
        nr_peaks: int = 100,
        n_jobs: int = 2,
    ) -> ty.Any:
        """Calculate peak match per spectrum score."""
        score = np.zeros(self.reference.shape[1])

        A = Parallel(n_jobs=n_jobs, backend="threading")(  # max 51/52
            delayed(self._peak_match)(i, nr_peaks) for i in range(self.reference.shape[1])
        )

        for a in A:
            score[a[0]] = a[1]

        return np.array(score)

    def compression_scores(self, verbose: bool = False) -> tuple:
        """Calculate compression scores."""
        raw_dense_size = np.prod(self.reference.shape) * 4 / (10**9)
        raw_sparse_size = (
            self.reference.data.nbytes + self.reference.indices.nbytes + self.reference.indptr.nbytes
        ) / (10**9)
        compressed_size = (np.prod(self.factored[0].shape) + np.prod(self.factored[1].shape)) * 4 / (10**9)
        comp1 = raw_dense_size / compressed_size
        comp2 = raw_sparse_size / compressed_size
        if verbose:
            print("Dense vs. Compressed:", comp1, "\nSparse vs. Compressed:", comp2)
        return (comp1, comp2)

    def reconstruction_score_old(self, dense: bool = False) -> float:  # Something wrong in here
        a_fro = linalg.norm(self.reference.data, 2)
        x_sparse = ss.csc_matrix((np.zeros_like(self.reference.data), self.reference.indices, self.reference.indptr))
        u = self.factored[0]
        v = self.factored[1]

        if dense:
            uv = u @ v

        for i in range(self.reference.shape[1]):
            col = i
            row = x_sparse.indices[x_sparse.indptr[i] : x_sparse.indptr[i + 1]]
            x_sparse.data[x_sparse.indptr[i] : x_sparse.indptr[i + 1]] = (
                uv[row, col] if dense else u[row, :] @ v[:, col]
            )

        return linalg.norm((self.reference.data - x_sparse.data), 2) / a_fro

    def _reconstruction(self, i):
        col = i
        row = self.reference.indices[self.reference.indptr[i] : self.reference.indptr[i + 1]]
        # self.x_sparse.data[self.x_sparse.indptr[col]:self.x_sparse.indptr[col+1]] = self.factored[0][row, :]@self.factored[1][:, col]
        # data = self.factored[0][row, :]@self.factored[1][:, col]

        return self.factored[0][row, :] @ self.factored[1][:, col]

    def reconstruction_score(self, n_jobs: int = 2) -> float:
        a_fro = linalg.norm(self.reference.data, 2)

        A = Parallel(n_jobs=n_jobs)(  # max 51/52
            delayed(self._reconstruction)(i) for i in range(self.reference.shape[1])
        )

        x_sparse_data = np.concatenate(A, dtype="float32")
        # for a in A:
        #     col = a[0]
        #     row = self.x_sparse.indices[self.x_sparse.indptr[col]:self.x_sparse.indptr[col+1]]
        #     self.x_sparse.data[self.x_sparse.indptr[col]:self.x_sparse.indptr[col+1]] = a[1]

        return linalg.norm((self.reference.data - x_sparse_data), 2) / a_fro

    def _reconstruction_test(self, i):
        col = i
        row = self.reference.indices[self.reference.indptr[i] : self.reference.indptr[i + 1]]
        return (
            self.factored[0][row, :] @ self.factored[1][:, col]
            - self.reference.data[self.reference.indptr[i] : self.reference.indptr[i + 1]]
        )

    def reconstruction_score_test(self, n_jobs: int = 2) -> float:
        a_fro = linalg.norm(self.reference.data, 2)

        A = Parallel(n_jobs=n_jobs)(  # max 51/52
            delayed(self._reconstruction_test)(i) for i in range(self.reference.shape[1])
        )

        # for a in A:
        #     col = a[0]
        #     row = self.x_sparse.indices[self.x_sparse.indptr[col]:self.x_sparse.indptr[col+1]]
        #     self.x_sparse.data[self.x_sparse.indptr[col]:self.x_sparse.indptr[col+1]] = a[1]

        return linalg.norm((np.concatenate(A, dtype="float32")), 2) / a_fro

    def reconstruction_score_peak_picking(self, nr_peaks: int) -> float:
        a_fro = linalg.norm(self.reference.data, 2)
        total_ion_sum = np.array(self.reference.sum(axis=1))
        sort = np.argsort(total_ion_sum.T)
        peak_selection = (sort[0])[:-nr_peaks]
        B_subsampled = self.reference[peak_selection, :]
        return linalg.norm((B_subsampled.data), 2) / a_fro

    def reconstruction_score_peak_picking_match(self) -> float:
        compressed_size = np.prod(self.factored[0].shape) + np.prod(self.factored[1].shape)
        return self.reconstruction_score_peak_picking(int(compressed_size / self.reference.shape[1])), int(
            compressed_size / self.reference.shape[1]
        )


# for back compatibility with previous versions
scores = Scores
