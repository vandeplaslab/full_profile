"""SVT."""

import time
import typing as ty

import cupyx.scipy.sparse as cpss
import numpy as np
import scipy.sparse as ss
import unipy.types as uty
from unipy import find_package, linalg, matmul, multiply, to_arraytype, zeros


class SVT:
    def __init__(
        self,
        maxiter: int = 10,
        tau_factor: float = 0.05,
        delta: float = 1,
        sv_factor: float = 0.05,
        dense: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> ty.NoReturn:
        """
        Initialize the SVT object.

        Parameters
        ----------
        - maxiter: int
            Maximum number of iterations.
        - tau_factor: float
            Multiplicative factor for the threshold parameter tau.
        - delta: float
            Step size for the update of the augmented Lagrangian variable Y.

        Returns
        -------
            None
        """
        self.maxiter = maxiter
        self.tau_factor = tau_factor
        self.delta = delta
        self.sv_factor = sv_factor
        self.dense = dense
        self.verbose = verbose
        self.kwargs = kwargs
        self.tic = 0

    @property
    def b(self) -> uty.Array:
        """Reconstruct b (low-rank component) from svd components.

        Returns
        -------
            array : reconstructed low-rank component

        """
        return matmul(multiply(self._b[0].todense(), self._b[1].todense()), self._b[2].todense())

    @b.setter
    def b(self, a: tuple[uty.Array]) -> None:
        """Sets b from tuple.

        Parameters
        ----------
        a : tuple[array]
            Tuple containing arrays of svd to set b

        """
        # print(type(self._b[1]))
        self._b[0] = a[0]
        self._b[1] = a[1]
        self._b[2] = a[2]
        return None

    def _initialize_b(self) -> None:
        """Initialize b."""
        self._b = [0, 0, 0]
        self._b[0] = zeros((self._m, 1), atype=self.atype, dtype=self.datatype)
        self._b[1] = zeros((1, 1), atype=self.atype, dtype=self.datatype)
        self._b[2] = zeros((1, self._n), atype=self.atype, dtype=self.datatype)
        return None

    def _datatype(self, dtype: str) -> str:
        """Returns datatype and promotes if required."""
        dtype = dtype if dtype not in ["int8", "int16", "int32", "int64"] else "float32"
        return dtype

    def initialize(self, a: uty.Array) -> None:
        """
        Initialize the SVT algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        time.time()
        self.a = a
        self.datatype = self._datatype(str(a.dtype))
        self._m, self._n = self.a.shape
        # Initialize variables
        self.atype = (
            "scipy.sparse"
            if find_package(self.a)[1] == "numpy" or find_package(self.a)[1] == "scipy.sparse"
            else "cupyx.scipy.sparse"
        )

        self.btype = (
            "numpy" if find_package(self.a)[1] == "numpy" or find_package(self.a)[1] == "scipy.sparse" else "cupy"
        )

        self._ss = ss if find_package(self.a)[1] == "numpy" or find_package(self.a)[1] == "scipy.sparse" else cpss

        self._initialize_b()

        self.y = self._ss.csc_matrix(
            (np.zeros_like(self.a.data), self.a.indices, self.a.indptr), shape=(self._m, self._n)
        )
        # zeros((self._m, self._n), atype=self.atype, dtype=self.datatype)
        self.x_sparse = self._ss.csc_matrix(
            (np.zeros_like(self.a.data), self.a.indices, self.a.indptr), shape=(self._m, self._n)
        )
        # zeros((self._m, self._n), atype=self.atype, dtype=self.datatype)
        self.tau = self.tau_factor * max(self.a.shape)
        self.a_fro = linalg.norm(self.a.data, 2)
        self.k0 = 1
        self.k = 0
        self._sv = 10  # set sv
        # print('Initialization', time.time() - tic)
        return None

    def stopping_criterion(self) -> bool:
        """
        Check the stopping criterion.

        Parameters
        ----------
        None

        Returns
        -------
        - bool
            True if the stopping criterion is met, False otherwise.
        """
        crit = linalg.norm((self.a.data - self.x_sparse.data), 2) / self.a_fro
        if crit < 0.05:
            if self.verbose:
                print(
                    "Iteration "
                    + str(self.k)
                    + " - Crit = "
                    + str(crit)
                    + " - Rank = "
                    + str(self._svp)
                    + " - Time = "
                    + str(time.time() - self.tic)
                )
            return True
        else:
            if self.verbose:
                print(
                    "Iteration "
                    + str(self.k)
                    + " - Crit = "
                    + str(crit)
                    + " - Rank = "
                    + str(self._svp)
                    + " - Time = "
                    + str(time.time() - self.tic)
                )
                self.tic = time.time()
            return False

    def soft_thresholding(self) -> None:
        """Update b by performing truncated svd."""
        # print('\n Soft Threshold\n')
        time.time()
        if "method" in self.kwargs:
            sv = self._sv + 10 if linalg.requires_sv(self.kwargs["method"]) is True else 0
            if self.k > 0:
                if self.kwargs["method"] == "sparse_propack":
                    self.kwargs.update(
                        {"v0": (self._b[2][[0], :].todense())[0, :] if sum(self._b[1].data) > 1e-2 else None}
                    )
                else:
                    if self._b[0].shape[0] < self._b[2].shape[1]:
                        self.kwargs.update(
                            {"v0": (self._b[0][:, [0]].todense())[:, 0] if sum(self._b[1].data) > 1e-2 else None}
                        )
                    else:
                        self.kwargs.update(
                            {"v0": (self._b[2][[0], :].todense())[0, :] if sum(self._b[1].data) > 1e-2 else None}
                        )

        else:
            sv = 0

        self.kwargs.update({"sv": sv})

        self.b = linalg.svd(self.y, self.kwargs)
        # print(self._b[1].data)#, self._b[1])
        self._svp = int(sum(self._b[1].data > self.tau))
        if self._svp != 0:
            self._threshold(self._svp, self.tau)
        else:  # if svp is 0, just reset the svd result
            self._initialize_b()
        # print('soft_thresholding', time.time() - tic)
        return None

    def _threshold(self, num: int = 1, value: float = 0.0) -> None:
        """Threshold b to certain value.

        Parameters
        ----------
        num: int
            Truncate to this number of values

        value : float
            Threshold to this value

        """
        self._b[0] = self._b[0][:, :num]
        self._b[1] = self._b[1][[0], :num]
        self._b[1].data -= value
        self._b[2] = self._b[2][:num, :]

        return None

    def _update_sv(self) -> None:
        """Update (predict) the number of singular values to be calculated by svd."""
        n = min(self._m, self._n)
        if self._svp < self._sv:
            self._sv = int(min(self._svp + 1, n))
        else:
            self._sv = int(min(self._svp + 10, n))

    def update_x_sparse(self) -> None:
        time.time()
        # self.x_sparse.data = np.einsum(
        #     'ij,jj,ji->i',
        #     self._b[0][self.a.row,:],
        #     self._b[1],
        #     self._b[2][:, self.a.col],
        # optimize='greedy')

        # self.x_sparse.data = sum((to_arraytype(self._b[0], self.btype)[self.x_sparse.row,:]*to_arraytype(self._b[1], self.btype))*to_arraytype((self._b[2]).T, self.btype)[self.x_sparse.col,:], axis=1)
        u = to_arraytype(self._b[0], self.btype) * to_arraytype(self._b[1], self.btype)
        v = to_arraytype(self._b[2], self.btype)

        if self.dense:
            uv = u @ v

        for i in range(self.a.shape[1]):
            col = i
            row = self.x_sparse.indices[self.x_sparse.indptr[i] : self.x_sparse.indptr[i + 1]]
            self.x_sparse.data[self.x_sparse.indptr[i] : self.x_sparse.indptr[i + 1]] = (
                uv[row, col] if self.dense else u[row, :] @ v[:, col]
            )

        # self.x_sparse.data = sum((to_arraytype(self._b[0], self.btype)[self.x_sparse.row,:]*to_arraytype(self._b[1], self.btype))*to_arraytype((self._b[2]).T, self.btype)[self.x_sparse.col,:], axis=1)

        # print('update_x_sparse', time.time() - tic)
        return None

    def update_y(self):
        """
        Update the augmented Lagrangian variable Y.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        time.time()
        self.y.data += self.delta * (self.a.data - self.x_sparse.data)
        # print('Update Y', time.time() - tic)

    def update_delta(self):
        """
        Update the step size delta if needed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.delta *= 1
        return self.delta

    def run(self, a: uty.Array) -> None:
        """
        Execute the SVT algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.initialize(a)
        self.tic = time.time()

        while self.k < self.maxiter:
            self.soft_thresholding()
            self._update_sv()
            self.update_x_sparse()

            if self.stopping_criterion():
                break

            self.tic = time.time()
            self.update_y()
            self.update_delta()

            self.k += 1

        return None


# for back compatibility with previous versions
svt = SVT
