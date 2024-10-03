"""DFC."""

import joblib
import numpy as np
from pyspa import SPAReader
from tqdm import tqdm
from unipy import linalg

from full_profile.utilities import tqdm_joblib


class DFC:
    def __init__(self, reader, selection, svt, C, save_path: str = ""):
        self.reader = reader
        self.selection = selection
        self.svt = svt
        self.C = C
        self.A = []
        self.save_path = save_path
        self.Uc = None
        self.Mc = None
        self.partition = {}

    def divide(self, bin_width: int = 100) -> None:
        # Create Subsampling
        vect = np.arange(len(self.selection))
        np.random.shuffle(vect)
        n_i = int(np.ceil(len(self.selection) / bin_width))
        for i in range(n_i - 1):
            self.partition[i] = vect[i * bin_width : (i + 1) * bin_width]

        self.partition[n_i - 1] = vect[(n_i - 1) * bin_width :]
        print(len(self.partition), "times", self.reader.n_mz_bins, "x", bin_width)

        return None

    def factor(self, n_jobs: int = 10):
        with tqdm_joblib(tqdm(desc="Factor", total=len(self.partition))):
            self.A = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self._svt)(self.selection[part[1]]) for part in self.partition.items()
            )
        return None

    def combine(self, p: int = 5, rank_oversample: int = 0) -> None:
        # Projection
        rank = []
        for aa in self.A:
            rank.append(aa._svp)

        rank = np.array(rank)
        median_rank = int(np.median(rank))
        print("Median Rank", median_rank)

        G = np.random.randn(len(self.selection), median_rank + rank_oversample + p)
        Cc = self.block_multiply(G, False)  # M@G
        Dc = self.block_multiply(Cc, True)  # M.T@M@G
        Ec = self.block_multiply(Dc, False)  # M@M.T@M@G
        Fc = self.block_multiply(Ec, True)  # M.T@M@M.T@M@G
        Gc = self.block_multiply(Fc, False)  # M@M.T@M@M.T@M@G
        Q, _, _ = linalg.svd(Gc)

        self.Uc = Q[:, : median_rank + rank_oversample]
        # Projection
        self.Mc = np.zeros((self.Uc.shape[1], len(selection)), dtype="float32")
        i = 0

        for aa in A:
            U = np.array(aa[0])
            S = np.array(aa[1])
            Vt = np.array(aa[2])
            self.Mc[:, self.partition[i]] = (self.Uc.T @ (U * S)) @ Vt

            i += 1

        return None

    def block_multiply(self, B, transpose):  # Parallelize this process
        i = 0
        if transpose:
            C = np.zeros((len(self.selection), B.shape[1]))
        else:
            C = np.zeros((self.A[0][0].shape[0], B.shape[1]))

        for aa in self.A:  # use enumerate here
            if transpose:
                U = np.array(aa[2]).T
                S = np.array(aa[1])
                Vt = np.array(aa[0]).T
                C[self.partition[i], :] = (U * S) @ (Vt @ B)
            else:
                U = np.array(aa[0])
                S = np.array(aa[1])
                Vt = np.array(aa[2])
                C += (U * S) @ (Vt @ B[self.partition[i], :])

            i += 1
        return C

    def _read_in(self, path, selection):
        # instantiate the reader
        reader = SPAReader(path)
        f = reader[0]
        iter_var = reader.framelist
        out = np.zeros((reader.n_mz_bins, len(selection)), dtype=f.dtype)
        C_sp = ss.csr_matrix(
            1 / (self.C[iter_var[selection]] / np.median(self.C[iter_var[selection]])), dtype="float32"
        )

        for k, i in enumerate(iter_var[selection]):
            f = reader[i]._init_csc()
            out[f.indices, k] = f.data

        B = ss.csc_matrix(out, dtype="float32").multiply(C_sp)  # double check this
        return B

    def _svt(self, selection):
        a = self._read_in(selection, self.C)  # Read In Data in A
        inst = self.svt
        inst.run(a)
        obj = [
            inst._b[0].todense(),
            inst._b[1].todense(),
            inst._b[2].todense(),
            selection,
        ]  # Export rank, U, S, Vt and that's it.

        return obj

    def _save_intermediate(self, data):
        np.savez(self.save_path + "/" + str(np.random.randint(0, 10000000, 1)) + ".npz", data=data)
        return None

    def combine_from_files(self) -> None:  # recombine from files in here!
        return None
