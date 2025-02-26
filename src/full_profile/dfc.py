"""Module for the DFC class which performs subsampling, factorization, and recombination of data.

This module implements the DFC (Divide-Factor-Combine) class that:
    - Divides a dataset into random partitions.
    - Applies singular value thresholding (SVT) to each partition in parallel.
    - Combines the factors using block matrix multiplications.
"""

import joblib
import numpy as np
import scipy.sparse as ss  # Added missing import for sparse matrix operations.
from pyspa import SPAReader
from tqdm import tqdm
from unipy import linalg

from full_profile.utilities import tqdm_joblib


class DFC:
    """Divide-Factor-Combine (DFC) class for data processing.

    Attributes:
        reader: An instance of SPAReader for reading data.
        selection: An array or list specifying the data indices to process.
        svt: An instance responsible for performing singular value thresholding.
        C: A parameter (or array) used in data scaling.
        save_path (str): Optional directory path to save intermediate results.
        A (list): List to store factorization results for each partition.
        Uc (np.ndarray): Combined left singular vectors from the recombination step.
        Mc (np.ndarray): Recombined matrix from factorized partitions.
        partition (dict): Dictionary mapping partition indices to their respective data indices.
    """

    def __init__(self, reader, selection, svt, C, save_path: str = ""):
        """
        Initialize the DFC object with the given parameters.

        Args:
            reader: Reader instance to access data.
            selection: Array or list of indices indicating which data to process.
            svt: Object with a 'run' method to perform singular value thresholding.
            C: Array or parameter used for scaling in the data reading process.
            save_path (str): Optional path to save intermediate results.
        """
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
        """
        Randomly divide the selection indices into partitions of a given bin width.

        Args:
            bin_width (int): The number of indices in each partition.
                             Default is 100.
        """
        # Create an array of indices corresponding to the selection.
        vect = np.arange(len(self.selection))
        # Shuffle the indices randomly.
        np.random.shuffle(vect)
        # Calculate the number of partitions needed.
        n_i = int(np.ceil(len(self.selection) / bin_width))
        # Create partitions by slicing the shuffled indices.
        for i in range(n_i - 1):
            self.partition[i] = vect[i * bin_width: (i + 1) * bin_width]
        # Last partition takes the remaining indices.
        self.partition[n_i - 1] = vect[(n_i - 1) * bin_width:]
        print(len(self.partition), "times", self.reader.n_mz_bins, "x", bin_width)

    def factor(self, n_jobs: int = 10) -> None:
        """
        Factorize each partition using singular value thresholding (SVT) in parallel.

        The method uses joblib with tqdm to provide a progress bar.
        The SVT is applied on the data subset corresponding to each partition.

        Args:
            n_jobs (int): The number of parallel jobs to run. Default is 10.
        """
        with tqdm_joblib(tqdm(desc="Factor", total=len(self.partition))):
            # Process each partition in parallel.
            self.A = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self._svt)(self.selection[indices])
                for _, indices in self.partition.items()
            )

    def combine(self, p: int = 5, rank_oversample: int = 0) -> None:
        """
        Combine the factorized partitions to form the recombined matrix.

        This method computes a median rank from the factors, generates a random projection,
        and then performs several block multiplications to form a combined matrix. It then
        computes an SVD on the result to determine the combined left singular vectors (Uc)
        and reconstructs the combined matrix (Mc).

        Args:
            p (int): An oversampling parameter used in projection. Default is 5.
            rank_oversample (int): Additional oversampling for the rank. Default is 0.
        """
        # Determine the median rank from the factorization results.
        ranks = [aa[0].shape[1] for aa in self.A]
        median_rank = int(np.median(ranks))
        print("Median Rank", median_rank)

        # Generate a random projection matrix.
        G = np.random.randn(len(self.selection), median_rank + rank_oversample + p)
        # Compute a series of block multiplications.
        Cc = self.block_multiply(G, transpose=False)   # Compute M @ G
        Dc = self.block_multiply(Cc, transpose=True)    # Compute M.T @ (M @ G)
        Ec = self.block_multiply(Dc, transpose=False)    # Compute M @ (M.T @ (M @ G))
        Fc = self.block_multiply(Ec, transpose=True)     # Compute M.T @ (M @ (M.T @ (M @ G)))
        Gc = self.block_multiply(Fc, transpose=False)    # Compute M @ (M.T @ (M @ (M.T @ (M @ G))))
        # Perform singular value decomposition on the final product.
        Q, _, _ = linalg.svd(Gc)

        # Store the combined left singular vectors.
        self.Uc = Q[:, :median_rank + rank_oversample]
        # Initialize the combined matrix.
        self.Mc = np.zeros((self.Uc.shape[1], len(self.selection)), dtype="float32")

        # Reconstruct each partition and insert into the combined matrix.
        for i, aa in enumerate(self.A):
            # Unpack the SVT outputs: U, S, Vt, and the corresponding selection.
            U = np.array(aa[0])
            S = np.array(aa[1])
            Vt = np.array(aa[2])
            # Project the factors into the combined space and assign to the partition.
            self.Mc[:, self.partition[i]] = (self.Uc.T @ (U * S)) @ Vt

    def block_multiply(self, B: np.ndarray, transpose: bool) -> np.ndarray:
        """
        Perform block multiplication on the factorized partitions.
    
        Depending on the 'transpose' flag, the multiplication will use the transpose
        of the computed factors. This method aggregates contributions from each partition.
    
        Args:
            B (np.ndarray): The matrix to multiply.
            transpose (bool): Flag indicating whether to use transposed factors.
    
        Returns:
            np.ndarray: The result of the block multiplication.
        """
        # Initialize the output matrix with appropriate shape.
        if transpose:
            C = np.zeros((len(self.selection), B.shape[1]))
        else:
            C = np.zeros((self.A[0][0].shape[0], B.shape[1]))
    
        # Iterate over each partition's factorization result using enumerate.
        for idx, aa in enumerate(self.A):
            if transpose:
                # When transpose is True, use the transposed factors.
                U = np.array(aa[2]).T
                S = np.array(aa[1])
                Vt = np.array(aa[0]).T
                # Assign block multiplication result to the corresponding indices.
                C[self.partition[idx], :] = (U * S) @ (Vt @ B)
            else:
                # When transpose is False, compute the contribution and accumulate.
                U = np.array(aa[0])
                S = np.array(aa[1])
                Vt = np.array(aa[2])
                C += (U * S) @ (Vt @ B[self.partition[idx], :])
        return C


    def _read_in(self, selection) -> ss.csc_matrix:
        """
        Read and preprocess data from the given path using SPAReader.

        This method initializes the reader, retrieves the frame list,
        and constructs a sparse matrix representation of the data after applying
        a scaling factor based on the provided parameter C.

        Args:
            path: The path to the data file.
            selection: Indices of the frames to read.

        Returns:
            scipy.sparse.csc_matrix: A sparse matrix containing the preprocessed data.
        """
        # Obtain the first frame to determine data type.
        f = self.reader[0]
        iter_var = self.reader.framelist
        if np.min(iter_var) == 1: # Framelist is sometimes not properly starting from index 0, but 1
            iter_var -= 1
        # Initialize output array with zeros.
        out = np.zeros((self.reader.n_mz_bins, len(selection)), dtype=f.dtype)
        # Compute a scaling sparse matrix.
        C_sp = ss.csr_matrix(
            1 / (self.C[iter_var[selection]] / np.median(self.C[iter_var[selection]])),
            dtype="float32",
        )

        # Read in each frame specified by the selection.
        for k, i in enumerate(iter_var[selection]):
            f = self.reader[i]._init_csc()
            out[f.indices, k] = f.data

        # Multiply with the scaling factor.
        B = ss.csc_matrix(out, dtype="float32").multiply(C_sp)
        return B

    def _svt(self, selection) -> list:
        """
        Perform singular value thresholding (SVT) on the selected data subset.

        This method reads the data using the _read_in function, runs the SVT algorithm,
        and returns the factorization results.

        Args:
            selection: The data subset (indices) to process.

        Returns:
            list: A list containing the following elements:
                - Left singular vectors (U) as a dense matrix.
                - Singular values (S) as a dense matrix.
                - Right singular vectors (Vt) as a dense matrix.
                - The selection corresponding to this data block.
        """
        # Read in the data for the given selection.
        a = self._read_in(selection)
        # Run the singular value thresholding algorithm.
        self.svt.run(a)
        # Export the factorization results.
        obj = [
            self.svt._b[0].toarray(),
            self.svt._b[1].toarray(),
            self.svt._b[2].toarray(),
            selection,
        ]
        return obj

