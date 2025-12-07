import numpy as np
import time
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------
# 1. BASELINE MATRIX MULTIPLICATION
# ---------------------------------------------------------
def matmul_baseline(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


# ---------------------------------------------------------
# 2. LOOP TILING / CACHE BLOCKING
# ---------------------------------------------------------
def matmul_tiled(A, B, tile_size=32):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(0, n, tile_size):
        for j in range(0, n, tile_size):
            for k in range(0, n, tile_size):
                # Smaller blocks that fit in cache
                i_end = min(i + tile_size, n)
                j_end = min(j + tile_size, n)
                k_end = min(k + tile_size, n)

                for ii in range(i, i_end):
                    for jj in range(j, j_end):
                        temp = 0
                        for kk in range(k, k_end):
                            temp += A[ii, kk] * B[kk, jj]
                        C[ii, jj] += temp
    return C


# ---------------------------------------------------------
# 3. VECTORIZATION (NUMPY SIMD)
# ---------------------------------------------------------
def matmul_vectorized(A, B):
    return np.dot(A, B)  # uses BLAS, SIMD, cache optimizations


# ---------------------------------------------------------
# 4. PARALLELIZATION (MULTIPROCESSING)
# ---------------------------------------------------------
def parallel_worker(args):
    A_block, B = args
    return np.dot(A_block, B)


def matmul_parallel(A, B):
    n = A.shape[0]
    num_workers = cpu_count()

    # Split matrix A for distribution to workers
    A_split = np.array_split(A, num_workers)

    with Pool(num_workers) as pool:
        results = pool.map(parallel_worker, [(block, B) for block in A_split])

    return np.vstack(results)


# ---------------------------------------------------------
# 5. TEST & COMPARE PERFORMANCE
# ---------------------------------------------------------
if __name__ == "__main__":
    n = 300  # Adjust for your system
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    print("\n--- HPC Optimization Python Prototype ---")

    # Baseline
    start = time.time()
    C1 = matmul_vectorized(A, B)   # using vectorized as reference
    baseline_time = time.time() - start
    print(f"Vectorized (Reference) Time: {baseline_time:.4f} s")

    # Tiled
    start = time.time()
    C2 = matmul_tiled(A, B, tile_size=32)
    tiled_time = time.time() - start
    print(f"Tiled Matmul Time: {tiled_time:.4f} s")

    # Parallel
    start = time.time()
    C3 = matmul_parallel(A, B)
    parallel_time = time.time() - start
    print(f"Parallel Matmul Time: {parallel_time:.4f} s")

    # Validate correctness
    print("\nChecking correctness...")
    print("Tiled correct:", np.allclose(C1, C2, atol=1e-5))
    print("Parallel correct:", np.allclose(C1, C3, atol=1e-5))

    print("\n--- Performance Summary ---")
    print(f"Vectorized (SIMD):   {baseline_time:.4f} s")
    print(f"Tiled:               {tiled_time:.4f} s")
    print(f"Parallel:            {parallel_time:.4f} s")
