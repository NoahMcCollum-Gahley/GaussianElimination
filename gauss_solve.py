#----------------------------------------------------------------
# File:     gauss_solve.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Thu Sep 26 10:38:32 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# A Python wrapper module around the C library libgauss.so

import ctypes
import numpy as np

gauss_library_path = './libgauss.so'

def unpack(A):
    """ Extract L and U parts from A, fill with 0's and 1's """
    n = len(A)
    L = [[A[i][j] for j in range(i)] + [1] + [0 for j in range(i+1, n)]
         for i in range(n)]

    U = [[0 for j in range(i)] + [A[i][j] for j in range(i, n)]
         for i in range(n)]

    return L, U

def lu_c(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Modify the array in C (e.g., add 10 to each element)
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    return unpack(modified_array_2d)

def lu_python(A):
    n = len(A)
    for k in range(n):
        for i in range(k,n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)


def lu(A, use_c=False):
    if use_c:
        return lu_c(A)
    else:
        return lu_python(A)

def plu(A, use_c=False):
    if use_c:
        return plu_c(A)
    else:
        return plu_python(A)

def plu_python(A):
    n = len(A)
    L=np.zeros((n,n))
    #U=np.array(A)
    P_flat= np.linspace(0,n-1,n)

    for k in range(n):
        pivot_row = k
        pivot_elt = abs(A[k][k])
        for i in range(k+1, n):
            if abs(A[i][k]) > pivot_elt:
                pivot_elt = abs(A[i][k])
                pivot_row = i
        if pivot_row != k:
            P_flat[k], P_flat[pivot_row] = P_flat[pivot_row], P_flat[k]
            A[k], A[pivot_row] = A[pivot_row], A[k]
        for i in range(k,n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]
    
    P=P_flat.astype(int).tolist()
    L, U = unpack(A)
    return P,L,U

def plu_c(A):
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_A = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)
    c_P = (ctypes.c_int * n)()
    # Define the function signature
    lib.plu.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))

    # Modify the array in C (e.g., add 10 to each element)
    lib.plu(n, c_A, c_P)

    # Convert back to a 2D Python list of lists
    updated_A = [
        [c_A[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    L,U = unpack(updated_A)
    P = list(c_P)
    return P,L,U

if __name__ == "__main__":

    def get_A():
        """ Make a test matrix """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        return A

    A = get_A()

    L, U = lu(A, use_c = False)
    print(L)
    print(U)

    # Must re-initialize A as it was destroyed
    A = get_A()

    L, U = lu(A, use_c=True)
    print(L)
    print(U)
