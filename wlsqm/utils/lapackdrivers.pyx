# -*- coding: utf-8 -*-
#
# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
#
"""Simple Python interface to LAPACK for solving many independent linear equation systems efficiently in parallel. Built on top of scipy.linalg.cython_lapack.

Some scaling (preconditioning) routines are also provided, including two symmetry-preserving iterative algorithms.

This module:
   - adds OpenMP parallelization for independent problem instances on top of SciPy's Cython-level LAPACK interface,
   - makes it convenient to solve with many LHSs, also in parallel
   - hides practical details such as creation/destruction of work arrays, making the interface simple to use.

Two interfaces are provided:
   - a simple Python interface using Cython memoryviews (compatible with np.arrays)
   - a low-level C interface using raw pointers and explicit sizes (useful for cimport'ing from other Cython modules)

Note that cython_lapack already converts LAPACK errors into Python exceptions; we simply pass them through.

** IMPORTANT **
  - Real-valued data only (dtype np.float64)
  - Compatible with Fortran-contiguous arrays only! (so pass order='F' when creating np.arrays to use with this)
  - Fast, not safe; be sure to provide correct sizes etc.!


Naming scheme (in shellglob notation):
  *s = multiple RHS (but with the same LHS for all).
       These are one-shot deals that reuse the matrix factorization internally.
       However, the pivot information is not returned, so the matrix A
       is destroyed (overwritten) during the call.

  m* = multiple LHS (a separate single RHS for each)
       These simply loop over the problem instances.

  *p = parallel (multi-threaded using OpenMP)
       These introduce parallel looping over problem instances.

  *factor*   = the routine that factors the matrix and generates pivot information.
  *factored* = the solver routine that uses the factored matrix and pivot information.

  The factoring routines are useful for solving with many RHSs, when all the RHSs are not available at once
  (e.g. in PDE solvers, timestepping with a mass matrix that remains constant in time).

See the function docstrings for details.


JJ 2016-10-07
"""

from __future__ import division
from __future__ import absolute_import

cimport cython
cimport cython.parallel
cimport openmp

from libc.math cimport sqrt
from libc.math cimport fabs   as c_abs

from libc.stdlib cimport malloc, free

import numpy as np

from scipy.linalg.cython_lapack cimport dgtsv,    dsysv, dsytrf, dsytrs,    dgesv, dgetrf, dgetrs
from scipy.linalg.cython_lapack cimport dgesvd

# fast inline min/max for C code
#
cdef inline int cimin(int a, int b) nogil:
    return a if a < b else b
cdef inline int cimax(int a, int b) nogil:
    return a if a > b else b


##############################################################################################################
# Parallelization helper
##############################################################################################################

def distribute_items( int nitems, int ntasks ):  # Python wrapper
    """def distribute_items( int nitems, int ntasks ):

Distribute work items (numbered 0, 1, ..., nitems-1) across ntasks tasks, assuming equal workload per item.
"""
    cdef int[::1] blocksizes = np.empty( (ntasks,), dtype=np.int32 )
    cdef int[::1] baseidxs   = np.empty( (ntasks,), dtype=np.int32 )
    distribute_items_c( nitems, ntasks, &blocksizes[0], &baseidxs[0] )  # ntasks usually very small, no point in "with nogil:"
    return (blocksizes, baseidxs)

# nitems     : in, number of items
# ntasks     : in, number of tasks
# blocksizes : out, an array of size (ntasks,) containing the number of items for each task
# baseidxs   : out, an array of size (ntasks,) containing the index of the first item for each task
#
# All arrays must be allocated by caller!
#
cdef void distribute_items_c( int nitems, int ntasks, int* blocksizes, int* baseidxs ) nogil:  # C implementation
    cdef int baseblocksize = nitems // ntasks   # how many items per task
    cdef int remainder     = nitems %  ntasks
    if baseblocksize == 0:  # no whole blocks?
        ntasks = remainder

    cdef int k
    for k in range(ntasks):
        blocksizes[k] = baseblocksize
        if k < remainder:
            blocksizes[k] += 1  # distribute the remainder across the first tasks

    baseidxs[0] = 0
    for k in range(1,ntasks):
        baseidxs[k] = baseidxs[k-1] + blocksizes[k-1]


##############################################################################################################
# Matrix handling helpers
##############################################################################################################

def copygeneral( double[::1,:] O, double[::1,:] I ):
    """def copygeneral( double[::1,:] O, double[::1,:] I ):

Copy a Fortran-contiguous rank-2 array I into a Fortran-contiguous rank-2 array O.

O must have the same shape as I, and must have been allocated by the caller.
"""
    cdef int ncols=I.shape[0], nrows=I.shape[1]
    with nogil:
        copygeneral_c( &O[0,0], &I[0,0], nrows, ncols )

cdef void copygeneral_c( double* O, double* I, int nrows, int ncols ) nogil:
    cdef int nelems=ncols*nrows
    cdef int e  # element storage offset as counted from the beginning of the matrix
    for e in range(nelems):  # TODO: maybe we could even memcpy(&O[0], &I[0], nelems*sizeof(double))?
        O[e] = I[e]


def copysymmu( double[::1,:] O, double[::1,:] I ):
    """def copysymmu( double[::1,:] O, double[::1,:] I ):

Copy the upper triangle (including the diagonal) of a square Fortran-contiguous rank-2 array I into a square Fortran-contiguous rank-2 array O.

The strict lower triangles of I and O are not referenced.

O must have the same shape as I, and must have been allocated by the caller.
"""
    cdef int ncols=I.shape[0], nrows=I.shape[1]
    with nogil:
        copysymmu_c( &O[0,0], &I[0,0], nrows, ncols )

cdef void copysymmu_c( double* O, double* I, int nrows, int ncols ) nogil:
    cdef int j, i, colbaseidx, e
    for j in range(ncols):        # column
        colbaseidx = j*nrows
        # general rectangular array would need range(min(j+1, m)), we skip the min since we need this to work for square arrays only
        for i in range(j+1):  # row - handle the upper triangular part only (since A is symmetric and the lower triangle is not used)
            e = colbaseidx + i  # element storage offset as counted from the beginning of the matrix
            O[e] = I[e]


def symmetrize( double[::1,:] A ):
    """def symmetrize( double[::1,:] A ):

Symmetrize a square Fortran-contiguous rank-2 array in-place.

This is a fast Cython implementation of:
    A = 0.5 * (A + A.T)
"""
    cdef int ncols=A.shape[0], nrows=A.shape[1]  # cols, rows
    with nogil:
        symmetrize_c( &A[0,0], nrows, ncols )

cdef void symmetrize_c( double* A, int nrows, int ncols ) nogil:
    cdef int i, j
    cdef double tmp

    # loop over strict upper triangle only
    for j in range(1,ncols):  # column
        for i in range(j):    # row
            tmp = 0.5 * (A[i + j*nrows] + A[j + i*nrows])  # A[i,j] + A[j,i]
            A[i + j*nrows] = tmp
            A[j + i*nrows] = tmp


def msymmetrize( double[::1,:,:] A ):
    """def msymmetrize( double[::1,:,:] A ):

Symmetrize many square Fortran-contiguous arrays in-place, single-threaded.

A : a Fortran-contiguous rank-3 array, shape (N,n,n).

This is a fast Cython implementation of:
    for j in range(N):
        A[j,:,:] = 0.5 * (A[j,:,:] + A[j,:,:].T)
"""
    cdef int ncols=A.shape[0], nrows=A.shape[1], nlhs=A.shape[2]  # cols, rows, arrays
    with nogil:
        msymmetrize_c( &A[0,0,0], nrows, ncols, nlhs )

cdef void msymmetrize_c( double* A, int nrows, int ncols, int nlhs ) nogil:
    cdef int nelems=nrows*ncols
    cdef int i, j, k
    cdef double tmp

    for k in range(nlhs):
        # loop over strict upper triangle only
        for j in range(1,ncols):  # column
            for i in range(j):    # row
                tmp = 0.5 * ( A[i + nrows*j + nelems*k] + A[j + nrows*i + nelems*k] )  # A[i,j,k] + A[j,i,k]
                A[i + nrows*j + nelems*k] = tmp
                A[j + nrows*i + nelems*k] = tmp


def msymmetrizep( double[::1,:,:] A, int ntasks ):
    """def msymmetrizep( double[::1,:,:] A, int ntasks ):

Symmetrize many square Fortran-contiguous arrays in-place, multi-threaded.

A      : a Fortran-contiguous rank-3 array, shape (N,n,n).
ntasks : number of threads for OpenMP
"""
    cdef int ncols=A.shape[0], nrows=A.shape[1], nlhs=A.shape[2]  # cols, rows, arrays
    with nogil:
        msymmetrizep_c( &A[0,0,0], nrows, ncols, nlhs, ntasks )

cdef void msymmetrizep_c( double* A, int nrows, int ncols, int nlhs, int ntasks ) nogil:
    cdef int nelems=nrows*ncols
    cdef int i, j, k
    cdef double tmp

    for k in cython.parallel.prange(nlhs, num_threads=ntasks):
        # loop over strict upper triangle only
        for j in range(1,ncols):  # column
            for i in range(j):    # row
                tmp = 0.5 * ( A[i + nrows*j + nelems*k] + A[j + nrows*i + nelems*k] )  # A[i,j,k] + A[j,i,k]
                A[i + nrows*j + nelems*k] = tmp
                A[j + nrows*i + nelems*k] = tmp


##############################################################################################################
# Preconditioning (scaling)
##############################################################################################################

# With the exception of apply_scaling_c(), the low-level routines treat A as input only.
# The scaling factors will be written to the arrays row_scale[] and col_scale[].
#
# The routine apply_scaling_c() freezes the current scaling by applying the scaling factors in-place.
# This should be done last, before handing A to one of the solver routines.
#
# But be sure not to destroy row_scale[] and col_scale[] immediately; it is the caller's responsibility
# to scale the RHS accordingly (b[j] *= row_scale[j]), and after solving the linear system,
# to scale the solution (x[m] *= col_scale[m]).

# For all C routines in this section:
#
# - arrays passed as raw pointer + explicit size
# - all arrays must be allocated by caller
#
# A         : in (out for apply_scaling_c() only), matrix of size (nrows,ncols), Fortran-contiguous
# nrows     : in, number of rows
# ncols     : in, number of columns
# row_scale : out, vector of row scalings, size (nrows,)     (needed by caller for scaling RHSs; scaled_b[j] = original_b[j] * row_scale[j])
# col_scale : out, vector of column scalings, size (ncols,)  (needed by caller for scaling the solution; final_x[m] = scaled_x[m] * col_scale[m])

# Initialize scaling for rows and columns of a general matrix A.
cdef void init_scaling_c( int nrows, int ncols, double* row_scale, double* col_scale ) nogil:
    cdef int j, m
    for m in range(ncols):  # col
        col_scale[m] = 1.
    for j in range(nrows):  # row
        row_scale[j] = 1.

# Freeze the given scaling factors by applying them to A in-place.
cdef void apply_scaling_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil:
    cdef int j, m
    cdef double c
    for m in range(ncols):
        c = col_scale[m]
        for j in range(nrows):
            A[j + nrows*m] *= (row_scale[j] * c)


# Helpers for do_rescale()
#
ctypedef int (*rescale_func_ptr)(double*, int, int, double*, double*) nogil
class ScalingAlgo:  # enum (TODO: use real enum type for Python 3.4+)
    """Enum for scaling algorithms for do_rescale()."""
    ALGO_COLS_EUCL = 1
    ALGO_ROWS_EUCL = 2
    ALGO_TWOPASS   = 3
    ALGO_RUIZ2001  = 4
    ALGO_SCALGM    = 5

def do_rescale( double[::1,:] A, int algo ):
    """def do_rescale( double[::1,:] A, int algo ):

Generic dispatcher for matrix scaling (preconditioning) routines.

Scaling is useful to reduce the condition number of A, giving more correct digits in the numerical solution
of a linear equation system involving A.

Specifically, DGESV (used by general()) works by LU factorization with pivoting. For this algorithm the
relative accuracy is O(kappa(A) * machine_epsilon), where at double precision, machine_epsilon ~ 1e-16.
( See e.g. http://scicomp.stackexchange.com/questions/19289/are-direct-solvers-affect-by-the-condition-number-of-a-matrix )

This routine handles the creation of the row_scale and col_scale arrays, calls the scaling routine,
scales A in-place, and returns the pair of arrays (row_scale, col_scale).

Parameters:
    A    : general matrix as Fortran-contiguous rank-2 arraylike, shape (nrows,ncols) (shape determined automatically)
    algo : int, see class ScalingAlgo for available values

Return value: tuple (row_scale, col_scale), where
    row_scale : out, vector of row scalings, shape (nrows,)
                Needed by caller for scaling the RHS. In A*x = b, on the input side:

                for j in range(nrows):
                    scaled_b[j] = original_b[j] * row_scale[j]

    col_scale : out, vector of column scalings, shape (ncols,)
                Needed by caller for scaling the solution. In A*x = b, on the output side:

                for m in range(ncols):
                    true_x[m]   = scaled_x[m]   * col_scale[m]
"""
    # create result arrays
    cdef int nrows=A.shape[0], ncols=A.shape[1]
    cdef double[::1] row_scale = np.empty( (nrows,), dtype=np.float64 )
    cdef double[::1] col_scale = np.empty( (ncols,), dtype=np.float64 )

    cdef rescale_func_ptr scaler
    if algo == ScalingAlgo.ALGO_COLS_EUCL:
        scaler = rescale_columns_c
    elif algo == ScalingAlgo.ALGO_ROWS_EUCL:
        scaler = rescale_rows_c
    elif algo == ScalingAlgo.ALGO_TWOPASS:
        scaler = rescale_twopass_c
    elif algo == ScalingAlgo.ALGO_RUIZ2001:
        scaler = rescale_ruiz2001_c
    elif algo == ScalingAlgo.ALGO_SCALGM:
        scaler = rescale_scalgm_c
    else:
        raise ValueError("Unknown algorithm identifier, got %d" % (algo))

    with nogil:
        scaler( &A[0,0], nrows, ncols, &row_scale[0], &col_scale[0] )  # internally calls init_scaling_c()
        apply_scaling_c( &A[0,0], nrows, ncols, &row_scale[0], &col_scale[0] )

    return (row_scale, col_scale)


# Python interface methods for specific algorithms.
#
# A : in, rank-2 arraylike (Fortran-contiguous)
#
# return value: tuple (row_scale, col_scale) of rank-1 arrays
#
def rescale_columns( double[::1,:] A ):
    """def rescale_columns( double[::1,:] A ):

Algorithm: column scaling only.

Do not call this directly; instead, use do_rescale(). This function is exported only to make its docstring visible.

  - changes the "size of units" of the elements of x
    - so the caller must undo this when reading "x" after solving the linear equation system.
    - in an ODE/PDE context, intuitively, this balances the size of numbers needed for function values
      themselves and different orders of derivatives, by undoing the effect of multiplications by
      different powers of grid spacing (usually "one unit of derivative" is very small,
      so many of them are required to represent even a moderate slope).
  - destroys symmetry of the matrix!
"""
    return do_rescale( A, ScalingAlgo.ALGO_COLS_EUCL )

cdef int rescale_columns_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil:
    init_scaling_c(       nrows, ncols, row_scale, col_scale )

    cdef int j, m
    cdef double acc, tmp, c

    for m in range(ncols):
        c = col_scale[m]  # old column scaling multiplier in column m
        acc = 0.
        for j in range(nrows):
            tmp = A[j + nrows*m] * (c * row_scale[j])  # A[j,m], with old scaling
            acc += tmp*tmp  # | A[j,m] |**2
        col_scale[m] /= sqrt(acc)  # update column scaling by euclidean norm of A[:,m] (in old scaling)

    return 1


def rescale_rows( double[::1,:] A ):
    """def rescale_rows( double[::1,:] A ):

Algorithm: row scaling only

Do not call this directly; instead, use do_rescale(). This function is exported only to make its docstring visible.

  - changes the scaling of the RHS
    - so the caller must scale "b" by the same factor, too
    - intuitively, balances the size of numbers needed in different equations of the linear equation system
  - does NOT affect the scaling of the components of x
  - destroys symmetry of the matrix!
"""
    return do_rescale( A, ScalingAlgo.ALGO_ROWS_EUCL )

cdef int rescale_rows_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil:
    init_scaling_c(       nrows, ncols, row_scale, col_scale )

    cdef int j, m
    cdef double acc, tmp, r

    for j in range(nrows):
        r = row_scale[j]  # old row scaling multiplier on row j
        acc = 0.
        for m in range(ncols):
            tmp = A[j + nrows*m] * (r * col_scale[m])  # A[j,m], with old scaling
            acc += tmp*tmp  # | A[j,m] |**2
        row_scale[j] /= sqrt(acc)  # update row scaling by euclidean norm of A[j,:] (in old scaling)

    return 1


def rescale_twopass( double[::1,:] A ):
    """def rescale_twopass( double[::1,:] A ):

Naive two-pass rescale of columns first, then rows.

Do not call this directly; instead, use do_rescale(). This function is exported only to make its docstring visible.

Destroys symmetry, but is simple and fast (no iteration needed).

  - we scale the columns of A, producing a column-scaled matrix A'
  - then we scale the rows of A', producing A''
  - after the second step the columns are no longer balanced, so A'' will *not* be doubly stochastic

If one wants a doubly stochastic matrix, iterative approaches are possible.
See rescale_ruiz2001() and rescale_scalgm().
"""

    return do_rescale( A, ScalingAlgo.ALGO_TWOPASS )

cdef int rescale_twopass_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil:
    init_scaling_c(       nrows, ncols, row_scale, col_scale )

    cdef int j, m
    cdef double acc, tmp, c, r

    for m in range(ncols):
        c = col_scale[m]  # old column scaling multiplier in column m
        acc = 0.
        for j in range(nrows):
            tmp = A[j + nrows*m] * (c * row_scale[j])  # A[j,m], with old scaling
            acc += tmp*tmp  # | A[j,m] |**2
        col_scale[m] /= sqrt(acc)  # update column scaling by euclidean norm of A[:,m] (in old scaling)

    for j in range(nrows):
        r = row_scale[j]  # old row scaling multiplier on row j
        acc = 0.
        for m in range(ncols):
            tmp = A[j + nrows*m] * (r * col_scale[m])  # A[j,m], with old scaling
            acc += tmp*tmp  # | A[j,m] |**2
        row_scale[j] /= sqrt(acc)  # update row scaling by euclidean norm of A[j,:] (in old scaling)

    return 1


def rescale_ruiz2001( double[::1,:] A ):
    """def rescale_ruiz2001( double[::1,:] A ):

Simultaneous row and column iterative scaling using algorithm of Ruiz (2001).

Do not call this directly; instead, use do_rescale(). This function is exported only to make its docstring visible.

Preserves matrix symmetry, at the cost of requiring an iterative process to find the scaling factors.

The rows and columns are equilibrated in the l-infinity norm (maximum absolute value of element).

Reference:
    Daniel Ruiz, A Scaling Algorithm to Equilibrate Both Rows and Columns Norms in Matrices. Report RAL-TR-2001-034. 2001.
"""
    return do_rescale( A, ScalingAlgo.ALGO_RUIZ2001 )

cdef int rescale_ruiz2001_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil:
    # temporary work space
    cdef double* DR     = <double*>malloc( nrows*sizeof(double) )
    cdef double* DC     = <double*>malloc( ncols*sizeof(double) )
    cdef double* DRprev = <double*>malloc( nrows*sizeof(double) )
    cdef double* DCprev = <double*>malloc( ncols*sizeof(double) )

    init_scaling_c(       nrows, ncols, row_scale, col_scale )  # D1 and D2 in algorithm 2.1 in Ruiz (2001)
    init_scaling_c(       nrows, ncols, DR,        DC        )  # DR and DC in algorithm 2.1 (curr iterate)
    init_scaling_c(       nrows, ncols, DRprev,    DCprev    )  # DR and DC in algorithm 2.1 (prev iterate)

    cdef int k, j, m
    cdef double acc, tmp
    DEF epsilon = 1e-15  # maybe appropriate for double precision?
    DEF niters = 100     # maximum number of iterations to take
    for k in range(niters):
        # compute new DR
        for j in range(nrows):
            r = DRprev[j]  # old row scaling multiplier on row j
            acc = 0.
            for m in range(ncols):
                tmp = c_abs( A[j + nrows*m] / (r * DCprev[m]) )  # |A[j,m]|, with old scaling
                if tmp > acc:  # l-infinity norm (largest abs(element))
                    acc = tmp
            DR[j] = sqrt(acc)  # update row scaling by norm of A[j,:] (in old scaling)

        # compute new DC
        for m in range(ncols):
            c = DCprev[m]  # old column scaling multiplier in column m
            acc = 0.
            for j in range(nrows):
                tmp = c_abs( A[j + nrows*m] / (c * DRprev[j]) )  # |A[j,m]|, with old scaling
                if tmp > acc:  # l-infinity norm (largest abs(element))
                    acc = tmp
            DC[m] = sqrt(acc)  # update column scaling by norm of A[:,m] (in old scaling)

        # update D1, D2, DR, DC
        for j in range(nrows):
            DRprev[j] *= DR[j]
            row_scale[j] /= DR[j]  # our convention: one should *multiply* by the scale factors
        for m in range(ncols):
            DCprev[m] *= DC[m]
            col_scale[m] /= DC[m]

        # convergence check
        #
        # find max abs(1 - |r_j|_infty), j = 0, ..., nrows-1
        #
        acc = c_abs(1. - DR[0]*DR[0])  # DR[j] stores the square root of the infinity-norm
        for j in range(1, nrows):
            tmp = c_abs(1. - DR[j]*DR[j])
            if tmp > acc:
                acc = tmp
        if acc < epsilon:  # rows converged, check columns too
            # find max abs(1 - |c_m|_infty), m = 0, ..., ncols-1
            #
            acc = c_abs(1. - DC[0]*DC[0])  # DC[m] stores the square root of the infinity-norm
            for m in range(1, ncols):
                tmp = c_abs(1. - DC[m]*DC[m])
                if tmp > acc:
                    acc = tmp
            if acc < epsilon: # columns converged too
                break

    free( <void*>DCprev )
    free( <void*>DRprev )

    free( <void*>DC )
    free( <void*>DR )

    return k+1  # number of iterations taken


def rescale_scalgm( double[::1,:] A ):
    """def rescale_scalgm( double[::1,:] A ):

Simultaneous row and column iterative scaling using the SCALGM algorithm of Chiang and Chandler (2008).

Preserves matrix symmetry, at the cost of requiring an iterative process to find the scaling factors.

The rows and columns are equilibrated in the l-infinity norm (maximum absolute value of element).
The result is the same as by Ruiz (2001) (see rescale_ruiz2001()); both algorithms are provided,
because they may exhibit different performance.

Reference:
     Chin-Chieh Chiang and John P. Chandler. An Approximate Equation for the Condition Numbers
     of Well-scaled Matrices. Paper 036, ENG 183, in Proceedings of The 2008 IAJC-IJME
     International Conference. ISBN 978-1-60643-379-9.
"""
    return do_rescale( A, ScalingAlgo.ALGO_SCALGM )

# Helper for SCALGM
cdef void basic_scale_up_rows( double* A, int nrows, int ncols, double* rs, double* cs, double* mod_cs, double* new_rs ) nogil:
    cdef int j, m
    cdef double r, acc, tmp
    if mod_cs != <double*>0:  # multiplicative modifier from previous column scaling not yet applied in current rs,cs
                              # (this can be used to implement "scale columns first, then rows" without updating rs,cs)
        for j in range(nrows):
            r = rs[j]  # current row scaling multiplier on row j
            acc = 0.
            for m in range(ncols):
                tmp = c_abs( A[j + nrows*m] * (r * cs[m] * mod_cs[m]) )  # |A[j,m]|, with current scaling and mod_cs applied
                if acc == 0.  or  tmp > 0.  and  tmp < acc:
                    acc = tmp
            new_rs[j] = 1./acc  # acc = smallest non-zero magnitude of any element on row j

    else:  # no modifier
        for j in range(nrows):
            r = rs[j]  # current row scaling multiplier on row j
            acc = 0.
            for m in range(ncols):
                tmp = c_abs( A[j + nrows*m] * (r * cs[m]) )  # |A[j,m]|, with current scaling
                if acc == 0.  or  tmp > 0.  and  tmp < acc:
                    acc = tmp
            new_rs[j] = 1./acc  # acc = smallest non-zero magnitude of any element on row j

# Helper for SCALGM
cdef void basic_scale_up_cols( double* A, int nrows, int ncols, double* rs, double* mod_rs, double* cs, double* new_cs ) nogil:
    cdef int j, m
    cdef double c, acc, tmp
    if mod_rs != <double*>0:  # multiplicative modifier from previous row scaling not yet applied in current rs,cs
                              # (this can be used to implement "scale rows first, then columns" without updating rs,cs)
        for m in range(ncols):
            c = cs[m]  # current column scaling multiplier in column j
            acc = 0.
            for j in range(nrows):
                tmp = c_abs( A[j + nrows*m] * (c * rs[j] * mod_rs[j]) )  # |A[j,m]|, with current scaling and mod_rs applied
                if acc == 0.  or  tmp > 0.  and  tmp < acc:
                    acc = tmp
            new_cs[m] = 1./acc  # acc = smallest non-zero magnitude of any element in column m

    else:  # no modifier
        for m in range(ncols):
            c = cs[m]  # current column scaling multiplier in column j
            acc = 0.
            for j in range(nrows):
                tmp = c_abs( A[j + nrows*m] * (c * rs[j]) )  # |A[j,m]|, with current scaling
                if acc == 0.  or  tmp > 0.  and  tmp < acc:
                    acc = tmp
            new_cs[m] = 1./acc  # acc = smallest non-zero magnitude of any element in column m

# Helper for SCALGM
cdef void basic_scale_down_rows( double* A, int nrows, int ncols, double* rs, double* cs, double* mod_cs, double* new_rs ) nogil:
    cdef int j, m
    cdef double r, acc, tmp
    if mod_cs != <double*>0:  # multiplicative modifier from previous column scaling not yet applied in current rs,cs
                              # (this can be used to implement "scale columns first, then rows" without updating rs,cs)
        for j in range(nrows):
            r = rs[j]  # current row scaling multiplier on row j
            acc = 0.
            for m in range(ncols):
                tmp = c_abs( A[j + nrows*m] * (r * cs[m] * mod_cs[m]) )  # |A[j,m]|, with current scaling and mod_cs applied
                if tmp > acc:
                    acc = tmp
            new_rs[j] = 1./acc  # acc = largest magnitude of any element on row j

    else:  # no modifier
        for j in range(nrows):
            r = rs[j]  # current row scaling multiplier on row j
            acc = 0.
            for m in range(ncols):
                tmp = c_abs( A[j + nrows*m] * (r * cs[m]) )  # |A[j,m]|, with current scaling
                if tmp > acc:
                    acc = tmp
            new_rs[j] = 1./acc  # acc = largest magnitude of any element on row j

# Helper for SCALGM
cdef void basic_scale_down_cols( double* A, int nrows, int ncols, double* rs, double* mod_rs, double* cs, double* new_cs ) nogil:
    cdef int j, m
    cdef double c, acc, tmp
    if mod_rs != <double*>0:  # multiplicative modifier from previous row scaling not yet applied in current rs,cs
                              # (this can be used to implement "scale rows first, then columns" without updating rs,cs)
        for m in range(ncols):
            c = cs[m]  # current column scaling multiplier in column j
            acc = 0.
            for j in range(nrows):
                tmp = c_abs( A[j + nrows*m] * (c * rs[j] * mod_rs[j]) )  # |A[j,m]|, with current scaling and mod_rs applied
                if tmp > acc:
                    acc = tmp
            new_cs[m] = 1./acc  # acc = largest magnitude of any element in column m

    else:  # no modifier
        for m in range(ncols):
            c = cs[m]  # current column scaling multiplier in column j
            acc = 0.
            for j in range(nrows):
                tmp = c_abs( A[j + nrows*m] * (c * rs[j]) )  # |A[j,m]|, with current scaling
                if tmp > acc:
                    acc = tmp
            new_cs[m] = 1./acc  # acc = largest magnitude of any element in column m

# SCALGM driver
cdef int rescale_scalgm_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil:
    # temporary work space
    cdef double* DR1 = <double*>malloc( nrows*sizeof(double) )
    cdef double* DC1 = <double*>malloc( ncols*sizeof(double) )
    cdef double* DR2 = <double*>malloc( nrows*sizeof(double) )
    cdef double* DC2 = <double*>malloc( ncols*sizeof(double) )

    init_scaling_c(       nrows, ncols, row_scale, col_scale )
    init_scaling_c(       nrows, ncols, DR1,        DC1      )
    init_scaling_c(       nrows, ncols, DR2,        DC2      )

    cdef int k, j, m
    cdef double acc, acc2, tmp
    cdef int operation_mode = 1
    DEF epsilon = 1e-15  # maybe appropriate for double precision?
    DEF niters = 100     # maximum number of iterations to take
    for k in range(niters):
        # let "original matrix" = the matrix at the start of the iteration.

        if operation_mode == 1:  # still iterating also steps 1-3?
            # step 1: scale up the original matrix by rows, then columns using the "basic" algorithm
            #
            basic_scale_up_rows( A, nrows, ncols, row_scale, col_scale, <double*>0, DR1 )
            basic_scale_up_cols( A, nrows, ncols, row_scale, DR1, col_scale, DC1 )  # apply also DR1 when computing DC1

            # step 2: scale up the original matrix by columns, then rows using the "basic" algorithm
            #
            basic_scale_up_cols( A, nrows, ncols, row_scale, <double*>0, col_scale, DC2 )
            basic_scale_up_rows( A, nrows, ncols, row_scale, col_scale, DC2, DR2 )  # apply also DC2 when computing DR2

            # step 3: scale up the original matrix by the geometric mean of the two row/column factors
            #
            # (updating the scaling info!)
            #
            for j in range(nrows):
                row_scale[j] *= sqrt( DR1[j] * DR2[j] )
            for m in range(ncols):
                col_scale[m] *= sqrt( DC1[m] * DC2[m] )

        # let "resulting matrix" = the result of step 3, if operation_mode == 1; else "resulting matrix" = the original matrix.

        # step 4: scale down the resulting matrix down by rows, then columns using the "basic" algorithm
        #
        basic_scale_down_rows( A, nrows, ncols, row_scale, col_scale, <double*>0, DR1 )
        basic_scale_down_cols( A, nrows, ncols, row_scale, DR1, col_scale, DC1 )  # apply also DR1 when computing DC1

        # step 5: scale down the resulting matrix by columns, then rows using the "basic" algorithm
        #
        basic_scale_down_cols( A, nrows, ncols, row_scale, <double*>0, col_scale, DC2 )
        basic_scale_down_rows( A, nrows, ncols, row_scale, col_scale, DC2, DR2 )  # apply also DC2 when computing DR2

        # step 6: scale down the resulting matrix by the geometric mean of the two row/column factors
        #
        # (again, updating the scaling info!)
        #
        for j in range(nrows):
            row_scale[j] *= sqrt( DR1[j] * DR2[j] )
        for m in range(ncols):
            col_scale[m] *= sqrt( DC1[m] * DC2[m] )

        # convergence check
        #
        # find max abs(1 - |r_j|_infty), j = 0, ..., nrows-1
        #
        acc2 = 0.
        for j in range(nrows):
            r = row_scale[j]  # old row scaling multiplier on row j
            acc = 0.
            for m in range(ncols):
                tmp = c_abs( A[j + nrows*m] * (r * col_scale[m]) )  # |A[j,m]|, with old scaling
                if tmp > acc:
                    acc = tmp
            tmp = c_abs(1. - acc)
            if tmp > acc2:
                acc2 = tmp

        if acc2 < epsilon:  # rows converged, check columns too
            # find max abs(1 - |c_m|_infty), m = 0, ..., ncols-1
            #
            acc2 = 0.
            for m in range(ncols):
                c = col_scale[m]  # old row scaling multiplier in column m
                acc = 0.
                for j in range(nrows):
                    tmp = c_abs( A[j + nrows*m] * (c * row_scale[j]) )  # |A[j,m]|, with old scaling
                    if c_abs(tmp) > acc:
                        acc = c_abs(tmp)
                tmp = c_abs(1. - acc)
                if tmp > acc2:
                    acc2 = tmp

            if acc2 < epsilon: # columns converged too
                if operation_mode == 1:
                    operation_mode = 2  # switch operation mode to iterate only steps 4-6
                else:
                    break  # all done

    free( <void*>DC2 )
    free( <void*>DR2 )
    free( <void*>DC1 )
    free( <void*>DR1 )

    return k+1  # number of iterations taken


##############################################################################################################
# Simple example (tridiagonal matrices)
##############################################################################################################

cpdef int tridiag( double[::1] a, double[::1] b, double[::1] c, double[::1] x ) nogil except -1:
    """cpdef int tridiag( double[::1] a, double[::1] b, double[::1] c, double[::1] x ) nogil except -1:

Tridiagonal solver example from:

Henriksen, Ian. Circumventing The Linker: Using SciPy’s BLAS and LAPACK Within Cython. PROC. OF THE 14th PYTHON IN SCIENCE CONF. (SCIPY 2015), 49-52.
http://conference.scipy.org/proceedings/scipy2015/pdfs/ian_henriksen.pdf

This routine is provided mainly as a minimal form of documentation; the main purpose of this module is
in the routines for general and symmetric matrices.

For the parameters, see a,b,c,x of LAPACK's DGTSV.

For the double[::1] syntax, see "Typed Memoryviews" in Cython documentation:
http://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html

For the "except -1", see:
http://docs.cython.org/en/latest/src/reference/language_basics.html#error-and-exception-handling
"""
    cdef int n=b.shape[0], nrhs=1, info
    # Solution is written over the values in x.
    # http://www.netlib.org/lapack/lapack-3.1.1/html/dgtsv.f.html
    dgtsv(&n, &nrhs, &a[0], &b[0], &c[0], &x[0], &n, &info)
    return 0


##############################################################################################################
# Symmetric matrices
##############################################################################################################

def symmetric2x2( double[::1,:] A, double[::1] b ):
    """def symmetric2x2( double[::1,:] A, double[::1] b ):

Solve a symmetric 2x2 system.

This is provided for completeness; the special case of a 2x2 matrix is fastest to just compute directly.

To match the other routines, we use only the upper triangle.

A : symmetric 2x2 matrix as Fortran-contiguous rank-2 arraylike, shape (2,2)
b : rank-1 arraylike, shape (2,)
    in  : RHS
    out : solution x to  A*x = b
"""
    symmetric2x2_c( &A[0,0], &b[0] )

cdef int symmetric2x2_c( double* A, double* b ) nogil except -1:
    # Ainv = 1/Adet * [ [A22, -A12], [-A21, A11] ]
    # x = Ainv*b
    #
    # with A21 = A12

    # each of these gets at least two accesses (a01 gets four)
    cdef double a00 = A[0]
    cdef double a01 = A[2]  # A[0,1] in Fortran-contiguous storage
    cdef double a11 = A[3]
    cdef double b0  = b[0]
    cdef double b1  = b[1]

    # solve, overwrite b with solution
    cdef double dm1 = 1. / ( a00*a11 - a01*a01 )
    b[0] = dm1 * ( a11*b0 - a01*b1 )
    b[1] = dm1 * ( a00*b1 - a01*b0 )

    return 0


def symmetric( double[::1,:] A, double[::1] b ):
    """def symmetric( double[::1,:] A, double[::1] b ):

Solve a symmetric system.

Only the upper triangle of A is used.

A : symmetric matrix as Fortran-contiguous rank-2 arraylike, shape (n,n)
    in  : matrix A
    out : destroyed (overwritten)
b : rank-1 arraylike, shape (n,)
    in  : RHS
    out : solution x to  A*x = b
"""
    cdef int n=b.shape[0]
    with nogil:
        symmetrics_c( &A[0,0], &b[0], n, 1 )

cdef int symmetric_c( double* A, double* b, int n ) nogil except -1:
    return symmetrics_c( A, b, n, 1 )


def symmetricfactor( double[::1,:] A ):
    """def symmetricfactor( double[::1,:] A ):

Compute the UDUT factorization only.

A : symmetric matrix as Fortran-contiguous rank-2 arraylike, shape (n,n)
    in  : matrix A
    out : the UDUT factorization of A, as returned by DSYTRF

Return value:
    pivot array (rank-1, shape (n,), dtype np.int32). This is needed for the solve step.
"""
    cdef int n=A.shape[0]
    cdef int[::1] ipiv = np.empty( (n,), dtype=np.int32 )
    with nogil:
        symmetricfactor_c( &A[0,0], &ipiv[0], n )
    return ipiv

cdef int symmetricfactor_c( double* A, int* ipiv, int n ) nogil except -1:
    return msymmetricfactor_c( A, ipiv, n, 1 )

def symmetricfactored( double[::1,:] A, int[::1] ipiv, double[::1] b ):
    """def symmetricfactored( double[::1,:] A, int[::1] ipiv, double[::1] b ):

Solve a symmetric system using an already factored A and its pivot array.

Caveat:
    The solution is computed by LAPACK's DSYTRS, which uses level 2 BLAS, so the performance is not optimal.

    The solution may slightly differ from that returned by symmetric(), because DSYSV uses DSYTRS2
    (with level 3 BLAS) internally.

    This difference is because cython_lapack does not expose DSYTRS2.

A : symmetric matrix as Fortran-contiguous rank-2 arraylike, shape (n,n)
    in  : the UDUT factorization of A
    out : unchanged

ipiv : pivot array as rank-1 arraylike, shape (n,), dtype np.int32
    in  : the pivot array (see symmetricfactor())
    out : unchanged

b : rank-1 arraylike, shape (n,)
    in  : RHS
    out : solution x to  A*x = b
"""
    cdef int n=A.shape[0]
    with nogil:
        symmetricfactored_c( &A[0,0], &ipiv[0], &b[0], n )

cdef int symmetricfactored_c( double* A, int* ipiv, double* b, int n ) nogil except -1:
    return msymmetricfactored_c( A, ipiv, b, n, 1 )


def symmetrics( double[::1,:] A, double[::1,:] b ):
    """def symmetrics( double[::1,:] A, double[::1,:] b ):

Like symmetric(); single-threaded for multiple RHS.

Uses LAPACK's multiple RHS functionality. Factorizes A only once.

A : shape (n, n)
    in  : matrix A
    out : destroyed (overwritten)
b : shape (n, nrhs)
    in  : RHSs
    out : solutions x to  A*x = b  for each RHS

"""
    cdef int n=b.shape[0], nrhs=b.shape[1]
    with nogil:
        symmetrics_c( &A[0,0], &b[0,0], n, nrhs )

cdef int symmetrics_c( double* A, double* b, int n, int nrhs ) nogil except -1:
    cdef int info
    cdef char uplo='U'

    # query optimal work size (will be written at &worksize[0])
    cdef double worksize
    cdef int lwork=-1
    dsysv( &uplo, &n, &nrhs, A, &n, <int*>0, b, &n, &worksize, &lwork, &info )
    lwork  = <int>worksize
    cdef double* p_work = <double*>malloc( lwork*sizeof(double) )

    # solve (on exit, b is overwritten by the solution)
    #
    # http://www.netlib.org/lapack/lapack-3.1.1/html/dsysv.f.html
    # UPLO, N, NRHS, A, LDA, IPIV, B, LDB, WORK, LWORK, INFO
    #
    cdef int* p_ipiv = <int*>malloc( n*sizeof(int) )
    dsysv( &uplo, &n, &nrhs, A, &n, p_ipiv, b, &n, p_work, &lwork, &info )

    free( <void*>p_ipiv )
    free( <void*>p_work )

    return 0


def symmetricsp( double[::1,:] A, double[::1,:] b, int ntasks ):
    """def symmetricsp( double[::1,:] A, double[::1,:] b, int ntasks ):

Like symmetric(); multi-threaded for multiple RHS.

Divides into parallel tasks, then uses LAPACK's multiple RHS functionality.
Factorizes A once per thread (this is done internally by LAPACK's DSYSV).

A : shape (n, n)
    in  : matrix A
    out : unchanged (temporary copies are made for factorization, since ntasks copies are needed)
b : shape (n, nrhs)
    in  : RHSs
    out : solutions x to  A*x = b  for each RHS
ntasks : number of threads for OpenMP
"""
    cdef int n=b.shape[0], nrhs=b.shape[1]
    with nogil:
        symmetricsp_c( &A[0,0], &b[0,0], n, nrhs, ntasks )

cdef int symmetricsp_c( double* A, double* b, int n, int nrhs, int ntasks ) nogil except -1:
    cdef int info, nelems=n*n  # nelems = total number of elements in A
    cdef char uplo='U'

    # Compute block sizes and start indices
    #
    cdef int* blocksizes = <int*>malloc( ntasks*sizeof(int) )
    cdef int* baseidxs   = <int*>malloc( ntasks*sizeof(int) )
    distribute_items_c( nrhs, ntasks, blocksizes, baseidxs )

    # The optimal work size is n*NB, where NB is the optimal block size returned by ILAENV for DSYTRF.
    # (Look at the source code of DSYSV.)
    #
    # Thus the optimal work size only depends on LAPACK configuration and n, and hence will be the same for all our problem instances.
    #
    # query optimal work size (using lwork=-1), will be written at &worksize[0]
    cdef double worksize
    cdef int lwork=-1
    dsysv( &uplo, &n, &nrhs, A, &n, <int*>0, b, &n, &worksize, &lwork, &info )
    lwork  = <int>worksize
    cdef double* p_work = <double*>malloc( ntasks*lwork*sizeof(double) )

    # parallel solve
    #
    cdef int* p_ipivs = <int*>malloc( ntasks*n*sizeof(int) )  # each task needs n ints for pivot data
    cdef double* As   = <double*>malloc( ntasks*nelems*sizeof(double) )  # each task needs a copy of the original A since the input "A" will be overwritten by dsysv
    cdef int k
    for k in cython.parallel.prange(ntasks, num_threads=ntasks):
        # make a copy of A for each task, preserving the memory layout (Fortran contiguous)
        copysymmu_c( &As[k*nelems], A, n, n )

        # - nrhs = blocksizes[k]
        # - start index in b[] = baseidxs[k]; in b, each problem is n elements long
        # - "k"th copy of matrix (A), "k"th pivot array
        # - also "k"th work storage, since there are only ntasks iterations in the loop and each runs in its own thread
        dsysv( &uplo, &n, &blocksizes[k], &As[k*nelems], &n, &p_ipivs[k*n], &b[baseidxs[k]*n], &n, &p_work[k*lwork], &lwork, &info )

    free( <void*>As )
    free( <void*>baseidxs )
    free( <void*>blocksizes )
    free( <void*>p_ipivs )
    free( <void*>p_work )

    return 0


def msymmetric( double[::1,:,:] A, double[::1,:] b ):
    """def msymmetric( double[::1,:,:] A, double[::1,:] b ):

Like symmetric(), single-threaded for multiple LHS, one RHS per each LHS.

Uses a loop at the C level in Cython.

A : shape (n, n, nlhs)
    in  : matrices A
    out : destroyed (overwritten)
b : shape (n, nlhs)
    in  : RHSs (one for each LHS)
    out : solutions x to  A*x = b  for each RHS
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        msymmetric_c( &A[0,0,0], &b[0,0], n, nlhs )

cdef int msymmetric_c( double* A, double* b, int n, int nlhs ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem
    cdef char uplo='U'

    # The optimal work size is n*NB, where NB is the optimal block size returned by ILAENV for DSYTRF.
    # (Look at the source code of DSYSV.)
    #
    # Thus the optimal work size only depends on LAPACK configuration and n, and hence will be the same for all our problem instances.
    #
    # query optimal work size (will be written at &worksize[0])
    cdef double worksize
    cdef int lwork=-1
    dsysv( &uplo, &n, &nrhs, A, &n, <int*>0, b, &n, &worksize, &lwork, &info )
    lwork  = <int>worksize
    cdef double* p_work = <double*>malloc( lwork*sizeof(double) )

    # solve
    cdef int* p_ipivs = <int*>malloc( nlhs*n*sizeof(int) )
    cdef int k
    for k in range(nlhs):
        dsysv( &uplo, &n, &nrhs, &A[k*nelems], &n, &p_ipivs[k*n], &b[k*n], &n, p_work, &lwork, &info )

    free( <void*>p_ipivs )
    free( <void*>p_work )

    return 0


def msymmetricp( double[::1,:,:] A, double[::1,:] b, int ntasks ):
    """def msymmetricp( double[::1,:,:] A, double[::1,:] b, int ntasks ):

Like symmetric(), multi-threaded for multiple LHS, one RHS per each LHS.

Uses a parallel loop at the C level in Cython.

A : shape (n, n, nlhs)
    in  : matrices A
    out : destroyed (overwritten)
b : shape (n, nlhs)
    in  : RHSs (one for each LHS)
    out : solutions x to  A*x = b  for each RHS
ntasks : number of threads for OpenMP
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        msymmetricp_c( &A[0,0,0], &b[0,0], n, nlhs, ntasks )

cdef int msymmetricp_c( double* A, double* b, int n, int nlhs, int ntasks ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem
    cdef char uplo='U'

    # query optimal work size (will be written at &worksize[0])
    cdef double worksize
    cdef int lwork=-1
    dsysv( &uplo, &n, &nrhs, A, &n, <int*>0, b, &n, &worksize, &lwork, &info )
    lwork  = <int>worksize
    cdef double* p_work = <double*>malloc( ntasks*lwork*sizeof(double) )

    # solve
    cdef int* p_ipivs = <int*>malloc( nlhs*n*sizeof(int) )
    cdef int k, tid
    for k in cython.parallel.prange(nlhs, num_threads=ntasks):
        # careful:
        #    - "k"th problem (A), RHS (b), pivot array
        #    - "tid"th work storage, since we need only one per thread (in the general case, nlhs >> ntasks)
        tid = openmp.omp_get_thread_num()
        dsysv( &uplo, &n, &nrhs, &A[k*nelems], &n, &p_ipivs[k*n], &b[k*n], &n, &p_work[tid*lwork], &lwork, &info )

    free( <void*>p_work )
    free( <void*>p_ipivs )

    return 0


def msymmetricfactor( double[::1,:,:] A, int[::1,:] ipiv ):
    """def msymmetricfactor( double[::1,:,:] A, int[::1,:] ipiv ):

Compute the UDUT factorization only. Multiple symmetric LHS, single-threaded.

A : shape (n, n, nlhs)
    in  : symmetric matrices A (only upper triangle of each is used)
    out : overwritten by UDUT factorization of each A as returned by DSYTRF
ipiv : shape (n, nlhs), must have been allocated by caller (dtype=np.int32, order='F')
    in  : not read
    out : pivot information
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        msymmetricfactor_c( &A[0,0,0], &ipiv[0,0], n, nlhs )

cdef int msymmetricfactor_c( double* A, int* ipiv, int n, int nlhs ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem (and actually unused since we only use DSYSV to query work size)
    cdef char uplo='U'

    # query optimal work size (will be written at &worksize[0]).
    #
    cdef double worksize
    cdef int lwork=-1
    dsytrf( &uplo, &n, A, &n, ipiv, &worksize, &lwork, &info )
    lwork  = <int>worksize
    cdef double* p_work = <double*>malloc( lwork*sizeof(double) )

    cdef int k
    for k in range(nlhs):
        dsytrf( &uplo, &n, &A[k*nelems], &n, &ipiv[k*n], p_work, &lwork, &info )

    free( <void*>p_work )

    return 0


def msymmetricfactored( double[::1,:,:] A, int[::1,:] ipiv, double[::1,:] b ):
    """def msymmetricfactored( double[::1,:,:] A, int[::1,:] ipiv, double[::1,:] b ):

Solve multiple symmetric systems using already factored A and their pivot arrays. Single-threaded.

Caveat:
    The solution is computed by LAPACK's DSYTRS, which uses level 2 BLAS, so the performance is not optimal.

    The solution may slightly differ from that returned by symmetric(), because DSYSV uses DSYTRS2
    (with level 3 BLAS) internally.

    This difference is because cython_lapack does not expose DSYTRS2.

A: shape (n, n, nlhs)
    in  : UDUT factorization of each A
    out : unchanged
ipiv: shape (n, nlhs)
    in  : pivot information
    out : unchanged
b: shape (n, nlhs)
    in  : RHSs (one for each LHS)
    out : solutions x to  A*x = b  for each RHS
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        msymmetricfactored_c( &A[0,0,0], &ipiv[0,0], &b[0,0], n, nlhs )

cdef int msymmetricfactored_c( double* A, int* ipiv, double* b, int n, int nlhs ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem
    cdef char uplo='U'

    cdef int k
    for k in range(nlhs):
        # DSYTRS uses level 2 BLAS, so the performance is not optimal. There is DSYTRS2 (using level 3 BLAS) in newer LAPACKs, but SciPy doesn't currently export that.
        dsytrs( &uplo, &n, &nrhs, &A[k*nelems], &n, &ipiv[k*n], &b[k*n], &n, &info )

    return 0


def msymmetricfactorp( double[::1,:,:] A, int[::1,:] ipiv, int ntasks ):
    """def msymmetricfactorp( double[::1,:,:] A, int[::1,:] ipiv, int ntasks ):

Compute the UDUT factorization only. Multiple symmetric LHS, multi-threaded.

A : shape (n, n, nlhs)
    in  : symmetric matrices A (only upper triangle of each is used)
    out : overwritten by UDUT factorization of each A as returned by DSYTRF
ipiv : shape (n, nlhs), must have been allocated by caller (dtype=np.int32, order='F')
    in  : not read
    out : pivot information
ntasks : number of threads for OpenMP
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        msymmetricfactorp_c( &A[0,0,0], &ipiv[0,0], n, nlhs, ntasks )

cdef int msymmetricfactorp_c( double* A, int* ipiv, int n, int nlhs, int ntasks ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem (and actually unused since we only use DSYSV to query work size)
    cdef char uplo='U'

    # query optimal work size (will be written at &worksize[0])
    cdef double worksize
    cdef int lwork=-1
    dsytrf( &uplo, &n, A, &n, ipiv, &worksize, &lwork, &info )
    lwork  = <int>worksize
    cdef double* p_work = <double*>malloc( ntasks*lwork*sizeof(double) )

    # solve
    cdef int k, tid
    for k in cython.parallel.prange(nlhs, num_threads=ntasks):
        # careful:
        #    - "k"th problem (A), RHS (b), pivot array
        #    - "tid"th work storage, since we need only one per thread (in the general case, nlhs >> ntasks)
        tid = openmp.omp_get_thread_num()
        dsytrf( &uplo, &n, &A[k*nelems], &n, &ipiv[k*n], &p_work[tid*lwork], &lwork, &info )

    free( <void*>p_work )

    return 0


def msymmetricfactoredp( double[::1,:,:] A, int[::1,:] ipiv, double[::1,:] b, int ntasks ):
    """def msymmetricfactoredp( double[::1,:,:] A, int[::1,:] ipiv, double[::1,:] b, int ntasks ):

Solve multiple symmetric systems using already factored A and their pivot arrays. Multi-threaded.

Caveat:
    The solution is computed by LAPACK's DSYTRS, which uses level 2 BLAS, so the performance is not optimal.

    The solution may slightly differ from that returned by symmetric(), because DSYSV uses DSYTRS2
    (with level 3 BLAS) internally.

    This difference is because cython_lapack does not expose DSYTRS2.

A: shape (n, n, nlhs)
    in  : UDUT factorization of each A
    out : unchanged
ipiv: shape (n, nlhs)
    in  : pivot information
    out : unchanged
b: shape (n, nlhs)
    in  : RHSs (one for each LHS)
    out : solutions x to  A*x = b  for each RHS
ntasks : number of threads for OpenMP
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        msymmetricfactoredp_c( &A[0,0,0], &ipiv[0,0], &b[0,0], n, nlhs, ntasks )

cdef int msymmetricfactoredp_c( double* A, int* ipiv, double* b, int n, int nlhs, int ntasks ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem
    cdef char uplo='U'
    cdef int k

    for k in cython.parallel.prange(nlhs, num_threads=ntasks):
        # DSYTRS uses level 2 BLAS, so the performance is not optimal. There is DSYTRS2 (using level 3 BLAS) in newer LAPACKs, but SciPy doesn't currently export that.
        dsytrs( &uplo, &n, &nrhs, &A[k*nelems], &n, &ipiv[k*n], &b[k*n], &n, &info )

    return 0


##############################################################################################################
# General matrices
##############################################################################################################

def general2x2( double[::1,:] A, double[::1] b ):
    """def general2x2( double[::1,:] A, double[::1] b ):

Solve a general 2x2 system.

This is provided for completeness; the special case of a 2x2 matrix is fastest to just compute directly.

A : general 2x2 matrix as Fortran-contiguous rank-2 arraylike, shape (2,2)
b : rank-1 arraylike, shape (2,)
    in  : RHS
    out : solution x to  A*x = b
"""
    general2x2_c( &A[0,0], &b[0] )

cdef int general2x2_c( double* A, double* b ) nogil except -1:
    # Ainv = 1/Adet * [ [A22, -A12], [-A21, A11] ]
    # x = Ainv*b

    # each of these gets two accesses
    cdef double a00 = A[0]
    cdef double a10 = A[1]  # A[1,0] in Fortran-contiguous storage
    cdef double a01 = A[2]  # A[0,1] in Fortran-contiguous storage
    cdef double a11 = A[3]
    cdef double b0  = b[0]
    cdef double b1  = b[1]

    # solve and overwrite b with solution
    cdef double dm1 = 1. / ( a00*a11 - a01*a10 )
    b[0] = dm1 * ( a11*b0 - a01*b1 )
    b[1] = dm1 * ( a00*b1 - a10*b0 )

    return 0


def general( double[::1,:] A, double[::1] b ):
    """def general( double[::1,:] A, double[::1] b ):

Solve a general system.

A : general matrix as Fortran-contiguous rank-2 arraylike, shape (n,n)
    in  : matrix A
    out : destroyed (overwritten)
b : rank-1 arraylike, shape (n,)
    in  : RHS
    out : solution x to  A*x = b
"""
    cdef int n=b.shape[0]
    with nogil:
        generals_c( &A[0,0], &b[0], n, 1 )

cdef int general_c( double* A, double* b, int n ) nogil except -1:
    return generals_c( A, b, n, 1 )


def generalfactor( double[::1,:] A ):
    """def generalfactor( double[::1,:] A ):

Compute the LU factorization only.

A : symmetric matrix as Fortran-contiguous rank-2 arraylike, shape (n,n)
    in  : matrix A
    out : the LU factorization of A, as returned by DGETRF

Return value:
    pivot array (rank-1, shape (n,), dtype np.int32). This is needed for the solve step.
"""
    cdef int n=A.shape[0]
    cdef int[::1] ipiv = np.empty( (n,), dtype=np.int32 )
    with nogil:
        generalfactor_c( &A[0,0], &ipiv[0], n )
    return ipiv

cdef int generalfactor_c( double* A, int* ipiv, int n ) nogil except -1:
    return mgeneralfactor_c( A, ipiv, n, 1 )

# Solve a linear equation system using an already factored A and pivot data.
#
# b is overwritten with the solution.
#
def generalfactored( double[::1,:] A, int[::1] ipiv, double[::1] b ):
    """def generalfactored( double[::1,:] A, int[::1] ipiv, double[::1] b ):

Solve a general system using an already factored A and its pivot array.

A : general matrix as Fortran-contiguous rank-2 arraylike, shape (n,n)
    in  : the LU factorization of A
    out : unchanged

ipiv : pivot array as rank-1 arraylike, shape (n,), dtype np.int32
    in  : the pivot array (see generalfactor())
    out : unchanged

b : rank-1 arraylike, shape (n,)
    in  : RHS
    out : solution x to  A*x = b
"""
    cdef int n=A.shape[0]
    with nogil:
        generalfactored_c( &A[0,0], &ipiv[0], &b[0], n )

cdef int generalfactored_c( double* A, int* ipiv, double* b, int n ) nogil except -1:
    return mgeneralfactored_c( A, ipiv, b, n, 1 )


def generals( double[::1,:] A, double[::1,:] b ):
    """def generals( double[::1,:] A, double[::1,:] b ):

Like general(); single-threaded for multiple RHS.

Uses LAPACK's multiple RHS functionality. Factorizes A only once.

A : shape (n, n)
    in  : matrix A
    out : destroyed (overwritten)
b : shape (n, nrhs)
    in  : RHSs
    out : solutions x to  A*x = b  for each RHS

"""
    cdef int n=A.shape[0], nrhs=b.shape[1]
    with nogil:
        generals_c( &A[0,0], &b[0,0], n, nrhs )

cdef int generals_c( double* A, double* b, int n, int nrhs ) nogil except -1:
    cdef int info
    cdef int* p_ipiv = <int*>malloc( n*sizeof(int) )

    # solve (on exit, b is overwritten by the solution)
    #
    # http://www.netlib.org/lapack/lapack-3.1.1/html/dgesv.f.html
    # N, NRHS, A, LDA, IPIV, B, LDB, INFO
    #
    dgesv( &n, &nrhs, A, &n, p_ipiv, b, &n, &info )

    free( <void*>p_ipiv )

    return 0


def generalsp( double[::1,:] A, double[::1,:] b, int ntasks ):
    """def generalsp( double[::1,:] A, double[::1,:] b, int ntasks ):

Like general(); multi-threaded for multiple RHS.

Divides into parallel tasks, then uses LAPACK's multiple RHS functionality.
Factorizes A once per thread (this is done internally by LAPACK's DGESV).

A : shape (n, n)
    in  : matrix A
    out : unchanged (temporary copies are made for factorization, since ntasks copies are needed)
b : shape (n, nrhs)
    in  : RHSs
    out : solutions x to  A*x = b  for each RHS
ntasks : number of threads for OpenMP
"""
    cdef int n=A.shape[0], nrhs=b.shape[1]
    with nogil:
        generalsp_c( &A[0,0], &b[0,0], n, nrhs, ntasks )

cdef int generalsp_c( double* A, double* b, int n, int nrhs, int ntasks ) nogil except -1:
    cdef int info, nelems=n*n  # nelems = total number of elements in A

    # Compute block sizes and start indices
    #
    cdef int*blocksizes = <int*>malloc( ntasks*sizeof(int) )
    cdef int*baseidxs   = <int*>malloc( ntasks*sizeof(int) )
    distribute_items_c( nrhs, ntasks, blocksizes, baseidxs )

    # solve
    #
    cdef int* p_ipivs = <int*>malloc( ntasks*n*sizeof(int) )  # each task needs n ints
    cdef double* As   = <double*>malloc( ntasks*nelems*sizeof(double) )  # each task needs a copy of the original A since the input "A" will be overwritten by dgesv
    cdef int k
    for k in cython.parallel.prange(ntasks, num_threads=ntasks):
        # make a copy of A for each task, preserving the memory layout
        copygeneral_c( &As[k*nelems], A, n, n )

        #    - nrhs = blocksizes[k]
        #    - start index in b[] = baseidxs[k]; in b, each problem is n elements long
        #    - "k"th copy of matrix (A), "k"th pivot array
        dgesv( &n, &blocksizes[k], &As[k*nelems], &n, &p_ipivs[k*n], &b[baseidxs[k]*n], &n, &info )

    free( <void*>As )
    free( <void*>p_ipivs )
    free( <void*>baseidxs )
    free( <void*>blocksizes )

    return 0


def mgeneral( double[::1,:,:] A, double[::1,:] b ):
    """def mgeneral( double[::1,:,:] A, double[::1,:] b ):

Like general(), single-threaded for multiple LHS, one RHS per each LHS.

Uses a loop at the C level in Cython.

A : shape (n, n, nlhs)
    in  : matrices A
    out : destroyed (overwritten)
b : shape (n, nlhs)
    in  : RHSs (one for each LHS)
    out : solutions x to  A*x = b  for each RHS
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        mgeneral_c( &A[0,0,0], &b[0,0], n, nlhs )

cdef int mgeneral_c( double* A, double* b, int n, int nlhs ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem

    cdef int* p_ipivs = <int*>malloc( nlhs*n*sizeof(int) )
    cdef int k
    for k in range(nlhs):
        dgesv( &n, &nrhs, &A[k*nelems], &n, &p_ipivs[k*n], &b[k*n], &n, &info )
    free( <void*>p_ipivs )

    return 0


def mgeneralp( double[::1,:,:] A, double[::1,:] b, int ntasks ):
    """def mgeneralp( double[::1,:,:] A, double[::1,:] b, int ntasks ):

Like general(), multi-threaded for multiple LHS, one RHS per each LHS.

Uses a parallel loop at the C level in Cython.

A : shape (n, n, nlhs)
    in  : matrices A
    out : destroyed (overwritten)
b : shape (n, nlhs)
    in  : RHSs (one for each LHS)
    out : solutions x to  A*x = b  for each RHS
ntasks : number of threads for OpenMP
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        mgeneralp_c( &A[0,0,0], &b[0,0], n, nlhs, ntasks )

cdef int mgeneralp_c( double* A, double* b, int n, int nlhs, int ntasks ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem

    cdef int* p_ipivs = <int*>malloc( nlhs*n*sizeof(int) )
    cdef int k
    for k in cython.parallel.prange(nlhs, num_threads=ntasks):
        dgesv( &n, &nrhs, &A[k*nelems], &n, &p_ipivs[k*n], &b[k*n], &n, &info )
    free( <void*>p_ipivs )

    return 0


def mgeneralfactor( double[::1,:,:] A, int[::1,:] ipiv ):
    """def mgeneralfactor( double[::1,:,:] A, int[::1,:] ipiv ):

Compute the LU factorization only. Multiple general LHS, single-threaded.

A : shape (n, n, nlhs)
    in  : general matrices A
    out : overwritten by LU factorization of each A as returned by DGETRF
ipiv : shape (n, nlhs), must have been allocated by caller (dtype=np.int32, order='F')
    in  : not read
    out : pivot information
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        mgeneralfactor_c( &A[0,0,0], &ipiv[0,0], n, nlhs )

cdef int mgeneralfactor_c( double* A, int* ipiv, int n, int nlhs ) nogil except -1:
    cdef int info, nelems=n*n

    cdef int k
    for k in range(nlhs):
        dgetrf( &n, &n, &A[k*nelems], &n, &ipiv[k*n], &info )

    return 0


def mgeneralfactored( double[::1,:,:] A, int[::1,:] ipiv, double[::1,:] b ):
    """def mgeneralfactored( double[::1,:,:] A, int[::1,:] ipiv, double[::1,:] b ):

Solve multiple general systems using already factored A and their pivot arrays. Single-threaded.

A: shape (n, n, nlhs)
    in  : LU factorization of each A
    out : unchanged
ipiv: shape (n, nlhs)
    in  : pivot information
    out : unchanged
b: shape (n, nlhs)
    in  : RHSs (one for each LHS)
    out : solutions x to  A*x = b  for each RHS
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        mgeneralfactored_c( &A[0,0,0], &ipiv[0,0], &b[0,0], n, nlhs )

cdef int mgeneralfactored_c( double* A, int* ipiv, double* b, int n, int nlhs ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem
    cdef char transpose='N'

    cdef int k
    for k in range(nlhs):
        dgetrs( &transpose, &n, &nrhs, &A[k*nelems], &n, &ipiv[k*n], &b[k*n], &n, &info )

    return 0


def mgeneralfactorp( double[::1,:,:] A, int[::1,:] ipiv, int ntasks ):
    """def mgeneralfactorp( double[::1,:,:] A, int[::1,:] ipiv, int ntasks ):

Compute the LU factorization only. Multiple general LHS, multi-threaded.

A : shape (n, n, nlhs)
    in  : general matrices A
    out : overwritten by LU factorization of each A as returned by DGETRF
ipiv : shape (n, nlhs), must have been allocated by caller (dtype=np.int32, order='F')
    in  : not read
    out : pivot information
ntasks : number of threads for OpenMP
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        mgeneralfactorp_c( &A[0,0,0], &ipiv[0,0], n, nlhs, ntasks )

cdef int mgeneralfactorp_c( double* A, int* ipiv, int n, int nlhs, int ntasks ) nogil except -1:
    cdef int info, nelems=n*n

    cdef int k
    for k in cython.parallel.prange(nlhs, num_threads=ntasks):
        dgetrf( &n, &n, &A[k*nelems], &n, &ipiv[k*n], &info )

    return 0


def mgeneralfactoredp( double[::1,:,:] A, int[::1,:] ipiv, double[::1,:] b, int ntasks ):
    """def mgeneralfactoredp( double[::1,:,:] A, int[::1,:] ipiv, double[::1,:] b, int ntasks ):

Solve multiple general systems using already factored A and their pivot arrays. Multi-threaded.

A: shape (n, n, nlhs)
    in  : LU factorization of each A
    out : unchanged
ipiv: shape (n, nlhs)
    in  : pivot information
    out : unchanged
b: shape (n, nlhs)
    in  : RHSs (one for each LHS)
    out : solutions x to  A*x = b  for each RHS
ntasks : number of threads for OpenMP
"""
    cdef int n=A.shape[0], nlhs=A.shape[2]
    with nogil:
        mgeneralfactoredp_c( &A[0,0,0], &ipiv[0,0], &b[0,0], n, nlhs, ntasks )

cdef int mgeneralfactoredp_c( double* A, int* ipiv, double* b, int n, int nlhs, int ntasks ) nogil except -1:
    cdef int nrhs=1, info, nelems=n*n  # here nrhs is per problem
    cdef char transpose='N'

    cdef int k
    for k in cython.parallel.prange(nlhs, num_threads=ntasks):
        dgetrs( &transpose, &n, &nrhs, &A[k*nelems], &n, &ipiv[k*n], &b[k*n], &n, &info )

    return 0


##############################################################################################################
# Other stuff
##############################################################################################################

def svd( double[::1,:] A ):
    """def svd( double[::1,:] A ):

Singular value decomposition of general A.

Currently this gets the singular values only, ignoring the orthogonal matrices U and V.
This is mainly useful for estimating the 2-norm condition number of A, as S[0] / S[-1].

A : general matrix as Fortran-contiguous rank-2 arraylike, shape (n,n)
    in  : matrix A
    out : destroyed (overwritten)

Return value:
    S, a rank-1 array of shape (n,) containing the singular values, sorted so that S(i) >= S(i+1).

"""
    cdef int m = A.shape[0]  # rows
    cdef int n = A.shape[1]  # columns
    cdef int minmn = cimin(m,n)
    cdef double[::1] S = np.empty( (minmn,), dtype=np.float64 )

    with nogil:
        svd_c( &A[0,0], m, n, &S[0] )

    return S

cdef int svd_c( double* A, int m, int n, double* S ) nogil except -1:
    cdef int info, ldu=1, ldvt=1  # ldu and ldvt must be >= 1 even though they are not used if jobu='N' and jobvt='N'.
    cdef char jobu  = 'N'
    cdef char jobvt = 'N'

    cdef double worksize
    cdef int lwork=-1
    # LDA(A) = m
    # dgesvd (JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO)
    dgesvd( &jobu, &jobvt, &m, &n, A, &m, S, <double*>0, &ldu, <double*>0, &ldvt, &worksize, &lwork, &info )
    lwork  = <int>worksize
    cdef double* p_work = <double*>malloc( lwork*sizeof(double) )

    # LDA(A) = n
    dgesvd( &jobu, &jobvt, &m, &n, A, &m, S, <double*>0, &ldu, <double*>0, &ldvt, p_work, &lwork, &info )

    free( <void*>p_work )

    return 0

