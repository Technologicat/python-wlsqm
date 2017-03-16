# -*- coding: utf-8 -*-
#
# Cython interface for lapackdrivers.pyx.
#
# Naming scheme (in shellglob notation):
#   *s = multiple RHS (but with the same LHS for all).
#        These are one-shot deals that reuse the matrix factorization internally.
#        However, the pivot information is not returned, so the matrix A
#        is destroyed (overwritten) during the call.
#
#   m* = multiple LHS (a separate single RHS for each)
#        These simply loop over the problem instances.
#
#   *p = parallel (multi-threaded using OpenMP)
#        These introduce parallel looping over problem instances.
#
#   *factor*   = the routine that factors the matrix and generates pivot information.
#   *factored* = the solver routine that uses the factored matrix and pivot information.
#        These are useful for solving with many RHSs, when all the RHSs
#        are not available at once (e.g. in PDE solvers, timestepping
#        with a mass matrix that remains constant in time).
#
#   *_c = C version without memoryviews (only visible from Cython). Can be more convenient
#         for use in nogil blocks, in cases where the arrays need to be allocated dynamically (with malloc).
#         The purpose of the C version is to avoid the need to acquire the GIL to create a memoryview
#         into a malloc()'d array.
#
#         This .pxd file offers access to only the C versions. To import the corresponding Python versions
#         of the routines (same name, without the _c suffix), import the module normally in Python.
#
#         Note that the Python routines operate on memoryview slices (compatible with np.arrays),
#         so they have slightly different parameters and return values when compared to the C routines.
#
#         Generally, the Python versions will allocate arrays for you, while the C versions expect you
#         to provide pointers to already malloc()'d memory (and explicit sizes).
#
# See the function docstrings and comments in the .pyx source for details.
#
# JJ 2016-11-07

##############################################################################################################
# Helpers
##############################################################################################################

cdef void distribute_items_c( int nitems, int ntasks, int* blocksizes, int* baseidxs ) nogil  # distribute work items across tasks, assuming equal load per item.

cdef void copygeneral_c( double* O, double* I, int nrows, int ncols ) nogil  # copy general square array
cdef void copysymmu_c( double* O, double* I, int nrows, int ncols ) nogil  # copy symmetric square array, upper triangle only

cdef void symmetrize_c( double* A, int nrows, int ncols ) nogil
cdef void msymmetrize_c( double* A, int nrows, int ncols, int nlhs ) nogil
cdef void msymmetrizep_c( double* A, int nrows, int ncols, int nlhs, int ntasks ) nogil

##############################################################################################################
# Preconditioning (scaling)
##############################################################################################################

# Scaling reduces the condition number of A, helping DGESV (general()) to give more correct digits.
#
# The return value is the number of iterations taken; always 1 for non-iterative algorithms.

# helpers
cdef void init_scaling_c( int nrows, int ncols, double* row_scale, double* col_scale ) nogil  # init all scaling factors to 1.0
cdef void apply_scaling_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil  # freeze the scaling by applying it in-place

# simple, fast methods; these destroy the possible symmetry of A
cdef int rescale_columns_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil
cdef int rescale_rows_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil
cdef int rescale_twopass_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil  # scale columns, then rows
cdef int rescale_dgeequ_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil

# symmetry-preserving methods (iterative)
cdef int rescale_ruiz2001_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil
cdef int rescale_scalgm_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil

##############################################################################################################
# Tridiagonal matrices
##############################################################################################################

cpdef int tridiag( double[::1] a, double[::1] b, double[::1] c, double[::1] x ) nogil except -1

##############################################################################################################
# Symmetric matrices
##############################################################################################################

cdef int symmetric2x2_c( double* A, double* b ) nogil except -1

cdef int symmetric_c( double* A, double* b, int n ) nogil except -1
cdef int symmetricfactor_c( double* A, int* ipiv, int n ) nogil except -1
cdef int symmetricfactored_c( double* A, int* ipiv, double* b, int n ) nogil except -1

cdef int symmetrics_c( double* A, double* b, int n, int nrhs ) nogil except -1
cdef int symmetricsp_c( double* A, double* b, int n, int nrhs, int ntasks ) nogil except -1

cdef int msymmetric_c( double* A, double* b, int n, int nlhs ) nogil except -1
cdef int msymmetricp_c( double* A, double* b, int n, int nlhs, int ntasks ) nogil except -1

cdef int msymmetricfactor_c( double* A, int* ipiv, int n, int nlhs ) nogil except -1
cdef int msymmetricfactored_c( double* A, int* ipiv, double* b, int n, int nlhs ) nogil except -1
cdef int msymmetricfactorp_c( double* A, int* ipiv, int n, int nlhs, int ntasks ) nogil except -1
cdef int msymmetricfactoredp_c( double* A, int* ipiv, double* b, int n, int nlhs, int ntasks ) nogil except -1

##############################################################################################################
# General matrices
##############################################################################################################

cdef int general2x2_c( double* A, double* b ) nogil except -1

cdef int general_c( double* A, double* b, int n ) nogil except -1
cdef int generalfactor_c( double* A, int* ipiv, int n ) nogil except -1
cdef int generalfactored_c( double* A, int* ipiv, double* b, int n ) nogil except -1

cdef int generals_c( double* A, double* b, int n, int nrhs ) nogil except -1
cdef int generalsp_c( double* A, double* b, int n, int nrhs, int ntasks ) nogil except -1

cdef int mgeneral_c( double* A, double* b, int n, int nlhs ) nogil except -1
cdef int mgeneralp_c( double* A, double* b, int n, int nlhs, int ntasks ) nogil except -1

cdef int mgeneralfactor_c( double* A, int* ipiv, int n, int nlhs ) nogil except -1
cdef int mgeneralfactored_c( double* A, int* ipiv, double* b, int n, int nlhs ) nogil except -1
cdef int mgeneralfactorp_c( double* A, int* ipiv, int n, int nlhs, int ntasks ) nogil except -1
cdef int mgeneralfactoredp_c( double* A, int* ipiv, double* b, int n, int nlhs, int ntasks ) nogil except -1

##############################################################################################################
# Other stuff
##############################################################################################################

cdef int svd_c( double* A, int m, int n, double* S ) nogil except -1

