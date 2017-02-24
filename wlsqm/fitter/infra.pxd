# -*- coding: utf-8 -*-
#
# WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.
#
# Centralized memory allocation infrastructure.
#
# The implementation uses C-style object-oriented programming, with structs and
# class name prefixed methods using an explicit self pointer argument.
#
# JJ 2016-11-30

#################################################
# Helper functions
#################################################

cdef int number_of_dofs( int dimension, int order ) nogil
cdef int number_of_reduced_dofs( int n, long long mask ) nogil
cdef int remap( int* o2r, int* r2o, int n, long long mask ) nogil

#################################################
# class Allocator:
#################################################

cdef int ALLOC_MODE_PASSTHROUGH   # pass each call through to C malloc/free
cdef int ALLOC_MODE_ONEBIGBUFFER  # pre-allocate one big buffer to fit everything in

cdef struct Allocator:
    int mode       # operation mode, see constants ALLOC_MODE_*
    void* buffer   # start address of all storage
    int size_total # buffer size, bytes
    void* p        # first currently unused address in buffer
    int size_used  # bytes used up to now

cdef Allocator* Allocator_new( int mode, int total_size_bytes ) nogil except <Allocator*>0
cdef void* Allocator_malloc( Allocator* self, int size_bytes ) nogil
cdef void Allocator_free( Allocator* self, void* p ) nogil
cdef int Allocator_size_remaining( Allocator* self ) nogil
cdef void Allocator_del( Allocator* self ) nogil

#################################################
# class CaseManager:
#################################################

# Sizes needed for the various arrays in Case, as bytes.
#
# This is really just a struct; no methods.
#
cdef struct BufferSizes:
    int o2r        # DOF mapping original --> reduced
    int r2o        # DOF mapping reduced  --> original

    int c          # distance matrix
    int w          # weights

    int A          # problem matrix / its packed LU factorization
    int row_scale  # row scaling factors for A (needed by solver to scale RHS)
    int col_scale  # column scaling factors for A (needed by solver to scale solution)
    int ipiv       # pivot information of LU factored A (needed by solver)

    int fi         # coefficients of the fit; essentially, the function value and derivatives at the origin of the fit
    int fi2        # work space for coefficients for interpolating derivatives of the model (wlsqm.fitter.interp) to a general point

    int wrk        # solver work space for RHS (or zero in managed mode)

    int fk_tmp     # work space for iterative refinement (or zero in managed mode), remaining error at each point xk
    int fi_tmp     # work space for iterative refinement (or zero in managed mode), coefficients of error reduction fit

    int total      # sum of all the above

# Infra class for multiple problem instances having the same dimension, order, knowns mask and flags (do_sens, iterative).
#
# This centralizes the memory allocation (to avoid unnecessary fragmentation)
# when multiple problem instances are solved at one go.
#
# This class is only intended to be used from a Python thread.
#
cdef struct CaseManager:
    # parallel processing
    #
    # "per-task arrays": one work space per task, independent of the number of problem instances (cases).
    #
    int ntasks
    double** wrks     # array of work spaces for RHS
    double** fk_tmps  # array of work spaces for iterative refinement, remaining error at each point xk
    double** fi_tmps  # array of work spaces for iterative refinement, coefficients of error reduction fit

    # managed cases
    #
    Case** cases      # array to store the Case pointers
    int max_cases     # array capacity
    int ncases        # currently used capacity

    int bytes_needed  # total memory required for storing the work spaces and the arrays allocated by the Case objects

    # data common to all managed cases
    #
    Allocator* mal
    int dimension
    int do_sens
    int iterative

cdef CaseManager* CaseManager_new( int dimension, int do_sens, int iterative, int max_cases, int ntasks ) nogil except <CaseManager*>0
cdef int CaseManager_add( CaseManager* self, Case* case ) nogil except -1
cdef int CaseManager_commit( CaseManager* self ) nogil except -1
#cdef int CaseManager_allocate( CaseManager* self ) nogil except -1  # private (not exported from the module)
#cdef void CaseManager_deallocate( CaseManager* self ) nogil         # private
cdef void CaseManager_del( CaseManager* self ) nogil

#################################################
# class Case:
#################################################

# This class gathers some metadata and centralizes memory management for one problem instance.
#
# We use the above custom allocator to allocate all needed memory in one big block.
#
# A centralized mode is available (see the optional constructor parameter "cases")
# to centralize memory allocation for a set of cases.
#
# TODO: refactor: make_c_nD(), make_A(), preprocess_A(), solve() now look a lot like methods of Case (the first parameter is a Case*, and its member variables are used extensively).
# TODO: could also store the point xi and other relevant stuff. (OTOH, currently no actual use case that needs them)
#
cdef struct Case:
    # infra
    int have_manager  # 1 = has a CaseManager, using its allocator
                      # 0 = no CaseManager, create an allocator locally
    CaseManager* manager
    Allocator* mal  # custom memory allocator

    # case metadata
    int dimension         # number of space dimensions
    int order             # degree of polynomial to be fitted
    long long knowns      # knowns bitmask
    int weighting_method  # weighting: uniform or emphasize center region (see wlsqm.fitter.defs)
    int no                # number of DOFs in original (unreduced) system
    int nr                # number of DOFs in reduced system
    int nk                # number of neighbor points used in fit
    int do_sens           # flag: do sensitivity analysis? (affects memory usage)
    int iterative         # flag: iterative refinement? (affects memory usage)

    # the origin point of the model (needed by certain routines)
    double xi  # 1D, 2D, 3D
    double yi  #     2D, 3D
    double zi  #         3D

    # data pointers

    int geometry_owned    # guest mode support: possible to use o2r,r2o,c,w,A,row_scale,col_scale,ipiv off another Case instance

    # DOF mappings
    int* o2r
    int* r2o

    # low level stuff: "c" matrix, weights
    double* c
    double* w

    # higher-level stuff: "A" matrix
    double* A
    double* row_scale
    double* col_scale
    int* ipiv

    # condition number (for wlsqm.fitter.impl.preprocess_A() debug mode)
    double cond_orig
    double cond_scaled

    # coefficients of the fit
    #
    # This name was chosen because fi[0] is the function value ("f") at the point (xi,yi), hence "i",
    # and the other elements store derivatives at the same point.
    #
    double* fi
    double* fi2  # work space for coefficients for evaluating derivatives of the model (wlsqm.fitter.interp) at a general point

    # RHS work space for solver
    double* wrk

    # work space for iterative fitting algorithm
    double* fk_tmp
    double* fi_tmp

cdef Case* Case_new( int dimension, int order, double xi, double yi, double zi, int nk, long long knowns, int weighting_method, int do_sens, int iterative, CaseManager* manager, Case* host ) nogil except <Case*>0
cdef double* Case_get_wrk( Case* self, int taskid ) nogil
cdef double* Case_get_fk_tmp( Case* self, int taskid ) nogil
cdef double* Case_get_fi_tmp( Case* self, int taskid ) nogil
cdef void Case_make_weights( Case* self, double max_d2 ) nogil  # mainly for use by wlsqm.fitter.impl.make_c_nd()
cdef void Case_set_fi( Case* self, double* fi ) nogil
cdef void Case_get_fi( Case* self, double* out ) nogil
#cdef void Case_determine_sizes( Case* self, BufferSizes* sizes ) nogil  # private
#cdef int Case_allocate( Case* self ) nogil except -1  # private
#cdef void Case_deallocate( Case* self ) nogil  # private
cdef void Case_del( Case* self ) nogil

