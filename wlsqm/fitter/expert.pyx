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
"""WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.

This module contains the Python API for an "expert mode" in the LAPACK sense, with separate prepare and solve stages, and advanced features.

JJ 2016-12-07
"""

from __future__ import division, print_function, absolute_import

from libc.stdlib cimport malloc, free
from libc.math   cimport sqrt

cimport cython.parallel
cimport openmp

from cython cimport view  # for usage, see http://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#specifying-more-general-memory-layouts

cimport wlsqm.utils.ptrwrap as ptrwrap

cimport wlsqm.fitter.defs   as defs    # C constants
cimport wlsqm.fitter.infra  as infra   # centralized memory allocation infrastructure
cimport wlsqm.fitter.impl   as impl    # low-level routines (implementation)
cimport wlsqm.fitter.interp as interp  # model interpolation

# for model interpolation
import numpy as np
import scipy.spatial


####################################################
# Python API - expert mode
####################################################

# Expose this helper to Python; the number of DOFs (in the original unreduced system) is needed to allocate "fi" arrays of the correct size.
#
# As for the other helpers in infra, the reduced DOFs are an internal thing, not visible outside the module.
#
def number_of_dofs( int dimension, int order ):
    """def number_of_dofs( int dimension, int order ):

Return the number of DOFs (degrees of freedom) for given dimension (1,2,3) and fitting order (0,1,2,3,4).

This can be used to help allocate "fi" arrays of the correct size without wasting memory on unused DOFs."""
    return infra.number_of_dofs( dimension, order )


class ExpertSolver:
    """Advanced API / "expert mode" in the LAPACK sense, with separate prepare and solve stages.

Typical usage is:

s = ExpertSolver(...)  # Instantiating the solver creates the necessary internal objects and allocates memory.
s.prepare(...)         # If the geometry stays constant, it is enough to prepare just once; this generates and prepares the problem matrices.
s.solve(...)           # Then it is possible to solve many times with different data (fk); this solves all unknowns with given data fk and current geometry.

This is very useful when one wishes to perform many fits with the same geometry, but each time with new data,
since the expensive prepare step needs to be performed only once.

Note that if the geometry changes (i.e. the points move), their internal xi values must be updated.
This can be done by calling prepare() again to re-generate the matrices using the updated geometry.

Although in this case the prepare step must be performed again, this still has the advantage of re-using the already allocated memory
(avoiding memory fragmentation and saving some processing time), since the other settings do not change.

Once an ExpertSolver is instantiated, the settings given as constructor parameters cannot be changed for that instance!

This implies that even if the points move, their number of neighbors (xk used in the fitting) must not change!

If you need to do that, destroy the old ExpertSolver instance and create a new one. This limitation is due to the way memory allocation is managed.

""" # TODO/FIXME: the "cannot be changed" part about parameters above

    def __init__(self, int dimension, int[::view.generic] nk, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method,
                                      int algorithm=defs.ALGO_BASIC_c, int do_sens=False, int max_iter=10, int ntasks=1, int debug=False, host=None):
        """def __init__(self, int dimension, int[::view.generic] nk, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method,
                                      int algorithm=defs.ALGO_BASIC_c, int do_sens=False, int max_iter=10, int ntasks=1, int debug=False, host=None):

Constructor.

dimension        : in, number of space dimensions (1, 2 or 3)
nk               : in, array of shape (ncases,), the number of neighbor points (the "xk"s) to be used in the fitting, for each problem instance (dtype np.int32)
order            : in, array of shape (ncases,), the order of the surrogate polynomial to be fitted for each problem instance (0,1,2,3 or 4) (dtype np.int32)
knowns           : in, array of shape (ncases,), the knowns bitmask for each problem instance (see wlsqm.fitter.defs) (dtype np.int64)
weighting_method : in, array of shape (ncases,), the weighting method to use for each problem instance; each entry must be one of the constants wlsqm.fitter.defs.WEIGHT_* (dtype np.int32)
ncases           : in, number of problem instances
algorithm        : in, one of the ALGO_* constants in wlsqm.fitter.defs
do_sens          : in, boolean: whether to perform sensitivity analysis. If False, then in the solve() step, "sens" can be None.
max_iter         : in. If algorithm == ALGO_ITERATIVE, the maximum number of refinement iterations to perform. Unused if algorithm == ALGO_BASIC.
ntasks           : in, number of threads to use for computation
debug            : in, boolean. If True, compute the 2-norm condition number of the (scaled) problem matrix for each problem instance (can be slow). If False (default), skip it.
host             : in, ExpertSolver instance to use as host for guest mode. (Default None, guest mode off.)

                       If different datasets to be fitted live on the exact same geometry, this can be used to save both memory and time
                       by siphoning geometry data (problem matrices) off an already prepare()'d ExpertSolver instance.

                       The geometry must be exactly the same in the host ExpertSolver instance as in the ExpertSolver instance being created.
                       Also dimension, nk, order, knowns, weighting_method, debug must be the same.

                       Note that "geometry" includes both xi,yi.zi (point "xi") and the neighbor set (points "xk").
                       This cannot be checked automatically in this constructor, because the geometry is only specified at prepare() time!

                       In guest mode (host is not None), prepare() must still be called for the guest instance, too - it initializes the "xi" data for each Case instance.

                       IMPORTANT: Guest mode directly uses the same physical copy of the data as the host instance, saving memory. Thus when using guest mode,
                       the calling code must make sure that the host instance stays alive at least as long as its guest instances, or hope for a crash.
"""
        # sanity check input
        #
        # (except do_sens and debug, both of which are boolean)
        #
        cdef int ncases = nk.shape[0]
        if order.shape[0] != ncases  or  knowns.shape[0] != ncases  or  weighting_method.shape[0] != ncases:
            raise ValueError("nk, order, knowns and weighting method must have the same length; currently, len(nk)=%d, len(order)=%d, len(knowns)=%d, len(weighting_method)=%d" % (nk.shape[0], order.shape[0], knowns.shape[0], weighting_method.shape[0]))

        if dimension not in [1,2,3]:
            raise ValueError("Dimension must be 1, 2 or 3, got %d" % dimension)

        # int parameters will be assigned to C variables
        #
        if algorithm is None:
            raise ValueError("algorithm cannot be None")
        if do_sens is None:
            raise ValueError("do_sens cannot be None")
        if max_iter is None:
            raise ValueError("max_iter cannot be None")
        if ntasks is None:
            raise ValueError("ntasks cannot be None")
        if debug is None:
            raise ValueError("debug cannot be None")

        cdef int iterative
        if algorithm == defs.ALGO_BASIC_c:
            iterative = 0
        elif algorithm == defs.ALGO_ITERATIVE_c:
            iterative = 1
        else:
            raise ValueError("Unknown algorithm specifier %d; see wlsqm.fitter.defs for valid specifiers ALGO_*" % algorithm)

        if ntasks < 1:
            raise ValueError("ntasks must be >= 1, got %d" % ntasks)

        # guest mode sanity checks
        #
        if host is not None:
            # check that the host has been prepare()'d, otherwise there are no matrices to borrow
            if not host.ready:
                raise RuntimeError("In guest mode, host must be in the ready state (host.prepare() must have been called before creating another ExpertSolver instance in guest mode).")

            if host.ncases != ncases:
                raise RuntimeError( "In guest mode, number of cases (number of elements in nk) must match; got %d, host has %d" % (ncases, host.ncases) )

            if host.dimension != dimension:
                raise ValueError( "In guest mode, dimension must match; got %d, host has %d" % (dimension, host.dimension) )

            if host.debug != debug:
                raise ValueError( "In guest mode, debug flag must match; got %s, host has %s" % ( bool(debug), bool(host.debug) ) )

            # expensive O(n) checks, but still much faster than re-computing the problem matrices

            if (np.asanyarray(host.nk) != np.asanyarray(nk)).any():
                raise ValueError("In guest mode, 'nk' must match element-by-element.")

            if (np.asanyarray(host.order) != np.asanyarray(order)).any():
                raise ValueError("In guest mode, 'order' must match element-by-element.")

            if (np.asanyarray(host.knowns) != np.asanyarray(knowns)).any():
                raise ValueError("In guest mode, 'knowns' must match element-by-element.")

            if (np.asanyarray(host.weighting_method) != np.asanyarray(weighting_method)).any():
                raise ValueError("In guest mode, 'weighting_method' must match element-by-element.")

        # initialize

        self.host  = host

        self.ready = False  # safe to solve()? (problem matrices up to date)

        # save Python-accessible configuration variables for information only (this data goes into the CaseManager, which keeps its own copy)
        #
        self.dimension        = dimension
        self.algorithm        = algorithm
        self.max_iter         = max_iter
        self.ncases           = ncases
        self.do_sens          = do_sens
        self.ntasks           = ntasks
        self.debug            = debug

        # these will be filled in at a later stage
        #
        self.xk               = None
        self.xi               = None
        self.tree             = None

        # save the views, we'll need (some of) them later.
        #
        # (also for information; this data goes into the Case instances and affects memory allocation)
        #
        self.nk               = nk
        self.order            = order
        self.knowns           = knowns
        self.weighting_method = weighting_method

        # Create a CaseManager with the specified settings.
        #
        cdef infra.CaseManager* manager = infra.CaseManager_new( dimension, do_sens, iterative, ncases, ntasks )

        # For guest mode
        #
        cdef ptrwrap.PointerWrapper host_pw
        cdef infra.CaseManager* host_manager = <infra.CaseManager*>0

        # Create the Cases, adding them to the manager.
        #
        cdef double nan = 0./0.  # FIXME: IEEE-754 abuse
        cdef double xi=nan, yi=nan, zi=nan  # pass invalid coordinates for now, we'll fill them in later

        if host is None:  # usual mode of operation (create geometry data locally)
            with nogil:
                for j in range(ncases):  # initialization is not thread-safe, so do it serially regardless of ntasks
                    # this will automatically add the case to the manager, so we don't need to save the returned pointer
                    infra.Case_new( dimension, order[j], xi, yi, zi, nk[j], knowns[j], weighting_method[j], do_sens, iterative, manager, host=<infra.Case*>0 )
                infra.CaseManager_commit( manager )  # done adding cases (allocate memory!)
        else:  # guest mode
            # Get the manager from the host ExpertSolver instance
            host_pw      = <ptrwrap.PointerWrapper>(host.manager_pw)
            host_manager = <infra.CaseManager*>(host_pw.ptr)

            # exact same geometry means that our case j should get its geometry from host_manager.cases[j]
            with nogil:
                for j in range(ncases):
                    infra.Case_new( dimension, order[j], xi, yi, zi, nk[j], knowns[j], weighting_method[j], do_sens, iterative, manager, host_manager.cases[j] )
                infra.CaseManager_commit( manager )  # done adding cases (allocate memory!)

        # Wrap the CaseManager instance pointer in a PointerWrapper,
        # and save this PointerWrapper instance as a Python object
        # into an instance attribute.
        #
        cdef ptrwrap.PointerWrapper pw = ptrwrap.PointerWrapper()
        pw.set_ptr( <void*>manager )
        self.manager_pw = pw

        if host is not None:
            self.tree  = host.tree  # get the search tree too, if host has one


    # Release the CaseManager instance (automatically releasing also the managed Case instances).
    def __del__(self):
        """def __del__(self):

Destructor.

Releases the memory managed by the custom allocator."""
        # Retrieve the wrapped pointer for the CaseManager.
        #
        # We must first convert the stored Python object into a PointerWrapper instance,
        # after which we can access its "ptr" instance variable.
        #
        # We must use this two-step approach, because with pw *as a Python object*
        # the member lookup for "ptr" will fail (not present in the dictionary).
        # The correct way is to look it up with pw *as a ptrwrap.PointerWrapper instance*
        # at the Cython level.
        #
        cdef ptrwrap.PointerWrapper pw  = <ptrwrap.PointerWrapper>(self.manager_pw)
        cdef infra.CaseManager* manager = <infra.CaseManager*>(pw.ptr)
        with nogil:
            infra.CaseManager_del( manager )


    def memory_used(self):
        """def memory_used(self):

Report amount of memory (in bytes) allocated by the custom allocator.

Return value: tuple (currently_used_bytes, buffer_total_size_bytes)

(Implementation detail: if everything is working correctly, these values are the same - the constructor calls wlsqm.fitter.infra.CaseManager_commit(), which will allocate the memory.
 The buffer is made exactly as large as it needs to be, by calling wlsqm.fitter.infra.Case_determine_sizes() for the managed wlsqm.fitter.infra.Case instances.)
"""
        # get the manager
        cdef ptrwrap.PointerWrapper pw  = <ptrwrap.PointerWrapper>(self.manager_pw)
        cdef infra.CaseManager* manager = <infra.CaseManager*>(pw.ptr)

        # and its allocator
        cdef infra.Allocator* mal = manager.mal

        return (mal.size_used, mal.size_total)


    def prepare(self, xi, xk):
        """def prepare(self, xi, xk):

Prepare geometry, generating, preconditioning (scaling) and LU factorizing the problem matrix for each problem instance.

This function multi-threads automatically using self.ntasks threads.

xi : array containing the origin of fit for each problem instance (dtype np.float64)
     2D or 3D: double[::view.generic,::view.contiguous]  (problem_instance_index, x_or_y(_or_z))
     1D:       double[::view.generic]  (problem_instance_index)
xk : array containing the coordinates of the neighbor points for each problem instance (dtype np.float64)
     2D or 3D: double[::view.generic,::view.generic,::view.contiguous]  (problem_instance_index, neighbor_point_index, x_or_y(_or_z))
     1D:       double[::view.generic,::view.generic]  (problem_instance_index, neighbor_point_index)

Data from self.nk (see __init__()) is used to determine the number of entries in xk (along the second dimension) for each problem instance.
"""
        self.ready = False  # (while preparing, not ready)

        cdef int dimension          = self.dimension
        cdef int ncases             = self.ncases
        cdef int ntasks             = self.ntasks
        cdef int debug              = self.debug
        cdef int[::view.generic] nk = self.nk

        # we'll coerce our parameters into either format depending on dimension
        cdef double[::view.generic,::view.contiguous] xiManyD
        cdef double[::view.generic,::view.generic,::view.contiguous] xkManyD
        cdef double[::view.generic] xi1D
        cdef double[::view.generic,::view.generic] xk1D

        # get the manager
        cdef ptrwrap.PointerWrapper pw  = <ptrwrap.PointerWrapper>(self.manager_pw)
        cdef infra.CaseManager* manager = <infra.CaseManager*>(pw.ptr)

        cdef infra.Case* case
        cdef double* pxi = <double*>0
        cdef int j

        # guest mode:
        #
        # TODO: sanity check that the xi and xk arrays match element-by-element to those used in the host? (Very expensive.)
        if self.host is not None:
            self.xk    = self.host.xk
            self.xi    = self.host.xi

            if dimension >= 2:
                xkManyD = self.xk
                xiManyD = self.xi
            else:
                xk1D = self.xk
                xi1D = self.xi

            if dimension == 3:
                with nogil:
                    for j in range(ncases):
                        case    = manager.cases[j]
                        pxi     = &xiManyD[j,0]
                        case.xi = pxi[0]
                        case.yi = pxi[1]
                        case.zi = pxi[2]

            elif dimension == 2:
                with nogil:
                    for j in range(ncases):
                        case    = manager.cases[j]
                        pxi     = &xiManyD[j,0]
                        case.xi = pxi[0]
                        case.yi = pxi[1]

            else: # dimension == 1:
                with nogil:
                    for j in range(ncases):
                        case = manager.cases[j]
                        case.xi = xi1D[j]

        # not in guest mode:
        else:

            self.xk   = xk    # needed at solve step in ALGO_ITERATIVE mode
            self.xi   = xi    # for patching the local model into a global one
            self.tree = None  # for fast neighbor searching of the xi points (in patching)

            # prepare the Cases
            #
            if dimension == 3:
                xkManyD = xk
                xiManyD = xi
                with nogil:
                    if ntasks > 1:
                        for j in cython.parallel.prange(ncases, num_threads=ntasks):
                            expert_prepare_one_3D( manager.cases[j], &xiManyD[j,0], xkManyD[j,:nk[j],:], debug )  # note the debug flag; this will also compute the 2-norm condition number if debug=True
                    else: # ntasks == 1:
                        for j in range(ncases):  # don't bother with OpenMP for single-task case
                            expert_prepare_one_3D( manager.cases[j], &xiManyD[j,0], xkManyD[j,:nk[j],:], debug )

            elif dimension == 2:
                xkManyD = xk
                xiManyD = xi
                with nogil:
                    if ntasks > 1:
                        for j in cython.parallel.prange(ncases, num_threads=ntasks):
                            expert_prepare_one_2D( manager.cases[j], &xiManyD[j,0], xkManyD[j,:nk[j],:], debug )
                    else: # ntasks == 1:
                        for j in range(ncases):  # don't bother with OpenMP for single-task case
                            expert_prepare_one_2D( manager.cases[j], &xiManyD[j,0], xkManyD[j,:nk[j],:], debug )

            else: # dimension == 1:
                xk1D = xk
                xi1D = xi
                with nogil:
                    if ntasks > 1:
                        for j in cython.parallel.prange(ncases, num_threads=ntasks):
                            expert_prepare_one_1D( manager.cases[j], xi1D[j], xk1D[j,:nk[j]], debug )
                    else: # ntasks == 1:
                        for j in range(ncases):  # don't bother with OpenMP for single-task case
                            expert_prepare_one_1D( manager.cases[j], xi1D[j], xk1D[j,:nk[j]], debug )

        self.ready = True  # preparation done, now ready for solve()


    def conds(self):
        """def conds(self):

Return an array containing the 2-norm condition number of the (scaled) problem matrix for each problem instance.

The returned array has shape (self.ncases,) and dtype np.float64.

This method is only available if __init__() was called with debug=True, and prepare() has been called (the condition number
computation is internally done during prepare()). Otherwise calling this raises RuntimeError."""
        if not self.ready:
            raise RuntimeError("Solver is not in the ready state; prepare() must be called before conds()")
        if not self.debug:
            raise RuntimeError("Not in debug mode; condition number data has not been computed")

        # get the manager
        cdef ptrwrap.PointerWrapper pw  = <ptrwrap.PointerWrapper>(self.manager_pw)
        cdef infra.CaseManager* manager = <infra.CaseManager*>(pw.ptr)

        cdef int ncases = self.ncases
        cdef int j
        cdef infra.Case* case
        cdef double[::1] out = np.empty( (ncases,), dtype=np.float64 )
        for j in range(ncases):
            out[j] = manager.cases[j].cond_scaled  # prepare() fills this in if the debug flag is set

#        # this approach can't work, because the matrices have already been LU factored by prepare() (preprocess_A(), which performs the scaling, does that too).
#        # (we would get the condition number of the packed LU representation of A, interpreted as a general matrix, which is nonsense)
#        #
#        cdef int j
#        cdef int ncases = self.ncases
#        cdef infra.Case* case
#        for j in range(ncases):
#            case = manager.cases[j]
#            out[j] = np.linalg.cond( <double[:case.nr:1,:case.nr]>case.A )

        return np.asanyarray(out)  # make it printable (true ndarray, not memoryview slice)


    def solve(self, double[::view.generic,::view.generic] fk, double[::view.generic,::view.contiguous] fi, double[::view.generic,::view.generic,::view.contiguous] sens=None):
        """def solve(self, double[::view.generic,::view.generic] fk, double[::view.generic,::view.contiguous] fi, double[::view.generic,::view.generic,::view.contiguous] sens=None):

Using the current geometry, fit the model to the given data.

Note that the knowns bitmasks (and certain other settings) have already been set up in the constructor.

This function multi-threads automatically using self.ntasks threads.

fk   : in, (self.ncases,max(self.nk)) array of function values at the neighbor points for each problem instance.
       The points xk must be in the same order as they were when prepare() was called.
fi   : in/out, (self.ncases,max(number_of_dofs)) array of knowns and unknowns. See wlsqm.fitter.simple.fit_?D() for a description.
       on input:  those elements must be filled for each problem instance i that correspond to the bitmask self.knowns[i]
       on output: the unknown elements will be filled in (leaving the knowns untouched).
sens : out. If self.do_sens = True, (self.ncases,max(self.nk),max(number_of_dofs)) array for sensitivity information.
            If self.do_sens = False, sens can be None (default).
       If fi[i,j] is unknown: sens[i,k,j] = d( fi[i,j] ) / d( fk[i,k] )
       If fi[i,j] is known:   sens[i,:,j] = nan (to indicate "not applicable").
       Here i = problem instance, j = DOF number, k = neighbot point number.

Data from self.nk is used to determine nk[i] where needed.

Above, number_of_dofs[i] is determined by the fitting order of each problem instance. To obtain max(number_of_dofs)
for allocating the fi and sens arrays, call number_of_dofs() with your dimensionality and max(order),
where order is the array that was passed to __init__().
"""
        if not self.ready:
            raise RuntimeError("Solver is not in the ready state; prepare() must be called before solve()")

        cdef int dimension          = self.dimension
        cdef int algorithm          = self.algorithm
        cdef int max_iter           = self.max_iter
        cdef int ncases             = self.ncases
        cdef int ntasks             = self.ntasks
        cdef int do_sens            = self.do_sens
        cdef int[::view.generic] nk = self.nk

        # get the manager
        cdef ptrwrap.PointerWrapper pw  = <ptrwrap.PointerWrapper>(self.manager_pw)
        cdef infra.CaseManager* manager = <infra.CaseManager*>(pw.ptr)

        # Placeholder memoryviewslice of the correct shape for the case where we don't do sensitivity analysis.
        #
        # Faster to pre-generate this just once, because it saves a " __Pyx_PyObject_to_MemoryviewSlice_fsdc_double(Py_None)"
        # each time the solver is called.
        #
        cdef double[::view.generic,::view.contiguous] NO_SENS = None

        # for ALGO_ITERATIVE mode - we'll coerce self.xk into either format depending on dimension
        cdef double[::view.generic,::view.generic,::view.contiguous] xkManyD
        cdef double[::view.generic,::view.generic] xk1D
        cdef double[::view.generic,::view.contiguous] dummyManyD  # for passing a pre-generated None
        cdef double[::view.generic] dummy1D
        if dimension >= 2:
            xkManyD    = self.xk
            xk1D       = None
            dummy1D    = None
        else: # dimension == 1:
            xkManyD    = None
            dummyManyD = None
            xk1D       = self.xk

        # the solve loop
        #
        cdef int j, nkj, taskid
        DEF TASKID = 0  # for single-task case
        cdef int total_max_iterations_taken=0     # max across tasks (when solving in parallel, will be filled at the end)
        cdef int* max_iterations_taken = <int*>0  # max for each task
        cdef int iterations_taken=0               # current value in current task
        with nogil:
            if algorithm == defs.ALGO_BASIC_c:
                if ntasks > 1:
                    # solve
                    if do_sens:  # must check since None cannot be sliced
                        for j in cython.parallel.prange(ncases, num_threads=ntasks):
                            nkj = nk[j]
                            expert_solve_one_basic( manager.cases[j], &fi[j,0], fk[j,:nkj], sens[j,:nkj,:], do_sens, openmp.omp_get_thread_num() )
                    else:
                        for j in cython.parallel.prange(ncases, num_threads=ntasks):
                            expert_solve_one_basic( manager.cases[j], &fi[j,0], fk[j,:nk[j]], NO_SENS, do_sens, openmp.omp_get_thread_num() )

                    # Write the solution into user-given fi array.
                    #
                    # We must do this after the solve loop has finished, to ensure that all cases get equal treatment for their input data.
                    #
                    # (It is possible that fk and the user-given fi are actually views into the same physical array,
                    #  and that the solve updates the function value column of fi, where fk gets its data from.
                    #  Thus, to be sure, we update the user-given fi only after all cases have been solved.)
                    #
                    for j in cython.parallel.prange(ncases, num_threads=ntasks):
                        infra.Case_get_fi( manager.cases[j], &fi[j,0] )

                else: # ntasks == 1:
                    # solve
                    if do_sens:
                        for j in range(ncases):  # don't bother with OpenMP for single-task case
                            nkj = nk[j]
                            expert_solve_one_basic( manager.cases[j], &fi[j,0], fk[j,:nkj], sens[j,:nkj,:], do_sens, TASKID )
                    else:
                        for j in range(ncases):
                            expert_solve_one_basic( manager.cases[j], &fi[j,0], fk[j,:nk[j]], NO_SENS, do_sens, TASKID )

                    # get solution
                    for j in range(ncases):
                        infra.Case_get_fi( manager.cases[j], &fi[j,0] )

            else: # algorithm == defs.ALGO_ITERATIVE_c:
                if ntasks > 1:
                    # we need an ntasks-sized array to find max_iterations_taken, since the solving now proceeds in parallel
                    max_iterations_taken = <int*>malloc( ntasks*sizeof(int) )
                    for taskid in range(ntasks):
                        max_iterations_taken[taskid] = 0

                    # solve
                    if do_sens:
                        if dimension >= 2:
                            for j in cython.parallel.prange(ncases, num_threads=ntasks):
                                taskid = openmp.omp_get_thread_num()
                                nkj    = nk[j]
                                iterations_taken = expert_solve_one_iterative( manager.cases[j], &fi[j,0], fk[j,:nkj], sens[j,:nkj,:], do_sens, taskid, max_iter, xkManyD[j,:nkj,:], dummy1D )
                                if iterations_taken > max_iterations_taken[taskid]:
                                    max_iterations_taken[taskid] = iterations_taken
                        else: # dimension == 1:
                            for j in cython.parallel.prange(ncases, num_threads=ntasks):
                                taskid = openmp.omp_get_thread_num()
                                nkj    = nk[j]
                                iterations_taken = expert_solve_one_iterative( manager.cases[j], &fi[j,0], fk[j,:nkj], sens[j,:nkj,:], do_sens, taskid, max_iter, dummyManyD, xk1D[j,:nkj] )
                                if iterations_taken > max_iterations_taken[taskid]:
                                    max_iterations_taken[taskid] = iterations_taken
                    else: # not do_sens:
                        if dimension >= 2:
                            for j in cython.parallel.prange(ncases, num_threads=ntasks):
                                taskid = openmp.omp_get_thread_num()
                                nkj    = nk[j]
                                iterations_taken = expert_solve_one_iterative( manager.cases[j], &fi[j,0], fk[j,:nkj], NO_SENS, do_sens, taskid, max_iter, xkManyD[j,:nkj,:], dummy1D )
                                if iterations_taken > max_iterations_taken[taskid]:
                                    max_iterations_taken[taskid] = iterations_taken
                        else: # dimension == 1:
                            for j in cython.parallel.prange(ncases, num_threads=ntasks):
                                taskid = openmp.omp_get_thread_num()
                                nkj    = nk[j]
                                iterations_taken = expert_solve_one_iterative( manager.cases[j], &fi[j,0], fk[j,:nkj], NO_SENS, do_sens, taskid, max_iter, dummyManyD, xk1D[j,:nkj] )
                                if iterations_taken > max_iterations_taken[taskid]:
                                    max_iterations_taken[taskid] = iterations_taken

                    # get solution
                    for j in cython.parallel.prange(ncases, num_threads=ntasks):
                        infra.Case_get_fi( manager.cases[j], &fi[j,0] )

                    # get the maximum number of iterations taken in any thread
                    for taskid in range(ntasks):
                        if max_iterations_taken[taskid] > total_max_iterations_taken:
                            total_max_iterations_taken = max_iterations_taken[taskid]
                    free( <void*>max_iterations_taken )

                else: # ntasks == 1:
                    # solve
                    if do_sens:
                        if dimension >= 2:
                            for j in range(ncases):  # don't bother with OpenMP for single-task case
                                nkj = nk[j]
                                iterations_taken = expert_solve_one_iterative( manager.cases[j], &fi[j,0], fk[j,:nkj], sens[j,:nkj,:], do_sens, TASKID, max_iter, xkManyD[j,:nkj,:], dummy1D )
                                if iterations_taken > total_max_iterations_taken:
                                    total_max_iterations_taken = iterations_taken
                        else: # dimension == 1:
                            for j in range(ncases):
                                nkj = nk[j]
                                iterations_taken = expert_solve_one_iterative( manager.cases[j], &fi[j,0], fk[j,:nkj], sens[j,:nkj,:], do_sens, TASKID, max_iter, dummyManyD, xk1D[j,:nkj] )
                                if iterations_taken > total_max_iterations_taken:
                                    total_max_iterations_taken = iterations_taken
                    else: # not do_sens:
                        if dimension >= 2:
                            for j in range(ncases):
                                nkj = nk[j]
                                iterations_taken = expert_solve_one_iterative( manager.cases[j], &fi[j,0], fk[j,:nkj], NO_SENS, do_sens, TASKID, max_iter, xkManyD[j,:nkj,:], dummy1D )
                                if iterations_taken > total_max_iterations_taken:
                                    total_max_iterations_taken = iterations_taken
                        else: # dimension == 1:
                            for j in range(ncases):
                                nkj = nk[j]
                                iterations_taken = expert_solve_one_iterative( manager.cases[j], &fi[j,0], fk[j,:nkj], NO_SENS, do_sens, TASKID, max_iter, dummyManyD, xk1D[j,:nkj] )
                                if iterations_taken > total_max_iterations_taken:
                                    total_max_iterations_taken = iterations_taken

                    # get solution
                    for j in range(ncases):
                        infra.Case_get_fi( manager.cases[j], &fi[j,0] )

        return total_max_iterations_taken  # maximum number of refinement iterations actually taken in any task; always 0 for ALGO_BASIC_c


    def prep_interpolate(self):
        """def prep_interpolate(self):

Prepare the global patched model for performing interpolation.

This will construct a kdtree out of the set of points "xi" and cache it, allowing fast lookups.

You only need to call this if you wish to call interpolate().

The method interpolate() interpolates the model (or its derivatives) to points other than those
in the set of points "xi" that was passed to prepare().

At the points "xi", the data values and all derivatives are already available in the array "fi"
after calling solve().
"""
        if not self.ready:
            raise RuntimeError("Solver is not in the ready state; prepare() must be called before prep_interpolate()")

        if self.host is not None:  # in guest mode, use the host's tree for searching
            self.tree = self.host.tree
        else:
            self.tree = scipy.spatial.cKDTree( data=self.xi )


    # TODO: continuous mode: add options for falloff, weighting function shape
    # TODO: continuous mode: add option to use the k closest local models?
    #
    def interpolate(self, x, mode='nearest', r=None, int diff=0, I=None):
        """def interpolate(self, x, mode='nearest', r=None, int diff=0, I=None):

Interpolate the global model or one of its derivatives.

The model must be solve()'d first.

x    : Points to which to interpolate (dtype np.float64).
       In 3D, array of shape (nx, 3).
       In 2D, array of shape (nx, 2).
       In 1D, array of shape (nx,).

mode : 'nearest'    = for each point in x, use the local model whose reference point "xi" is nearest to that point.

       The result will be piecewise continuous in each Voronoi cell, with jumps across cell boundaries.
       Note that these cells are not even known explicitly! (If you need them, consider using scipy.spatial.Voronoi .)

       NOTE: in 'nearest' mode, this function multi-threads automatically using self.ntasks threads.

       'continuous' = for each point in x, average over all local models that have their origin inside a radius r from that point.

       A weight function is used, with more weight given to models whose origin is closer to the point. The weight falls off to zero
       exactly at r. The result will be continuous over the whole domain.

       NOTE: in 'continuous' mode, the computation is always single-threaded, and also very slow.

r    : averaging radius for mode='continuous'. (No default, because no sane default can exist; depends on the point cloud.)

diff : Interpolate a derivative of the function instead of the function value itself. (The default is to interpolate the function value.)

       One of the constants i3_* (in 3D), i2_* (in 2D) or i1_* (in 1D) in wlsqm.fitter.defs.

       See wlsqm.fitter.interp.interpolate_fit() for details.

I    : If mode='nearest': override which local model to use for each point in x, skipping the search for the nearest local model,
       thus making the interpolation run significantly faster.

       This is useful for re-interpolating a model with updated data values, if you wish to evaluate again at the same points x.
       To obtain this data, see I_out in the return value below.

Return value: tuple (out, I_out) where
        out   = function value (or derivative value, depending on "diff")
        I_out = if mode='nearest', index of local model used for each point in x (array of shape (nx,), dtype np.long).
                This can be passed back in as "I".

                if mode='continuous', this is always None (i.e. currently not supported).
"""
        if mode not in ['nearest', 'continuous']:
            raise ValueError("mode must be one of 'nearest', 'continuous'; got '%d'" % mode)
        if mode == 'continuous'  and  r is None:
            raise ValueError("r must be specified in mode='continuous'")
        if diff is None:
            raise ValueError("diff cannot be None")

        # TODO/FIXME: catch the case where the model has been prepared, but not solved (since this uses the internal xi arrays of the Case instances)
        if self.tree is None:
            raise RuntimeError("Points xi have not been indexed; prep_interpolate() must be called before interpolate()")

        if I is not None  and  len(I) != len(x):
            raise ValueError("When 'I' is specified, 'I' must have the same length as x; got len(I) = %d, len(x) = %d." % (len(I), len(x)))

        cdef int dimension = self.dimension
        cdef int ntasks    = self.ntasks

        # get the manager
        cdef ptrwrap.PointerWrapper pw  = <ptrwrap.PointerWrapper>(self.manager_pw)
        cdef infra.CaseManager* manager = <infra.CaseManager*>(pw.ptr)

        # perform interpolation
        #
        cdef int nx = x.shape[0]
        cdef double[::1] out = np.empty( (nx,), dtype=np.float64 )
        cdef long[::1] I_out
        if mode == 'nearest':
            if I is not None:
                I_out = None  # if 'I' was given, don't bother copying it to I_out in expert_interpolate_nearest()
            else:
                I_out = np.empty( (nx,), dtype=np.long )  # I_out[j] = index of the local model (in self.cases) that was used to produce out[j]

            expert_interpolate_nearest( dimension, self.tree, manager, x, out, I_out, I, diff, ntasks )

            # API consistency: if 'I' was given, we may return it as I_out (so that we always return a valid I_out)
            #
            if I is not None:
                I_out = I

        else: # mode == 'continuous':
            I_out = None  # TODO: currently returning the index set is not supported for continuous mode; may use an arbitrary number of local models for each point so needs some thinking
            expert_interpolate_continuous( dimension, self.xi, self.tree, manager, x, out, diff, r )  # TODO/FIXME: always single-threaded for now

        return (np.asanyarray(out), np.asanyarray(I_out))  # return true np.ndarrays, not memoryview slices (to make printable etc.)


####################################################
# Internal C-level helpers for ExpertSolver
####################################################

cdef void expert_prepare_one_3D( infra.Case* case, double *pxi, double[::view.generic,::view.contiguous] xk, int debug ) nogil:
    # update the position of the origin of the fit for this problem instance
    case.xi = pxi[0]
    case.yi = pxi[1]
    case.zi = pxi[2]

    # prepare the matrices
    impl.make_c_3D( case, xk )
    impl.make_A( case )
    impl.preprocess_A( case, debug )

cdef void expert_prepare_one_2D( infra.Case* case, double *pxi, double[::view.generic,::view.contiguous] xk, int debug ) nogil:
    # update the position of the origin of the fit for this problem instance
    case.xi = pxi[0]
    case.yi = pxi[1]

    # prepare the matrices
    impl.make_c_2D( case, xk )
    impl.make_A( case )
    impl.preprocess_A( case, debug )


cdef void expert_prepare_one_1D( infra.Case* case, double xi, double[::view.generic] xk, int debug ) nogil:
    # update the position of the origin of the fit for this problem instance
    case.xi = xi

    # prepare the matrices
    impl.make_c_1D( case, xk )
    impl.make_A( case )
    impl.preprocess_A( case, debug )


cdef void expert_solve_one_basic( infra.Case* case, double* fi, double[::view.generic] fk, double[::view.generic,::view.contiguous] sens, int do_sens, int taskid ) nogil:
    infra.Case_set_fi( case, fi )  # populate knowns
    impl.solve( case, fk, sens, do_sens, taskid )

cdef int expert_solve_one_iterative( infra.Case* case, double* fi, double[::view.generic] fk, double[::view.generic,::view.contiguous] sens, int do_sens,
                                        int taskid, int max_iter, double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D ) nogil:
    infra.Case_set_fi( case, fi )  # populate knowns
    return impl.solve_iterative( case, fk, sens, do_sens, taskid, max_iter, xkManyD, xk1D )


cdef void expert_interpolate_nearest( int dimension, xi_tree, infra.CaseManager* manager, x, double[::1] out, long[::1] I_out, long[::1] I_in, int diff, int ntasks ):
    # For each point in x, find the local model whose origin is nearest (the nearest point in xi).
    #
    # This search takes the majority of the runtime of this function.
    #
    cdef long[::1] I  # scipy.spatial.cKDTree.query() returns an array of long
    if I_in is None:  # usual case
        dummy, I = xi_tree.query( x, k=1 )  # distances,indices = ...  (in this case, we don't need the distances (TODO: could be a useful quality metric?))
    else:  # use the caller-specified model indices (useful when re-interpolating an updated model for the same points x)
        I = I_in

    # we'll coerce our parameters into either format depending on dimension
    #
    cdef double[::view.generic,::view.contiguous] xManyD   # [point, x_y_or_z]
    cdef double[::view.generic] x1D                        # [point]
    cdef int nx
    if dimension >= 2:
        xManyD = x
        nx     = xManyD.shape[0]
    else: # dimension == 1:
        x1D    = x
        nx     = x1D.shape[0]

    # copy to caller data about which local model was used at each point in x
    #
    cdef int j
    if I_out is not None:
        for j in range(nx):
            I_out[j] = I[j]

    # If any of the x is NaN, no neighbor will be found for that point. Catch this and cancel processing if it occurs.
    #
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html#scipy.spatial.cKDTree.query
    # "Missing neighbors are indicated with self.n."
    cdef double nan = 0./0. # FIXME: IEEE-754 abuse
    if (np.asanyarray(I) == manager.ncases).any():
        for j in range(nx):
            out[j] = nan
        return  # TODO: handle error more gracefully

    # FIXME/TODO: silly slicing on the first axis of x?D[] because our API cannot interpolate to a single point
    cdef int m
    with nogil:
        if dimension == 3:
            if ntasks > 1:
                for m in cython.parallel.prange(nx, num_threads=ntasks):
                    interp.interpolate_3D( manager.cases[I[m]], xManyD[m:m+1,:], &out[m], diff )
            else:
                for m in range(nx):
                    interp.interpolate_3D( manager.cases[I[m]], xManyD[m:m+1,:], &out[m], diff )
        elif dimension == 2:
            if ntasks > 1:
                for m in cython.parallel.prange(nx, num_threads=ntasks):
                    interp.interpolate_2D( manager.cases[I[m]], xManyD[m:m+1,:], &out[m], diff )
            else:
                for m in range(nx):
                    interp.interpolate_2D( manager.cases[I[m]], xManyD[m:m+1,:], &out[m], diff )
        else: # dimension == 1:
            if ntasks > 1:
                for m in cython.parallel.prange(nx, num_threads=ntasks):
                    interp.interpolate_1D( manager.cases[I[m]], x1D[m:m+1], &out[m], diff )
            else:
                for m in range(nx):
                    interp.interpolate_1D( manager.cases[I[m]], x1D[m:m+1], &out[m], diff )


cdef void expert_interpolate_continuous( int dimension, xi, xi_tree, infra.CaseManager* manager, x, double[::1] out, int diff, r ):  # TODO/FIXME: always single-threaded for now
    # Index the points where the global model is to be interpolated (for faster searching of pairs).
    # TODO: add caching to ExpertSolver if this is the same as the last used x ("if x is x_cached"?)  OTOH, this would keep alive a reference to a caller-given array (that is probably intended as a temporary)...
    input_tree = scipy.spatial.cKDTree( data=x )

    # For each point in x, find all local models whose origin is within radius r.
    #
    # From the docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_tree.html#scipy.spatial.cKDTree.query_ball_tree
    #     For each element self.data[i] of this tree, results[i] is a list of the indices of its neighbors in other.data.
    #
    model_idxs = input_tree.query_ball_tree( other=xi_tree, r=r )

    # TODO: handle case where no neighbor is found (if x contains some NaNs)

    # we'll coerce our parameters into either format depending on dimension
    #
    cdef double[::view.generic,::view.contiguous] xiManyD  # [problem_instance, x_y_or_z]
    cdef double[::view.generic,::view.contiguous] xManyD   # [point, x_y_or_z]
    cdef double[::view.generic] xi1D                       # [problem_instance]
    cdef double[::view.generic] x1D                        # [point]

    cdef int nx
    if dimension >= 2:
        xManyD  = x
        xiManyD = xi
        nx      = xManyD.shape[0]
    else: # dimension == 1:
        x1D     = x
        xi1D    = xi
        nx      = x1D.shape[0]

#    # convert list of lists to np.array (turns out this is actually SLOWER than the pure Python approach!)
#    DEF MAX_NEIGH = 100
#    cdef int[:,::1] idxs
#    cdef int[::1] nis
#    cdef int[::1] arr
#    cdef int m, j
#    idxs = np.zeros( (nx,MAX_NEIGH), dtype=np.int32 )  # idxs[m,:] = model indices (pointing to Case numbers) for x[m]
#    nis  = np.empty( (nx,), dtype=np.int32 )           # nis[m]    = number of models that need to be considered for x[m] (number of actually filled columns in idxs[m,:])
#    for m in range(nx):
#        L  = model_idxs[m]
#        ni = len(L)
#        arr = np.array( L, dtype=np.int32 )
#        for j in range(ni):
#            idxs[m,j] = arr[j]
#        nis[m] = ni

    # TODO: accelerate this if possible (pretty much pure Python for now)
    cdef int m, i, ni, li
    cdef double tmp, value, acc, w, sum_w
    cdef double dx, dy, dz    # distances along each axis
    cdef double max_d2 = r*r  # maximum squared distance for weighting
    for m in range(nx):
        L     = model_idxs[m]  # models whose origins were within radius r of x[m]
        ni    = len(L)
#        ni = nis[m]
        acc   = 0.
        sum_w = 0.
        for i in range(ni):
            li = L[i]
#            li = idxs[m,i]
            if dimension == 3:
                interp.interpolate_3D( manager.cases[li], xManyD[m:m+1,:], &value, diff )
                # compute distance from x[m] to self.xi[L[i],:]
                dx = xManyD[m,0] - xiManyD[li,0]
                dy = xManyD[m,1] - xiManyD[li,1]
                dz = xManyD[m,2] - xiManyD[li,2]
                d2 = dx*dx + dy*dy + dz*dz
            elif dimension == 2:
                interp.interpolate_2D( manager.cases[li], xManyD[m:m+1,:], &value, diff )
                # compute distance from x[m] to self.xi[L[i],:]
                dx = xManyD[m,0] - xiManyD[li,0]
                dy = xManyD[m,1] - xiManyD[li,1]
                d2 = dx*dx + dy*dy
            else: # dimension == 1:
                interp.interpolate_1D( manager.cases[li], x1D[m:m+1], &value, diff )
                dx = x1D[m] - xi1D[li]
                d2 = dx*dx

            # distance squared, flipped on the distance axis (fast falloff near origin)
            DEF alpha = 0.  # weight remaining at maximum distance
            DEF beta  = 1. - alpha
            tmp = 1. - sqrt(d2 / max_d2)
            w = alpha + beta * tmp*tmp

            acc   += w * value
            sum_w += w

        out[m] = acc / sum_w

