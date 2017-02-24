# -*- coding: utf-8 -*-
#
# WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.
#
# Centralized memory allocation infrastructure.
#
# JJ 2016-11-30

# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True

# Total memory needed for arrays:
#    - no*sizeof(int) bytes for o2r; one shared copy is enough (bypass custom alloc, since only remap(), which needs this, computes "nr")
#    - no*sizeof(int) bytes for r2o; one shared copy is enough (                                --''--                                  )
#    - nprob*nk*no*sizeof(double) bytes for c   (actually, sum(nk_j, j in problems)*no*sizeof(double))
#    - nprob*nr*sizeof(double) bytes for row_scale
#    - nprob*nr*sizeof(double) bytes for column_scale
#    - nprob*nr*nr*sizeof(double) bytes for A
#    - nprob*nr*sizeof(int) bytes for ipiv
#    - solve:
#    -   if do_sens, ntasks*nr*(nk+1)*sizeof(double) bytes for wrk
#           - use max(nk_j) here to fit the largest problem instance (since any thread may run it)
#    -   if not do_sens, ntasks*nr*sizeof(double) bytes for wrk
#    - iterative refinement:
#      - ntasks*nk*sizeof(double) bytes for fk_tmp
#           - max(nk_j) here too, same reason
#      - ntasks*no*sizeof(double) bytes for fi_tmp

from __future__ import division
from __future__ import absolute_import

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

cimport wlsqm.fitter.defs as defs  # C constants

# use GCC's intrinsics for counting the number of set bits in an int
#
# See
#   http://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer (algorithms, suggestions)
#   https://gist.github.com/craffel/e470421958cad33df550 (Cython defs; on popcounting a NumPy array)
#
cdef extern int __builtin_popcount(unsigned int) nogil
cdef extern int __builtin_popcountll(unsigned long long) nogil

#####################################
# Helper functions
#####################################

# Return number of DOFs in the original (unreduced) system.
#
# dimension : in, number of space dimensions (1, 2 or 3)
# order     : in, the order of the polynomial to be fitted
#
cdef int number_of_dofs( int dimension, int order ) nogil:
    if dimension not in [1,2,3]:
        return -1
#        with gil:
#            raise ValueError( "dimension must be 1, 2 or 3; got %d" % dimension )
    if order not in [0,1,2,3,4]:
        return -2
#        with gil:
#            raise ValueError( "order must be 0, 1, 2, 3 or 4; got %d" % order )

    cdef int no
    if dimension == 3:
        if order == 4:
            no = defs.i3_4th_end_c
        elif order == 3:
            no = defs.i3_3rd_end_c
        elif order == 2:
            no = defs.i3_2nd_end_c
        elif order == 1:
            no = defs.i3_1st_end_c
        else: # order == 0:
            no = defs.i3_0th_end_c
    elif dimension == 2:
        if order == 4:
            no = defs.i2_4th_end_c
        elif order == 3:
            no = defs.i2_3rd_end_c
        elif order == 2:
            no = defs.i2_2nd_end_c
        elif order == 1:
            no = defs.i2_1st_end_c
        else: # order == 0:
            no = defs.i2_0th_end_c
    else: # dimension == 1:
        if order == 4:
            no = defs.i1_4th_end_c
        elif order == 3:
            no = defs.i1_3rd_end_c
        elif order == 2:
            no = defs.i1_2nd_end_c
        elif order == 1:
            no = defs.i1_1st_end_c
        else: # order == 0:
            no = defs.i1_0th_end_c

    return no

# Return the number of DOFs in the reduced system, corresponding to an original (unreduced) number of DOFs  n  and a knowns mask.
#
# n    : in, number of DOFs in the original (unreduced) system
# mask : in, bitmask of knowns
#
cdef int number_of_reduced_dofs( int n, long long mask ) nogil:
    cdef int ne = __builtin_popcountll(mask)  # number of eliminated DOFs = number of bits set in mask
    return n - ne  # remaining DOFs

# Reduce the system size by removing the rows/columns for knowns from the DOF numbering.
#
# Specifically:
#
# Given a bitmask of DOFs to eliminate, construct DOF number mappings
# between the original full equation system and the reduced equation system.
#
# o2r  : out, mapping original --> reduced;  size (n,), must be allocated by caller
# r2o  : out, mapping reduces  --> original; size (n,), must be allocated by caller
# n    : in, number of DOFs in the original (unreduced) system
# mask : in, bitmask of knowns
#
# return value: the number of DOFs in the reduced system.
#
# In the arrays, non-existent DOFs will be represented by the special value -1.
#
# In o2r (original->reduced), non-existent DOFs are those that were eliminated
# (hence have no index in the reduced system).
#
# In r2o (reduced->original), non-existent DOFs are those with  index >= n_reduced,
# where  n_reduced = (n - n_eliminated),  since the reduced system has only n_reduced DOFs in total.
#
cdef int remap( int* o2r, int* r2o, int n, long long mask ) nogil:  # o = original, r = reduced
    # We always start the elimination with a full range(n) of DOFs.
    #
    # For example, if we have 4 DOFs, and we would like to eliminate the DOF "1",
    # we construct the following mappings:
    #
    #    orig -> reduced
    #
    #       0 ->  0
    #       1 -> -1 (original DOF "1" does not exist in reduced system)
    #       2 ->  1
    #       3 ->  2
    #
    # reduced -> orig
    #
    #       0 ->  0
    #       1 ->  2
    #       2 ->  3
    #       3 -> -1 (in the reduced system, there is no DOF "3")
    #
    # The right-hand sides can be expressed in array form, using the left-hand side as the array index:
    #
    # orig->reduced: [0, -1, 1, 2]
    # reduced->orig: [0, 2, 3, -1]
    #
    # These arrays are the output format.
    #
    # This says that e.g. the DOF "2" of orig maps to DOF "1" of reduced (array orig->reduced, index 2).
    # The DOF "1" of reduced maps to DOF "2" of orig (array reduced->orig, index 1).

    # We first generate orig -> reduced.
    #
    cdef int j, k=0  # k = first currently available DOF number (0-based) in the reduced system
    for j in range(n):
        if mask & (1LL << j):  # eliminate this DOF?
            o2r[j] = -1
        else:
            o2r[j] = k
            k += 1  # a DOF was introduced into the reduced system

    # k is now the number of DOFs in the reduced system

    # Construct the inverse to obtain reduced -> orig. See the example above.
    #
    for j in range(n):
        if o2r[j] == -1:
            continue
        r2o[ o2r[j] ] = j

    # In the reduced -> orig mapping, set the rest to -1, since the reduced system
    # has less DOFs than the original one.
    #
    for j in range(k, n):
        r2o[j] = -1

    return k


#################################################
# class Allocator:
#################################################

# To avoid memory fragmentation in the case with many instances of the model being fitted at once,
# we use a custom memory allocator.
#
# This is very simplistic; we do not need to support the re-use of already allocated blocks.
#
# Example:
#    int total_size_bytes = 1000000  # 1 MB
#    Allocator* a = Allocator_new( ALLOC_MODE_ONEBIGBUFFER, total_size_bytes )
#    int* my_block_1 = <int*>Allocator_malloc( a, 100*sizeof(int) )
#    # ...other Allocator_malloc()'s...
#    # ...
#    Allocator_free( a, my_block_1 )
#    # ...other Allocator_free()'s...
#    Allocator_del( a )

# Object-oriented programming, C style.
cdef int ALLOC_MODE_PASSTHROUGH  = 1  # pass each call through to C malloc/free
cdef int ALLOC_MODE_ONEBIGBUFFER = 2  # pre-allocate one big buffer to fit everything in

# Constructor.
#
# This class is only intended to be used from a Python thread.
#
# Note that ALLOC_MODE_PASSTRHOUGH doesn't use total_size_bytes, but .pxd files do not support default values for function arguments,
# and Cython's "int total_size_bytes=*" syntax (in the .pxd file) does not support nogil functions.
#
cdef Allocator* Allocator_new( int mode, int total_size_bytes ) nogil except <Allocator*>0:
    cdef Allocator* self = <Allocator*>malloc( sizeof(Allocator) )
    if self == <Allocator*>0:  # we promised Cython not to return NULL, so we must raise if the malloc fails
        with gil:
            raise MemoryError("Out of memory trying to allocate an Allocator object")

    if mode == ALLOC_MODE_ONEBIGBUFFER  and  total_size_bytes > 0:
        self.buffer = malloc( total_size_bytes )
        if self.buffer == <void*>0:
            with gil:
                raise MemoryError("Out of memory trying to allocate a buffer of %d bytes" % (total_size_bytes))
    else:
        self.buffer = <void*>0

    self.mode       = mode
    self.size_total = total_size_bytes
    self.p          = self.buffer
    self.size_used  = 0

    return self

cdef void* Allocator_malloc( Allocator* self, int size_bytes ) nogil:
    if self.mode == ALLOC_MODE_PASSTHROUGH:
#        with gil:
#            print "directly allocating %d bytes" % (size_bytes)
        return malloc( size_bytes )

    # else...

    # pathological case: no buffer, can't allocate
    if self.buffer == <void*>0:
        return <void*>0

    # check that there is enough space remaining in the buffer
    cdef int size_remaining = self.size_total - self.size_used
    if size_bytes > size_remaining:
#        with gil:
#            print "buffer full, cannot allocate %d bytes" % (size_bytes)  # DEBUG
        return <void*>0

    cdef void* p
    with gil:  # since we are called from Python threads only, we can use the GIL to make the operation thread-safe. (TODO/FIXME: well, not exactly, see e.g. http://www.slideshare.net/dabeaz/an-introduction-to-python-concurrency )
#        print "reserving %d bytes from buffer of size %d; after alloc, %d bytes remaining" % (size_bytes, self.size_total, size_remaining - size_bytes)  # DEBUG
        p = self.p
        self.p = <void*>( (<char*>p) + size_bytes )
        self.size_used += size_bytes

    return p

cdef void Allocator_free( Allocator* self, void* p ) nogil:
    if self.mode == ALLOC_MODE_PASSTHROUGH:
        free( p )
    # else do nothing; this simplistic allocator doesn't reuse blocks once they are allocated

cdef int Allocator_size_remaining( Allocator* self ) nogil:
    return self.size_total - self.size_used

# Destructor.
cdef void Allocator_del( Allocator* self ) nogil:
    if self != <Allocator*>0:
        free( self.buffer )
        free( self )


#################################################
# class CaseManager:
#################################################

# Constructor.
#
# max_cases is mandatory (with an invalid default value since no sane default can exist).
#
# ntasks is for parallel processing at solve time; effectively, it specifies how many per-task arrays to allocate.
# When processing serially, use the value 1.
#
cdef CaseManager* CaseManager_new( int dimension, int do_sens, int iterative, int max_cases, int ntasks ) nogil except <CaseManager*>0:
    # Generally speaking, fixing the array size at instantiation time is stupid (an automatically expanding buffer would be better),
    # but considering that this class has only one actual user, where we do know max_cases in advance, it is fine for our purposes.
    if max_cases < 1:
        with gil:
            raise ValueError("Must specify max_cases > 0 when creating a CaseManager.")

    cdef CaseManager* self = <CaseManager*>malloc( sizeof(CaseManager) )
    if self == <CaseManager*>0:  # we promised Cython not to return NULL, so we must raise if the malloc fails
        with gil:
            raise MemoryError("Out of memory trying to allocate an CaseManager object")

    # init parallel proc
    #
    self.ntasks  = ntasks
    self.wrks    = <double**>malloc( ntasks*sizeof(double*) )
    self.fk_tmps = <double**>malloc( ntasks*sizeof(double*) )
    self.fi_tmps = <double**>malloc( ntasks*sizeof(double*) )
    for j in range(ntasks):  # init to NULL needed to gracefully handle errors in CaseManager_allocate()
        self.wrks[j]    = <double*>0
        self.fk_tmps[j] = <double*>0
        self.fi_tmps[j] = <double*>0

    # init storage for Case pointers
    #
    self.cases = <Case**>malloc( max_cases*sizeof(Case*) )
    self.max_cases = max_cases
    self.ncases = 0

    # save metadata
    #
    self.dimension = dimension
    self.do_sens   = do_sens
    self.iterative = iterative

    # these will be set up at allocate time
    #
    self.mal          = <Allocator*>0
    self.bytes_needed = 0

    return self

# Add a case to this manager.
#
# This means the manager will manage the memory for the Case, and at destruction time,
# will also destroy the managed Case.
#
# Up to max_cases Case objects can be added to the manager (see CaseManager_new()).
#
# Case_new() will call this automatically, if a manager is specified.
#
cdef int CaseManager_add( CaseManager* self, Case* case ) nogil except -1:
    # sanity check remaining space
    if self.ncases == self.max_cases:
        with gil:
            raise MemoryError("Case pointer buffer full, max_cases = %d reached" % self.max_cases)

    # sanity check Case metadata for compatibility with this CaseManager instance
    if case.dimension != self.dimension:
        with gil:
            raise ValueError("Cannot add case with different dimension = %d; this manager has dimension = %d" % (case.dimension, self.dimension))
    if case.do_sens != self.do_sens:
        with gil:
            raise ValueError("Cannot add case with different setting for do_sens = %d; this manager has do_sens = %d" % (case.do_sens, self.do_sens))
    if case.iterative != self.iterative:
        with gil:
            raise ValueError("Cannot add case with different setting for iterative = %d; this manager has iterative = %d" % (case.iterative, self.iterative))

    # add the Case to the managed cases.
    self.cases[self.ncases] = case
    self.ncases += 1

    return 0

# Finish adding Case objects. Prepare the manaager for solving.
#
# This should be called exactly once (per instance of CaseManager), after all cases have been CaseManager_add()'d.
#
cdef int CaseManager_commit( CaseManager* self ) nogil except -1:
    return CaseManager_allocate( self )

# Create the memory buffer that will contain the data arrays for all cases.
#
cdef int CaseManager_allocate( CaseManager* self ) nogil except -1:
    cdef int j, problem_instance_bytes=0, task_bytes=0
    cdef int max_nk=0, max_no=0, max_nr=0
    cdef int size_wrk=0, size_fk_tmp=0, size_fi_tmp=0
    cdef BufferSizes sizes
    cdef Case* case
    with gil:
        try:
            if self.ncases == 0:
                raise ValueError("No cases; add some before allocating")

            # Determine total amount of memory needed by the per-problem-instance arrays.
            #
            # Also, find max_nk for allocation of per-task arrays. (nk may vary across cases)
            #
            for j in range(self.ncases):
                case = self.cases[j]
                Case_determine_sizes( case, &sizes )

                problem_instance_bytes += sizes.total

                if case.nk > max_nk:
                    max_nk = case.nk
                if case.no > max_no:
                    max_no = case.no
                if case.nr > max_nr:
                    max_nr = case.nr

            # Determine maximum needed size for one instance of the per-task arrays,
            # when working on this set of cases.
            #
            if self.do_sens:
                size_wrk = max_nr*(max_nk + 1)*sizeof(double)
            else:
                size_wrk = max_nr*sizeof(double)

            if self.iterative:
                size_fk_tmp = max_nk*sizeof(double)
                size_fi_tmp = max_no*sizeof(double)
            else:
                size_fk_tmp = 0
                size_fi_tmp = 0

            # The total for the per-task arrays is then just ntasks copies:
            #
            task_bytes = self.ntasks*(size_wrk + size_fk_tmp + size_fi_tmp)

            # Final total of memory needed is thus:
            #
            self.bytes_needed = task_bytes + problem_instance_bytes

            # NOTE: this may be big (e.g. ~5.5kB per problem instance for dimension=2, order=4, nk=25,
            #                        so for a moderate number of 1e4 problem instances, this is already 55MB)
            #
            # The good news is that even if multiple fits (against different data) are performed with the same points,
            # we can simply let the buffer be - there is no need to re-create it for each run.
            #
            self.mal = Allocator_new( mode=ALLOC_MODE_ONEBIGBUFFER, total_size_bytes=self.bytes_needed )

            # Allocate the per-task arrays.
            #
            for j in range(self.ntasks):
                self.wrks[j]    = <double*>Allocator_malloc( self.mal, size_wrk )
                self.fk_tmps[j] = <double*>Allocator_malloc( self.mal, size_fk_tmp )
                self.fi_tmps[j] = <double*>Allocator_malloc( self.mal, size_fi_tmp )

            # Finally, tell the Case objects to allocate their memory.
            #
            # They will automatically grab our allocator, since they are in managed mode.
            #
            for j in range(self.ncases):
                Case_allocate( self.cases[j] )

        except:
            # on error, leave the CaseManager in the state it was in before this method was called.
            CaseManager_deallocate( self )
            raise

    return 0

# The opposite of CaseManager_allocate().
#
cdef void CaseManager_deallocate( CaseManager* self ) nogil:
    if self != <CaseManager*>0:
        self.bytes_needed = 0

        if self.mal != <Allocator*>0:  # the allocator instantiation may have failed, so make sure we have an allocator before attempting this
            # the managed Case objects also use our allocator
            for j in range(self.ncases):
                Case_deallocate( self.cases[j] )  # it is safe to Case_deallocate() also a Case that has not yet been Case_allocate()'d.

            for j in range(self.ntasks):
                Allocator_free( self.mal, self.fi_tmps[j] )
                self.fi_tmps[j] = <double*>0
                Allocator_free( self.mal, self.fk_tmps[j] )
                self.fk_tmps[j] = <double*>0
                Allocator_free( self.mal, self.wrks[j] )
                self.wrks[j]    = <double*>0

            Allocator_del( self.mal )
            self.mal = <Allocator*>0

# Destructor. Destroys also the managed Case objects.
#
cdef void CaseManager_del( CaseManager* self ) nogil:
    cdef int j
    if self != <CaseManager*>0:
        CaseManager_deallocate( self )

        # destroy the managed Case objects
        for j in range(self.ncases):
            Case_del( self.cases[j] )

        # free manually allocated storage
        free( self.cases )
        free( self.fi_tmps )
        free( self.fk_tmps )
        free( self.wrks )

        free( self )


#################################################
# class Case:
#################################################

# Constructor.
#
# This class is only intended to be instantiated from a Python thread.
#
# manager: an already existing CaseManager object to use, to share the memory allocator among a set of cases.
#          The cases must have the same dimension, do_sens, iterative.
#
#          If null, an Allocator will be created locally.
#
# host:    for guest mode, an existing Case object to use. The geometry data (o2r,r2o,c,w,A,row_scale,col_scale,ipiv) will be borrowed off the host,
#          and no local copies will be created.
#
#          This can be used to save both memory and time when different fields (in an IBVP problem) live on the exact same geometry.
#          "Geometry" includes both xi,yi.zi (point "xi") and the neighbor set (points "xk"; see wlsqm.fitter.impl.make_c_?D()).
#
#          Thus, the host Case instance must have the exact same parameters (and geometry!) as the Case instance being created.
#          "Parameters" include  dimension, order, nk, knowns, weighting_method.
#
#          The parameter match is not checked! See wlswm2_expert.ExpertSolver for correct usage.
#          (It does some rudimentary checking, but does not check the geometry.)
#
#          When using guest mode, the calling code must make sure the host instance stays alive at least as long as its guest instances,
#          or hope for a crash.
#
#          If null, the geometry data will be allocated locally.
#
cdef Case* Case_new( int dimension, int order, double xi, double yi, double zi, int nk, long long knowns, int weighting_method, int do_sens, int iterative, CaseManager* manager, Case* host ) nogil except <Case*>0:
    cdef Case* self = <Case*>malloc( sizeof(Case) )
    if self == <Case*>0:  # we promised Cython not to return NULL, so we must raise if the malloc fails
        with gil:
            raise MemoryError("Out of memory trying to allocate a Case object")

    # tag unused components as NaN
    cdef double nan = 0./0.  # NaN as per IEEE-754
    self.xi = xi
    self.yi = yi  if dimension >= 2  else  nan
    self.zi = zi  if dimension == 3  else  nan

    # init data pointers to NULL to make it safe to dealloc partially initialized Case (when something goes wrong)
    self.o2r       = <int*>0
    self.r2o       = <int*>0
    self.c         = <double*>0
    self.w         = <double*>0
    self.A         = <double*>0
    self.row_scale = <double*>0
    self.col_scale = <double*>0
    self.ipiv      = <int*>0
    self.fi        = <double*>0
    self.fi2       = <double*>0
    self.wrk       = <double*>0
    self.fk_tmp    = <double*>0
    self.fi_tmp    = <double*>0

    if host == <Case*>0:
        self.geometry_owned = 1

        # set condition numbers to nan until computed (only computed if wlsqm.fitter.impl.prepare() is called with the debug flag set!)
        self.cond_orig   = nan
        self.cond_scaled = nan

    else:
        self.geometry_owned = 0
        self.o2r            = host.o2r
        self.r2o            = host.r2o
        self.c              = host.c
        self.w              = host.w
        self.A              = host.A
        self.row_scale      = host.row_scale
        self.col_scale      = host.col_scale
        self.ipiv           = host.ipiv

        # these may have been computed by host (this only works if the host has had preprocess_A() called on it already, but this is the best we can do)
        self.cond_orig      = host.cond_orig
        self.cond_scaled    = host.cond_scaled

    # Use the data from CaseManager if given
    self.have_manager = (manager != <CaseManager*>0)
    self.manager      = manager        # (copies the pointer also if NULL)
    self.mal          = <Allocator*>0  # this will be filled at allocate time

    # determine number of DOFs in the original (unreduced) and reduced systems (needed to determine array sizes)
    cdef int no = number_of_dofs( dimension, order )
    cdef int nr = number_of_reduced_dofs( no, knowns )

    # save metadata
    self.dimension        = dimension
    self.order            = order
    self.knowns           = knowns
    self.weighting_method = weighting_method
    self.no               = no
    self.nr               = nr
    self.nk               = nk
    self.do_sens          = do_sens
    self.iterative        = iterative

    # Now the Case is in a half-initialized state, with metadata available, but no memory allocated yet.

    if self.have_manager:
        # In managed mode, cases automatically add themselves to the manager.
        with gil:
            try:
                CaseManager_add( self.manager, self )  # this may raise if the buffer is full
            except:
                free( self )
                raise
    else:
        # In unmanaged mode, for caller convenience, cases fully initialize themselves, since there is no separate allocate step
        # that depends on having all the cases available (to compute the final buffer size).
        Case_allocate( self )

    return self

# Getters for work space pointers for parallel task "taskid" (0, 1, ..., ntasks-1).
#
# In managed mode, the work spaces live in the manager (there are ntasks copies).
#
# In unmanaged mode, each Case has its own work space.
#
cdef double* Case_get_wrk( Case* self, int taskid ) nogil:
    if self.have_manager:
        return self.manager.wrks[taskid]
    else:
        return self.wrk

cdef double* Case_get_fk_tmp( Case* self, int taskid ) nogil:
    if self.have_manager:
        return self.manager.fk_tmps[taskid]
    else:
        return self.fk_tmp

cdef double* Case_get_fi_tmp( Case* self, int taskid ) nogil:
    if self.have_manager:
        return self.manager.fi_tmps[taskid]
    else:
        return self.fi_tmp

# Helper: convert an (nk,) array of squared distances to corresponding weights.
#
# w      : in/out. On entry, squared distances from xi to each xk.
#                  On exit, weight factors for each xk.
# nk     : in, number of neighbor points (i.e. points xk)
# max_d2 : in, the largest squared distance seen (i.e. the max element of input w).
#          This is used for normalization.
# weighting_method : in, one of the constants WEIGHT_*. Specifies the type of weighting to use;
#                    different weightings are good for different use cases of WLSQM.
#
cdef void Case_make_weights( Case* self, double max_d2 ) nogil:
    cdef double* w = self.w
    cdef int nk    = self.nk
    cdef int weighting_method = self.weighting_method

    # no-op in guest mode (weights already computed in the host Case instance)
    if not self.geometry_owned:
        return

    cdef int k
    cdef double d2, tmp
    if weighting_method == defs.WEIGHT_UNIFORM_c:
        # Trivial weighting. Don't use distance information, treat all points as equally important.
        #
        # This gives the best overall fit of function values across all points xk,
        # at the cost of accuracy of derivatives at the point xi.
        #
        # (Essentially, this cost is because derivatives are local, so the information
        #  from far-away points corrupts them.)
        #
        for k in range(nk):
            w[k] = 1.

    else: # weighting_method == defs.WEIGHT_CENTER_c:
        # Emphasize points close to xi.
        #
        # Improves the fit of derivatives at the point xi, at the cost of the overall fit
        # of function values at points xk that are (relatively speaking) distant from xi.
        #
        for k in range(nk):
            d2 = w[k]  # the array w originally contains squared distances (without normalization)

            # distance squared, flipped on the distance axis (fast falloff near origin)
            DEF alpha = 1e-4  # weight remaining at maximum distance
            DEF beta  = 1. - alpha
            tmp = 1. - sqrt(d2 / max_d2)
            w[k] = alpha + beta * tmp*tmp

# Determine how many bytes of memory this Case will need for storing its arrays.
#
# Write the result into the given BufferSizes struct.
#
cdef void Case_determine_sizes( Case* self, BufferSizes* sizes ) nogil:
    cdef int no        = self.no
    cdef int nr        = self.nr
    cdef int nk        = self.nk
    cdef int do_sens   = self.do_sens
    cdef int iterative = self.iterative

    if self.geometry_owned:
        sizes.o2r       = no*sizeof(int)        # (no,)
        sizes.r2o       = no*sizeof(int)        # (no,)
        sizes.c         = nk*no*sizeof(double)  # (nk,no), C-contiguous
        sizes.w         = nk*sizeof(double)     # (nk,)
        sizes.A         = nr*nr*sizeof(double)  # (nr, nr), Fortran-contiguous
        sizes.row_scale = nr*sizeof(double)     # (nr,)
        sizes.col_scale = nr*sizeof(double)     # (nr,)
        sizes.ipiv      = nr*sizeof(int)        # (nr,)
    else:
        # This function computes only bytes needed, so in guest mode we can put zeroes here.
        # This function is not used for determining the number of elements in anything.
        sizes.o2r       = 0
        sizes.r2o       = 0
        sizes.c         = 0
        sizes.w         = 0
        sizes.A         = 0
        sizes.row_scale = 0
        sizes.col_scale = 0
        sizes.ipiv      = 0

    # The coefficient array is always needed.
    #
    sizes.fi        = no*sizeof(double)     # (no,)

    # For any polynomial of degree  d >= 1,  its (non-zero) derivatives are a polynomial of degree  d - 1.
    # In the zeroth order case, the derivative is everywhere zero.
    cdef int no2    = number_of_dofs( self.dimension, self.order - 1 )  if self.order >= 1  else  0
    sizes.fi2       = no2*sizeof(double)    # (no2,)

    # per-task work space arrays
    #
    if self.have_manager:
        # in managed mode, CaseManager will allocate one copy of the per-task (not per-problem-instance) arrays
        sizes.wrk    = 0
        sizes.fk_tmp = 0
        sizes.fi_tmp = 0
    else:
        # unmanaged mode - allocate also the per-task arrays locally.
        # see the header comment of solve() for the solver work space sizes
        if do_sens:
            sizes.wrk = nr*(nk + 1)*sizeof(double)
        else:
            sizes.wrk = nr*sizeof(double)

        if iterative:
            sizes.fk_tmp = nk*sizeof(double)
            sizes.fi_tmp = no*sizeof(double)
        else:
            sizes.fk_tmp = 0
            sizes.fi_tmp = 0

    sizes.total = sizes.o2r + sizes.r2o \
                + sizes.c + sizes.w \
                + sizes.A + sizes.row_scale + sizes.col_scale + sizes.ipiv \
                + sizes.fi + sizes.fi2 \
                + sizes.wrk \
                + sizes.fk_tmp + sizes.fi_tmp

# Load user-given data into the coefficients fi.
#
# The length of the input is assumed to be self.no.
#
# This can be used to populate knowns.
#
cdef void Case_set_fi( Case* self, double* fi ) nogil:
    cdef double* my_fi = self.fi
    cdef int no        = self.no
    cdef int om
    for om in range(no):
        my_fi[om] = fi[om]

# Populate user-given array of length self.no
# with the solution data (coefficients self.fi).
#
cdef void Case_get_fi( Case* self, double* out ) nogil:
    cdef double* my_fi = self.fi
    cdef int no        = self.no
    cdef int om
    for om in range(no):
        out[om] = my_fi[om]

# Perform memory allocation.
#
cdef int Case_allocate( Case* self ) nogil except -1:
    # At this point, the constructor has finished, so we have no, nr, nk and the flags (do_sens, iterative).
    #
    # Calculate the (space-)optimal buffer size for ONEBIGBUFFER mode.
    #
    # (This also calculates the various individual sizes, which we will use to actually allocate the memory.)
    #
    cdef BufferSizes sizes
    Case_determine_sizes( self, &sizes )

    # Acquire or create the custom memory allocator to allocate storage for actual data.
    #
    cdef int size_remaining=-1
    cdef Allocator* mal
    with gil:
        try:
            if self.have_manager:  # managed mode - external allocator given; check that it has enough space for us.
                mal = self.manager.mal
                size_remaining = Allocator_size_remaining( mal )
                if size_remaining < sizes.total:
                    raise MemoryError("%d bytes of memory needed, but the given allocator has only %d bytes remaining." % (sizes.total, size_remaining))

            else:  # unmanaged mode - instantiate our own Allocator. The Allocator constructor will raise MemoryError if it runs out of memory.
                mal = Allocator_new( mode=ALLOC_MODE_ONEBIGBUFFER, total_size_bytes=sizes.total )
        except:
            free( self )
            raise
    self.mal = mal

    # Allocate the storage, using the custom allocator.
    #
    if self.geometry_owned:
        self.o2r       = <int*>   Allocator_malloc( mal, sizes.o2r )
        self.r2o       = <int*>   Allocator_malloc( mal, sizes.r2o )

        self.c         = <double*>Allocator_malloc( mal, sizes.c )
        self.w         = <double*>Allocator_malloc( mal, sizes.w )

        self.A         = <double*>Allocator_malloc( mal, sizes.A )
        self.row_scale = <double*>Allocator_malloc( mal, sizes.row_scale )
        self.col_scale = <double*>Allocator_malloc( mal, sizes.col_scale )
        self.ipiv      = <int*>   Allocator_malloc( mal, sizes.ipiv )

    self.fi        = <double*>Allocator_malloc( mal, sizes.fi )

    if sizes.fi2:
        self.fi2   = <double*>Allocator_malloc( mal, sizes.fi2 )
    else:
        self.fi2   = <double*>0

    if self.have_manager:
        # in managed mode, CaseManager will allocate one copy of the per-task (not per-problem-instance) arrays
        self.wrk    = <double*>0
        self.fk_tmp = <double*>0
        self.fi_tmp = <double*>0

    else:
        # unmanaged mode - allocate the per-task arrays locally.
        self.wrk        = <double*>Allocator_malloc( mal, sizes.wrk )

        if self.iterative:
            self.fk_tmp = <double*>Allocator_malloc( mal, sizes.fk_tmp )
            self.fi_tmp = <double*>Allocator_malloc( mal, sizes.fi_tmp )
        else:
            self.fk_tmp = <double*>0
            self.fi_tmp = <double*>0

    # memory allocated; populate o2r and r2o
    #
    # (in guest mode, this has already been done in the host Case instance)
    #
    if self.geometry_owned:
        remap( self.o2r, self.r2o, self.no, self.knowns )

    return 0

# The opposite of Case_allocate().
#
cdef void Case_deallocate( Case* self ) nogil:
    if self != <Case*>0:
        # No guarantee that we'll be called only from the destructor;
        # we must set any deallocated pointers to NULL.
        Allocator_free( self.mal, self.fi_tmp )
        self.fi_tmp = <double*>0
        Allocator_free( self.mal, self.fk_tmp )
        self.fk_tmp = <double*>0

        Allocator_free( self.mal, self.wrk )
        self.wrk = <double*>0

        Allocator_free( self.mal, self.fi2 )
        self.fi2 = <double*>0
        Allocator_free( self.mal, self.fi )
        self.fi = <double*>0

        if self.geometry_owned:
            Allocator_free( self.mal, self.ipiv )
            self.ipiv = <int*>0
            Allocator_free( self.mal, self.col_scale )
            self.col_scale = <double*>0
            Allocator_free( self.mal, self.row_scale )
            self.row_scale = <double*>0
            Allocator_free( self.mal, self.A )
            self.A = <double*>0

            Allocator_free( self.mal, self.w )
            self.w = <double*>0
            Allocator_free( self.mal, self.c )
            self.c = <double*>0

            Allocator_free( self.mal, self.r2o )
            self.r2o = <int*>0
            Allocator_free( self.mal, self.o2r )
            self.o2r = <int*>0
        else:
            # In guest mode, the allocation of these arrays is managed by the host Case instance.
            self.ipiv      = <int*>0
            self.col_scale = <double*>0
            self.row_scale = <double*>0
            self.A         = <double*>0
            self.w         = <double*>0
            self.c         = <double*>0
            self.r2o       = <int*>0
            self.o2r       = <int*>0

# Destructor.
#
cdef void Case_del( Case* self ) nogil:
    if self != <Case*>0:
        Case_deallocate( self )

        if not self.have_manager:
            Allocator_del( self.mal )

        free( self )

