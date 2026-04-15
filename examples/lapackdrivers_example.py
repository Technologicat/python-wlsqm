# -*- coding: utf-8 -*-
#
"""Performance benchmarking and usage examples for the wlsqm.utils.lapackdrivers module.

JJ 2016-11-02
"""


import os
import time

import numpy as np
from numpy.linalg import solve as numpy_solve  # for comparison purposes

import matplotlib.pyplot as plt

# Where to write the timing-plot output files. Anchor to the project root
# (one level up from this script) so the PNG and PDF land in a stable,
# version-controllable location regardless of where the example was
# invoked from. README.md embeds `lapack_timings.png` from here.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

try:
    import wlsqm.utils.lapackdrivers as drivers
except ImportError:
    import sys
    sys.exit( "WLSQM not found; is it installed?" )

# from find_neighbors2.py
class SimpleTimer:
    def __init__(self, label="", n=None):
        self.label = label
        self.n     = n      # number of repetitions done inside the "with..." section (for averaging in timing info)

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, errtype, errvalue, traceback):
        dt         = time.time() - self.t0
        identifier = ("%s" % self.label) if len(self.label) else "time taken: "
        avg        = (", avg. %gs per run" % (dt/self.n)) if self.n is not None else ""
        print( "%s%gs%s" % (identifier, dt, avg) )

# from util.py
def f5(seq, idfun=None):
   """Uniqify a list (remove duplicates).

   This is the fast order-preserving uniqifier "f5" from
   http://www.peterbe.com/plog/uniqifiers-benchmark

   The list does not need to be sorted.

   The return value is the uniqified list.

   """
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result



def main():
    # Seed the legacy global RNG used by `np.random.sample(...)` calls below
    # so that the residual-check thresholds are reproducible across runs and
    # across machines. Without this, each run draws different random
    # matrices, the conditioning varies wildly, and the per-solver residuals
    # can swing by orders of magnitude run-to-run.
    np.random.seed(42)

#    # exact solution is (3/10, 2/5, 0)
#    A = np.array( ( (2., 1.,  3.),
#                    (2., 6.,  8.),
#                    (6., 8., 18.) ), dtype=np.float64, order='F' )
#    b = np.array(   (1., 3., 5.),    dtype=np.float64 )

#    # symmetric matrix for testing symmetric solver
#    A = np.array( ( (2., 1.,  3.),
#                    (1., 6.,  8.),
#                    (3., 8., 18.) ), dtype=np.float64, order='F' )
#    b = np.array(   (1., 3., 5.),    dtype=np.float64 )

    # random matrix
    n = 5
    A = np.random.sample( (n,n) )
    A = np.array( A, dtype=np.float64, order='F' )
    drivers.symmetrize( A )  # fast Cython implementation of  A = 0.5 * (A + A.T)
    b = np.random.sample( (n,) )

    # test that it works

    x = numpy_solve(A, b)
    print( "NumPy:", x )

    A2 = A.copy(order='F')
    x2 = b.copy()
    drivers.symmetric(A2, x2)
    print( "dsysv:", x2 )

    A3 = A.copy(order='F')
    x3 = b.copy()
    drivers.general(A3, x3)
    print( "dgesv:", x3 )

    assert (np.abs(x - x3) < 1e-10).all(), "Something went wrong, solutions do not match"  # check general solver first
    assert (np.abs(x - x2) < 1e-10).all(), "Something went wrong, solutions do not match"  # then check symmetric solver


    # test performance

    # for verification only - very slow (serial only!)
    use_numpy = True

    # parallel processing
    ntasks = 8

#    # overview, somewhat fast but not very accurate
#    sizes = f5( map( lambda x: int(x), np.ceil(3*np.logspace(0, 3, 21, dtype=int)) ) )
#    reps = map( lambda x: int(x), 10.**(4 - np.log10(sizes)) )

#    # "large n"
#    sizes = f5( map( lambda x: int(x), np.ceil(3*np.logspace(2, 3, 21, dtype=int)) ) )
#    reps = map( lambda x: int(x), 10.**(5 - np.log10(sizes)) )

    # "small n" (needs more repetitions to eliminate noise from other running processes since a single solve is very fast)
    sizes = f5( map( lambda x: int(x), np.ceil(3*np.logspace(0, 2, 21, dtype=int)) ) )
    reps = map( lambda x: int(x), 10.**(6 - np.log10(sizes)) )

    print( "performance test: %d tasks, sizes %s" % (ntasks, sizes) )

    results1 = np.empty( (len(sizes),), dtype=np.float64 )
    results2 = np.empty( (len(sizes),), dtype=np.float64 )
    results3 = np.empty( (len(sizes),), dtype=np.float64 )
    results4 = np.empty( (len(sizes),), dtype=np.float64 )
    results5 = np.empty( (len(sizes),), dtype=np.float64 )
    results6 = np.empty( (len(sizes),), dtype=np.float64 )
    results7 = np.empty( (len(sizes),), dtype=np.float64 )

#    # many LHS (completely independent problems)
#    n = 5
#    reps=int(1e5)
#    A = np.random.sample( (n,n,reps) )
#    A = 0.5 * (A + A.transpose(1,0,2))  # symmetrize
#    A = np.array( A, dtype=np.float64, order='F' )
#    b = np.random.sample( (n,reps) )
#    b = np.array( b, dtype=np.float64, order='F' )
#    with SimpleTimer(label="msymmetric ", n=reps) as s:
#        drivers.msymmetricp(A, b, ntasks)
#    with SimpleTimer(label="mgeneral ", n=reps) as s:
#        drivers.mgeneralp(A, b, ntasks)


    for j,item in enumerate(zip(sizes,reps)):
        n,r = item
        print( "testing size %d, reps = %d" % (n, r) )

        # same LHS, many different RHS

        print( "    prep same LHS, many RHS..." )

        A = np.random.sample( (n,n) )
        # symmetrize
#        A *= 0.5
#        A += A.T  # not sure if this works
        A = np.array( A, dtype=np.float64, order='F' )
#        A = 0.5 * (A + A.T)  # symmetrize
        drivers.symmetrize(A)
        b = np.random.sample( (n,r) )
        b = np.array( b, dtype=np.float64, order='F' )

        print( "    solve:" )

#        # for verification only - very slow (Python loop, serial!)
#        if use_numpy:
#            t0 = time.time()
#            x = np.empty( (n,r), dtype=np.float64 )
#            for k in range(r):
#                x[:,k] = numpy_solve(A, b[:,k])
#            results1[j] = (time.time() - t0) / r

        print( "        symmetricsp" )
        t0 = time.time()
        A2 = A.copy(order='F')
        x2 = b.copy(order='F')
        drivers.symmetricsp(A2, x2, ntasks)
        results2[j] = (time.time() - t0) / r

        print( "        generalsp" )
        t0 = time.time()
        A3 = A.copy(order='F')
        x3 = b.copy(order='F')
        drivers.generalsp(A3, x3, ntasks)
        results3[j] = (time.time() - t0) / r

        # different LHS for each problem

        print( "    prep independent problems..." )

        A = np.random.sample( (n,n,r) )
        # symmetrize
#        A *= 0.5
#        A += A.transpose(1,0,2)  # this doesn't work
        A = np.array( A, dtype=np.float64, order='F' )
#        A = 0.5 * (A + A.transpose(1,0,2))
        drivers.msymmetrizep(A, ntasks)
        b = np.random.sample( (n,r) )
        b = np.array( b, dtype=np.float64, order='F' )

        print( "    solve:" )

        # for verification only - very slow (Python loop, serial!)
        if use_numpy:
            print( "        NumPy" )
            t0 = time.time()
            x = np.empty( (n,r), dtype=np.float64, order='F' )
            for k in range(r):
                x[:,k] = numpy_solve(A[:,:,k], b[:,k])
            results1[j] = (time.time() - t0) / r

        print( "        msymmetricp" )
        t0 = time.time()
        A2 = A.copy(order='F')
        x2 = b.copy(order='F')
        drivers.msymmetricp(A2, x2, ntasks)
        results4[j] = (time.time() - t0) / r

        print( "        mgeneralp" )
        t0 = time.time()
        A3 = A.copy(order='F')
        x3 = b.copy(order='F')
        drivers.mgeneralp(A3, x3, ntasks)
        results5[j] = (time.time() - t0) / r

        print( "        msymmetricfactorp & msymmetricfactoredp" )  # factor once, then it is possible to solve multiple times (although we now test only once)
        t0 = time.time()
        ipiv = np.empty( (n,r), dtype=np.intc, order='F' )
        fact = A.copy(order='F')
        x4   = b.copy(order='F')
        drivers.msymmetricfactorp( fact, ipiv, ntasks )
        drivers.msymmetricfactoredp( fact, ipiv, x4, ntasks )
        results6[j] = (time.time() - t0) / r

        print( "        mgeneralfactorp & mgeneralfactoredp" )  # factor once, then it is possible to solve multiple times (although we now test only once)
        t0 = time.time()
        ipiv = np.empty( (n,r), dtype=np.intc, order='F' )
        fact = A.copy(order='F')
        x5   = b.copy(order='F')
        drivers.mgeneralfactorp( fact, ipiv, ntasks )
        drivers.mgeneralfactoredp( fact, ipiv, x5, ntasks )
        results7[j] = (time.time() - t0) / r

        # Verify each solver produced a valid solution.
        #
        # The right sanity check for a linear solver is the relative residual
        # ‖A x − b‖ / ‖b‖, NOT ‖x − x_reference‖. Two valid LAPACK calls on a
        # moderately ill-conditioned matrix can produce solutions that differ
        # at, say, 1e-6 — that is a property of the conditioning, not a bug —
        # while both still have a tiny residual ‖A x − b‖ ~ machine epsilon.
        # This matters here because msymmetrizep produces matrices with
        # κ(A) ~ 1e4 at n=117, and the historical (vs-NumPy) check used to
        # trip nondeterministically as a function of the unseeded RNG.
        #
        # Threshold rationale: 1e-8 covers both DGESV and DSYSV across the
        # whole size range used here (max n ~ 117). DSYSV (Bunch-Kaufman)
        # tends to produce noticeably larger residuals than DGESV on
        # indefinite matrices because of its different pivoting strategy,
        # and the random `(U + U.T) / 2` matrices we generate here are
        # almost always indefinite and moderately ill-conditioned. 1e-8 is
        # still 8 orders of magnitude above machine epsilon and tight
        # enough to catch any realistic regression in the Cython wrappers.
        #
        # einsum 'ijk,jk->ik' computes A[:,:,k] @ x[:,k] for each problem
        # instance k, vectorized.
        b_norm = np.linalg.norm(b, axis=0)
        b_norm = np.maximum(b_norm, 1.0)  # guard against the all-zero RHS edge case
        for label, solver_x in (("msymmetricp",                x2),
                                ("mgeneralp",                  x3),
                                ("msymmetricfactorp+factored", x4),
                                ("mgeneralfactorp+factored",   x5)):
            residual = np.linalg.norm(np.einsum('ijk,jk->ik', A, solver_x) - b, axis=0) / b_norm
            worst = residual.max()
            assert worst < 1e-8, f"{label}: relative residual {worst:.3e} exceeds 1e-8"
            print(f"        {label} max relative residual: {worst:.3e}")


# old, serial only
#
#    for j,item in enumerate(zip(sizes,reps)):
#        n,r = item
#        print( "testing size %d, reps = %d" % (n, r) )
#
#        A = np.random.sample( (n,n) )
#        A = 0.5 * (A + A.T)  # symmetrize
#        A = np.array( A, dtype=np.float64, order='F' )
#        b = np.random.sample( (n,) )
#
#        t0 = time.time()
#        for k in range(r):
#            x = numpy_solve(A, b)
#        results1[j] = (time.time() - t0) / r
#
#        t0 = time.time()
#        for k in range(r):
#            A2 = A.copy(order='F')
#            x2 = b.copy()
#            drivers.symmetric(A2, x2)
#        results2[j] = (time.time() - t0) / r
#
#        t0 = time.time()
#        for k in range(r):
#            A3 = A.copy(order='F')
#            x3 = b.copy()
#            drivers.general(A3, x3)
#        results3[j] = (time.time() - t0) / r


    # visualize

    plt.figure(1)
    plt.clf()
    if use_numpy:
        plt.loglog(sizes, results1, 'k-', label='NumPy')
    plt.loglog(sizes, results2, 'b--', label='dsysv, same LHS, many RHS')
    plt.loglog(sizes, results3, 'b-',  label='dgesv, same LHS, many RHS')
    plt.loglog(sizes, results4, 'r--', label='dsysv, independent problems')
    plt.loglog(sizes, results5, 'r-',  label='dgesv, independent problems')
    plt.loglog(sizes, results6, 'g--', label='dsytrf+dsytrs, independent problems')
    plt.loglog(sizes, results7, 'g-',  label='dgetrf+dgetrs, independent problems')
    plt.xlabel('n')
    plt.ylabel('t')
    plt.title('Average time per problem instance, %d parallel tasks' % (ntasks))
    plt.axis('tight')
    plt.grid(visible=True, which='both')
    plt.legend(loc='best')

    # Save both formats: PNG for embedding in README.md, PDF for printing
    # / vector-quality reuse. Both go to the project root.
    plt.savefig(os.path.join(PROJECT_ROOT, 'lapack_timings.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(PROJECT_ROOT, 'lapack_timings.pdf'),           bbox_inches='tight')


if __name__ == '__main__':
    main()
    plt.show()
