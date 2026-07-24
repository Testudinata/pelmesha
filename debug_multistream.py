"""
Debug script to test the hypothesis about _multistream_pipeline hanging.

Hypothesis: _procfunc_wrapper and _peakpick_wrapper are GENERATORS (they use `yield`),
but Pool.imap_unordered expects a regular function that returns a value.
When imap_unordered calls a generator function, it gets a generator object back,
NOT the yielded values.
"""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
from multiprocessing import Pool
from functools import partial
import time


def generator_func(x):
    """A generator function (uses yield)"""
    print(f"  WORKER: generator_func({x}) called, PID={os.getpid()}")
    result = x * 2
    print(f"  WORKER: about to yield {result}")
    yield result
    print(f"  WORKER: after yield (this should NOT print if consumed correctly)")


def regular_func(x):
    """A regular function (uses return)"""
    print(f"  WORKER: regular_func({x}) called, PID={os.getpid()}")
    result = x * 2
    print(f"  WORKER: returning {result}")
    return result


def wrapper_with_yield(x, multiplier=2, **kwargs):
    print(f"  WORKER: wrapper_with_yield({x}) called, PID={os.getpid()}")
    result = x * multiplier
    print(f"  WORKER: yielding {result}")
    yield result
    print(f"  WORKER: after yield")


class MockDataSource:
    def __init__(self, name):
        self.name = name
        self.metadata = {'continuous': True}
    def _get_local_roi_idx(self, idxs):
        return np.array([[0, 10]])


def process_function(mz, data_int, configs, **kwargs):
    return mz, data_int * 2


def procfunc_wrapper(idxs, datasource, configs, dtypeconv=None, **internal_configs):
    """Exact copy of the pattern from configs.py"""
    process_function_local = internal_configs.pop("process_pipeline")
    loc_idxs = datasource._get_local_roi_idx(idxs)
    if datasource.metadata['continuous']:
        mz = np.array([100.0, 200.0])
        data_int = np.array([[1.0, 2.0], [3.0, 4.0]])
        mz, data_int = process_function_local(mz, data_int, configs, **internal_configs)
        print(f"  WORKER PID={os.getpid()}: yielding result")
        yield loc_idxs, data_int
        print(f"  WORKER PID={os.getpid()}: after yield (SHOULD NOT PRINT)")


if __name__ == '__main__':
    # ============================================================
    # TEST 1: What does imap_unordered do with a generator function?
    # ============================================================
    print("=" * 70)
    print("TEST 1: imap_unordered with a generator function")
    print("=" * 70)

    print("\n--- Test 1a: Regular function with imap_unordered ---")
    with Pool(2) as p:
        results = list(p.imap_unordered(regular_func, [1, 2, 3]))
        print(f"  Results: {results}")

    print("\n--- Test 1b: Generator function with imap_unordered ---")
    with Pool(2) as p:
        results = list(p.imap_unordered(generator_func, [1, 2, 3]))
        print(f"  Results: {results}")
        print(f"  Result types: {[type(r).__name__ for r in results]}")
        for i, r in enumerate(results):
            if hasattr(r, '__next__'):
                try:
                    val = next(r)
                    print(f"  Iterated result[{i}]: got {val}")
                except Exception as e:
                    print(f"  Iterated result[{i}]: ERROR {type(e).__name__}: {e}")

    # ============================================================
    # TEST 2: What happens with partial + generator?
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: partial + generator function")
    print("=" * 70)

    partial_gen = partial(wrapper_with_yield, multiplier=3)

    with Pool(2) as p:
        results = list(p.imap_unordered(partial_gen, [10, 20, 30]))
        print(f"  Results: {results}")
        print(f"  Result types: {[type(r).__name__ for r in results]}")

    # ============================================================
    # TEST 3: Simulate the exact _procfunc_wrapper pattern
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: Simulating exact _procfunc_wrapper pattern")
    print("=" * 70)

    print("\n--- Test 3a: Generator wrapper with imap_unordered ---")
    internal = {"process_pipeline": process_function}
    partial_proc = partial(procfunc_wrapper, 
                           datasource=MockDataSource("test"),
                           configs={},
                           dtypeconv=np.float64,
                           **internal)

    with Pool(2) as p:
        results = list(p.imap_unordered(partial_proc, [np.array([[0, 10]])]))
        print(f"  Results: {results}")
        print(f"  Result types: {[type(r).__name__ for r in results]}")
        for i, r in enumerate(results):
            print(f"  Result[{i}] is generator: {hasattr(r, '__next__')}")
            if hasattr(r, '__next__'):
                try:
                    val = next(r)
                    print(f"  >> Iterated result[{i}]: {val}")
                except Exception as e:
                    print(f"  >> ERROR iterating: {type(e).__name__}: {e}")

    print("\nDone. Check output above to confirm hypothesis.")