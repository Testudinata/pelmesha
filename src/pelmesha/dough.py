'''various utility classes used by the library'''
import numpy as np
from typing import Any
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import warnings
import queue
import threading
from multiprocessing import Pool, cpu_count
from itertools import count

class LinkedList(np.ndarray):
    """
    NumPy ndarray subclass that keeps a second "linked" array in sync for
    element-wise updates and structural operations (sort, delete, reshape).

    Attributes
    ----------
    linked_array : array_like or None
        Secondary array kept in sync with the primary array.

    Methods
    -------
    sync_sort()
        Sort values and apply the same permutation to `linked_array`.
    sync_delete(index)
        Delete items by index in both arrays and return a new object when a
        `linked_array` is present.
    sync_reshape(size)
        Reshape both arrays consistently and return a new object when a
        `linked_array` is present.
    """
    def __new__(cls, input_array, linked_array=None):
        """
        Create a LinkedList view over an input array and attach an optional
        linked array.

        Parameters
        ----------
        input_array : array_like
            Data used to construct the primary ndarray view.
        linked_array : array_like or None, optional
            A secondary array to keep in sync with the primary array. Should be
            broadcast-compatible for the operations used (typically 1-D with the
            same length as `input_array`).

        Returns
        -------
        LinkedList
            An instance viewing `input_array` with a `linked_array` attribute
            set to the provided secondary array.
        """
        obj = np.asarray(input_array).view(cls)
        obj.linked_array = linked_array
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the view creation, propagating the `linked_array` attribute.

        Parameters
        ----------
        obj : ndarray or None
            The source object from which the view is derived. When None, the
            method is called from `__new__` and no action is required.

        Returns
        -------
        None
            This method does not return a value.
        """
        if obj is None: return
        self.linked_array = getattr(obj, 'linked_array', None)

    def __setitem__(self, index, value):
        """
        Set items and mirror the assignment into the linked array if present.

        Parameters
        ----------
        index : int, slice, or array_like
            Index specification for item assignment.
        value : Any
            Value(s) to assign at the specified index.

        Returns
        -------
        None
            This method does not return a value.
        """
        super().__setitem__(index, value)
        if self.linked_array is not None:
            self.linked_array[index] = value

    def sync_sort(self):
        """
        Sort the array in ascending order and apply the same permutation to the
        linked array.

        Returns
        -------
        None
            In-place operation; no return value.
        """
        sort_indices = np.argsort(self)
        sorted_self = self[sort_indices]
        sorted_linked = self.linked_array[sort_indices]

        self[:] = sorted_self
        self.linked_array[:] = sorted_linked

    def sync_delete(self, index):
        """
        Delete the specified index/indices from the array and its linked array.

        Parameters
        ----------
        index : int, slice, or array_like
            Indices to remove. Passed to ``np.delete``.

        Returns
        -------
        LinkedList or ndarray
            A new `LinkedList` with the specified entries removed when a
            `linked_array` is present; otherwise, a regular ndarray.
        """
        new_self = np.delete(self, index)
        if self.linked_array is not None:
            new_linked_array = np.delete(self.linked_array, index, axis=0)
            return LinkedList(new_self, new_linked_array)
        return new_self

    def sync_reshape(self,size):
        """
        Reshape both the array and the linked array to the specified size.

        Parameters
        ----------
        size : tuple of int
            New shape to apply to both arrays. Must be compatible with the
            number of elements.

        Returns
        -------
        LinkedList or ndarray
            A new `LinkedList` if a `linked_array` exists; otherwise, a regular
            ndarray.
        """
        new_self = np.reshape(self, size)
        if self.linked_array is not None:
            new_linked_array = np.reshape(self.linked_array, size)
            return LinkedList(new_self, new_linked_array)
        return new_self

    def sync_split(self,indices_or_sections,axis=0):
        """
        Split both the array and the linked array into multiple sub-arrays.

        Parameters
        ----------
        indices_or_sections : int or 1-D array_like
            If an integer, it indicates the number of equal splits to make.
            If an array, it indicates the indices at which to split.
        axis : int, optional
            Axis along which to split. Default is 0.

        Returns
        -------
        list of LinkedList or ndarray
            A list of `LinkedList` objects if a `linked_array` exists;
            otherwise, a list of regular ndarray arrays.
        """
        split_self = np.array_split(self, indices_or_sections, axis=axis)
        if self.linked_array is not None:
            split_linked = np.array_split(self.linked_array, indices_or_sections, axis=axis)
            return [LinkedList(s, l) for s, l in zip(split_self, split_linked)]
        return split_self
    
class AdaptiveParameter():
    """A parameter whose value is resolved dynamically by an adaptation rule.

    Holds a base *parameter* and an *adaptation_rule*. When the instance is
    called, the rule is applied to produce the resolved ``implicit`` value;
    until then the object behaves as the base *parameter*. Arithmetic,
    comparison, and array operations are proxied to the current ``implicit``
    value.
    """
    def __init__(self, parameter, adaptation_rule):
        self.parameter = parameter
        self.adaptation_rule = adaptation_rule
        self.implicit = None
    def __index__(self):
        return self.implicit if self.implicit is not None else self.parameter
    def __repr__(self):
        return f"{self.implicit}" if self.implicit is not None else f"{self.parameter}"

    def __call__(self, *args, **kwargs):
        if callable(self.implicit) and kwargs: # TODO: weakspot with kwargs
            return self.implicit(*args, **kwargs)
        if callable(self.adaptation_rule):
            self.implicit = self.adaptation_rule(self.parameter, *args)
        else:
            self.implicit = self.parameter
        return self.implicit
    def __len__(self):
        if callable(self.implicit): # TODO: weakspot for None len
            return True
        elif self.implicit is None:
            return False
        return len(self.implicit)
    def __array__(self):
        return np.array(self.implicit)
    # Arithmetic operations
    def __add__(self, other):
        return self.implicit + other
    def __sub__(self, other):
        return self.implicit  - other
    def __mul__(self, other):
        return self.implicit * other
    def __truediv__(self, other):
        return self.implicit / other
    def __floordiv__(self, other):
        return self.implicit // other
    def __mod__(self, other):
        return self.implicit % other
    def __pow__(self, other):
        return self.implicit ** other
    # Comparison operations
    def __eq__(self, other):
        return self.implicit == other
    def __ne__(self, other):
        return self.implicit != other
    def __lt__(self, other):
        return self.implicit < other
    def __le__(self, other):
        return self.implicit <= other
    def __gt__(self, other):
        return self.implicit > other
    def __ge__(self, other):
        return self.implicit >= other
    
# Indexers
class Indexator(np.ndarray):
    """
    A numpy ndarray subclass that represents a collection of index segments ``(start, end)``
    and provides iteration over individual indices.

    :param idxs: Index segments as a 2-D array of shape ``(n, 2)`` or a 1-D array of length 2.
    :type idxs: np.ndarray or list

    :raises ValueError: If a 1-D array does not have exactly 2 elements.
    """
    def __new__(cls, idxs):
        if not isinstance(idxs, np.ndarray):
            idxs = np.array(idxs, dtype=np.int64)
        if len(idxs.shape) == 1:
            if idxs.shape[0] != 2:
                raise ValueError('Indexes must be a 2D array with shape (n, 2)')
            idxs = idxs[np.newaxis, :]
        return np.asarray(idxs, dtype=np.int64).view(cls)
    def __getitem__(self, index):
        res = super().__getitem__(index)
        
        # If the result is a 2-D matrix, return it as an Indexator.
        if isinstance(res, np.ndarray) and len(res.shape) == 2:
            return res.view(Indexator)
            
        # For a row, column, or scalar, return a plain NumPy object.
        return res.view(np.ndarray) if isinstance(res, np.ndarray) else res
    
    @property
    def count(self):
        """
        Return the total number of individual indices across all segments.

        :return: Total count of indices.
        :rtype: int
        """
        full_size = 0
        for segment in self.view(np.ndarray):
            full_size += np.diff(segment)
        return full_size[0]
    def __iter__(self):
        for start, end in self.view(np.ndarray):
            yield from range(start, end)
class SliceIndexator(Indexator):
    """
    An :class:`Indexator` subclass that yields Python ``slice`` objects instead of individual indices.

    :param idxs: Index segments as a 2-D array of shape ``(n, 2)`` or a 1-D array of length 2.
    :type idxs: np.ndarray or list
    """
    def __iter__(self):
        for start, end in self.view(np.ndarray):
            yield slice(start, end)

def _run_with_file_ids(func, file_ids, *args, **kwargs):
    file_id, roi = file_ids
    return file_id, roi, func(*args, **kwargs)
def _skipped_roi_marker(payload):
    file_id, roi = payload
    return file_id, roi, None


@dataclass(slots=True)
class PartState:
    """
    Support class for MultiProcessorManager with meta and progress bar for each part of the job.
    """
    n_batches: int
    meta: Any = None
    pbar: Any = None

@dataclass(slots=True)
class JobState:
    """
    Support class for MultiProcessorManager with meta, file writer and process state.
    """
    n_parts: int
    done: int = 0
    meta: Any = None
    writer: Any = None
    parts: dict[Any, PartState] = field(default_factory=dict)

class MultiProcessorManager:
    """
    Manager for multi-processor workflows with progress bars.
    
    Parameters
    ----------
    dataset : DataSet
        The dataset to process.
    cpu_count : int
        The number of CPUs to use.
    """

    def __init__(self, 
                 main_bar: tqdm = None,
                 cpu_count: int = 1):
        self.cpu_count = cpu_count
        self.main_pbar = main_bar
        self.jobs = {}
        self._jobs_lock = threading.Lock()
        if cpu_count > 1:
            self._pool = Pool(self.cpu_count)
        else:
            self._pool = None
        self.batcher = None
        self._batch_queue = queue.Queue(maxsize = 3)

        self._id_counter = count()
        self._id_counter_lock = threading.Lock()

        self._cancel = threading.Event()        # emergency termination flag
        self._batcher_error = None              # excluding from thread
        self._closed = False                    # double-close protection
        
    def __iter__(self):
        self.batcher.start()
        if self.cpu_count > 1:
            for worker, batches in self.batch_generator():
                yield from self._pool.imap_unordered(worker, batches)
        else:
            for partial_function, batch in self.batch_generator(): 
                for b in batch:
                    yield partial_function(b) 
    
    def _batcher_runner(self, target, args, kwargs):
        try:
            target(self, *args, **kwargs)
        except BaseException as e:
            self._batcher_error = e
            self._cancel.set()
        finally:
            try:
                self._batch_queue.put_nowait(None)
            except queue.Full:
                pass
    def set_batching_thread(self, 
                              batching_function: callable,
                              *args,
                              **kwargs):
        self.batcher = threading.Thread(
            target=self._batcher_runner, 
            args=(batching_function, args, kwargs),
            daemon=True)

    def batch_generator(self):
        while True:
            if self._batcher_error is not None:
                raise RuntimeError("batcher thread failed") from self._batcher_error
            try:
                batch = self._batch_queue.get(timeout=0.1)
            except queue.Empty:
                if self._batcher_error is not None:
                    raise RuntimeError("batcher thread failed") from self._batcher_error
                if self.batcher is None or not self.batcher.is_alive():
                    break
                continue

            if batch is None:
                break            
            yield batch

    def register_job(self, job_id: int, n_parts: int, meta=None) -> None:
        with self._jobs_lock:
            if job_id in self.jobs:
                raise ValueError(f"job {job_id} already registered")
            self.jobs[job_id] = JobState(n_parts=n_parts, meta=meta)

    def add_part(self, job_id: int, part_id, n_batches: int, meta: Any = None) -> None:
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise ValueError(f"job {job_id} is not registered")
            job.parts[part_id] = PartState(
                n_batches=n_batches, meta=meta
            )

    def finish_part(self, job_id: int) -> bool:
        """
        Finish a part of a job.
        Returns True if all parts are finished.
        Calling ONLY in main thread.
        """

        job = self.jobs.get(job_id)
        if job is None:
            return False
        job.done += 1
        return job.done == job.n_parts
    
    def release_job(self, job_id: int) -> None:
        job = self.jobs.pop(job_id, None)
        if job is None:
            return
        if job.writer is not None:
            job.writer.close()
            job.writer = None
        for part in job.parts.values():
            if part.pbar is not None:
                part.pbar.close()
                part.pbar = None
        self.update()
        
    def load_journal(self):
        pass
    def create_journal(self):
        pass
    def add_to_journal(self):
        pass
    def delete_journal(self):
        pass
    def create_secondary_pbar(self,
                               job_id,
                               part_id,
                               desc: str = "File batches processed", 
                               unit: str = "batches"):
        # get free position for bar
        with tqdm.get_lock():
            busy = {abs(bar.pos) for bar in tqdm._instances}
            pos = 0
            while pos in busy:
                pos += 1
        # create secondary progress bar at position "pos"
        job = self.jobs.get(job_id, None)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        part = job.parts.get(part_id, None)
        if part is None:
            raise ValueError(f"Part {part_id} of job {job_id} not found")
        pbar = part.pbar
        if pbar is None or pbar.disable or pbar.n >= pbar.total:
            if pbar is not None and not pbar.disable:
                pbar.close()
            part.pbar = tqdm(total=part.n_batches,
                             desc=desc,
                             unit=unit,
                             leave=False,
                             position=pos)
        else:
            raise ValueError("Secondary progress bar already exists")

    def batch_queue_put(self, item, timeout: float = 0.1) -> bool:
        while not self._cancel.is_set():
            try:
                self._batch_queue.put(item, timeout=timeout)
                return True
            except queue.Full:
                continue
        return False
    
    def get_id(self):
        if self._cancel.is_set():
            return None
        with self._id_counter_lock:
            return next(self._id_counter)
        
    def secondary_pbar_update(self, job_id, part_id, n: int = 1):
        job = self.jobs.get(job_id)
        if job is None:
            return False
        part = job.parts.get(part_id)
        if part is None or part.pbar is None:
            return False
        pbar = part.pbar
        pbar.update(n)
        if pbar.n >= pbar.total:
            pbar.close()
            part.pbar = None
            return True
        return False
    
    def update(self, n: int = 1):
        if self.main_pbar is None:
            return
        main_pbar = self.main_pbar
        main_pbar.update(n)
        if main_pbar.n > main_pbar.total:
            raise ValueError("Main progress exceeded total num of batches") # TODO - пока для дебагинга
    
    def close(self):
        if self._closed:
            return
        self._closed = True

        # Stoping producers
        self._cancel.set()
        if self.batcher:
            self.batcher.join(timeout = 5)
            if self.batcher.is_alive():
                warnings.warn("Batcher thread did not stop in 5 s; continuing shutdown")

        # closing all resources in jobs
        for job in list(self.jobs.values()):
            if job.writer is not None:
                job.writer.close()
                job.writer = None
            for part in job.parts.values():
                if part.pbar is not None:
                    part.pbar.close()
                    part.pbar = None
        self.jobs.clear()

        if self.main_pbar:
            self.main_pbar.close()

        # 3) Closing pool
        if self._pool is not None:
            if self._batcher_error is not None:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()