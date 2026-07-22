'''various utility classes used by the library'''
import numpy as np

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
    
class DatasetHeaders(list):
    def __init__(self,attrs):
        self.indexes = {}
        self.headnames = [0]*len(attrs)
        for index, name in enumerate(attrs):
            self.headnames[index]=name
            self.indexes[name]=index
        super().__init__(self.headnames)
    # Getting indices by passing a list of column names or getting a list of column names by passing a list of indices
    def __call__(self,index_value): 

        if isinstance(index_value,list):
            list_ind = [0]*len(self.headnames)
            if isinstance(index_value[0],int):
                for i,ind in enumerate(index_value):
                    list_ind[i] = self.headnames[ind]
            elif isinstance(index_value[0],str):
                for i,ind in enumerate(index_value):
                    list_ind[i]=self.indexes[ind]
            return list_ind
        
        else:
            if isinstance(index_value,int):
                return self.headnames[index_value]
            elif isinstance(index_value,str):
                return self.indexes[index_value]
    # Code below for using class as list        
    def __len__(self):
        return len(self.headnames)
    def __getitem__(self,index):
        return self.headnames[index]
    def __iter__(self):
        return iter(self.headnames)
    def __contains__(self, item):
        return item in self.headnames

class AdaptiveParameter():
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
    # Методы для работы с арифметическими операциями
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
    # Методы для сравнения
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
    
# Индексаторы
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
        
        # Если результат — двумерная матрица, то возвращаем её как Indexator
        if isinstance(res, np.ndarray) and len(res.shape) == 2:
            return res.view(Indexator)
            
        # Если это строка, столбец или скаляр (число), возвращаем как обычный NumPy-объект
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