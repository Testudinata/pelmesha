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