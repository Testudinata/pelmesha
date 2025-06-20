"""
ProcceMSI

ProcceMSI is a Python library for loading, preprocessing, processing and storage of mass spectra Mass Spectrometry Imaging (MSI) data.
"""

__version__ = "0.1.0"
__author__ = 'Andrey Kuzin'
__credits__ = 'Moscow Institue of Physics and Technology'

"""Signal calibration and alignment by reference peaks - copy of MSALIGN function from MATLAB bioinformatics library."""
from typing import List

import numpy as np

try:
    from ._version import version as __version__  # noqa: F401
except ImportError:
    __version__ = "unknown"
from .align import Aligner

__all__ = ["msalign", "Aligner"]


def msalign(
    x: np.ndarray,
    array: np.ndarray,
    peaks: List,
    method: str = "cubic",
    width: float = 10,
    ratio: float = 2.5,
    resolution: int = 100,
    iterations: int = 5,
    grid_steps: int = 20,
    shift_range: List = None,
    weights: List = None,
    return_shifts: bool = False,
    align_by_index: bool = False,
    only_shift: bool = False,
):
    aligner = Aligner(
        x,
        array,
        peaks,
        method=method,
        width=width,
        ratio=ratio,
        resolution=resolution,
        iterations=iterations,
        grid_steps=grid_steps,
        shift_range=shift_range,
        weights=weights,
        return_shifts=return_shifts,
        align_by_index=align_by_index,
        only_shift=only_shift,
    )
    aligner.run()
    return aligner.apply()


msalign.__doc__ = Aligner.__doc__
