import numpy as np
import math
import os
import warnings
from KDEpy import FFTKDE, TreeKDE
from pybaselines import Baseline
from pelmesha.utensils import split_pdtable_by_peaks_gap, _set_KDE_X_plot, _peakpicker_core
from scipy.signal import savgol_filter
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from pelmesha import PipelineConfigurator, DataSource
    from pelmesha.cookbook import Configs

FWHM_TO_SIGMA_FACTOR = 1 / np.sqrt(8 * np.log(2))  # Conversion factor from FWHM to sigma.
###########################################
#   Base pipeline functions               #
###########################################

def preprocess_configuration_base(
    datasource: "DataSource",
    roi,
    rmeta,
    configs: "Configs | PipelineConfigurator"):
    """Preprocess the configuration for a given ROI.

    Parameters
    ----------
    datasource : DataSource
        The data source.
    roi : int
        The ROI index.
    rmeta : dict
        The ROI metadata.
    configs : Configs or PipelineConfigurator
        Configuration object holding the per-step parameters.

    Returns
    -------
    tuple of np.ndarray, list, dict
        The resampled m/z scale, headers list, and internal configuration.
    """
    
    preprocess_configs = configs.get_step_configs("preprocess")
    mz_range = datasource.roi_metadata.loc[roi,'mz_range']
    
    internal_configs = {}
    internal_configs['process'] = {}
    resampled_mz = resample_mz_scale(*mz_range, **preprocess_configs['resample_mz_scale'])
    internal_configs['process']['resampled_mz'] = resampled_mz
    internal_configs['process']['mz_discrete_coeffs'] = datasource.roi_metadata.loc[roi,'discret_coeffs']

    baseline_algo = configs.configs.get('methods', None)
    if baseline_algo:
        baseline_algo = baseline_algo.get('Baseline', None)
        if baseline_algo:
            baseline_algo = next(iter(baseline_algo))
    if baseline_algo:
        if datasource.dcont:
            internal_configs['process']['Baseliner'] = getattr(Baseline(datasource.get_mz(rmeta['idxroi'].ravel()[0]), **configs['Baseline'], assume_sorted = True), baseline_algo)
        else:
            internal_configs['process']['Baseliner'] = baseline_algo

    
    # Configure headers conditionally based on processing flags 
    if configs['peakpicker']["SNR_threshold"] and configs['peakpicker']["return_areas"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "Area", "SNR",  
                        "PextL", "PextR", "FWHM", "Noise", "Mean noise"]
        
    elif configs['peakpicker']["SNR_threshold"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "SNR",  
                        "PextL", "PextR", "FWHM", "Noise", "Mean noise"]
    elif configs['peakpicker']["return_areas"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "Area",  
                        "PextL", "PextR", "FWHM"]
    else: 
        headers_list = ["spectra_ind", "mz", "Intensity",  
                        "PextL", "PextR", "FWHM"]
    # internal_configs['peakpick'] = {"discret_coeffs": datasource.roi_metadata.loc[roi,'discret_coeffs'],
    internal_configs['peakpick'] = {"headers": headers_list}
    return resampled_mz, headers_list, internal_configs


def process_spectra_base(
    mz: np.ndarray,
    intensity: np.ndarray,
    configs: "Configs | PipelineConfigurator",
    **internal_configs) -> np.ndarray:
    """Run the full pre-processing cycle of a mass spectrum.

    The pipeline order is: smoothing → baseline correction → resampling →
    alignment.

    Parameters
    ----------
    mz : np.ndarray
        The m/z scale. Always a 1-D vector.
    intensity : np.ndarray
        The intensities. A matrix for continuous data and a vector for
        discontinuous data.
    configs : Configs or PipelineConfigurator
        Configuration object holding the per-step parameters.
    **internal_configs
        Internal per-step parameters (resampled m/z, baseliner, etc.).

    Returns
    -------
    tuple of np.ndarray
        The processed ``(mz, intensity)`` pair.
    """
    resampled_mz = internal_configs.get('resampled_mz', None)
    baseline_algo = internal_configs.get('Baseliner', None)
    mz_discrete_coeffs = internal_configs.get('mz_discrete_coeffs', None)

    smooth_configs = configs.get('smoothing', {})
    msalign_configs = configs.get('msalign', {})
    modify_raw_spectrum_configs = configs.get('modify_raw_spectrum', {})

    #premodifying spectrum
    if modify_raw_spectrum_configs.get('zero_points_to_peaks_ext') or modify_raw_spectrum_configs.get('mz_segments_to_zero'):
        mz, intensity = modify_raw_spectrum(mz, intensity, mz_discrete_coeffs, **modify_raw_spectrum_configs) 

    #Smoothing step
    if smooth_configs.get('smooth_algo') is not None: 
        intensity = smoothing(intensity, **smooth_configs)
    
    # BaselineCorrection step
    if baseline_algo is not None: 
        if isinstance(baseline_algo, str):
            if baseline_algo == 'asls': #Hack just for setting default params
                intensity = intensity - Baseline(mz, assume_sorted = True, **configs['Baseline']).asls(intensity, **configs['asls'])[0]
            else:
                intensity = intensity - getattr(Baseline(mz, assume_sorted = True,**configs['Baseline']), baseline_algo)(intensity, **configs[baseline_algo])[0]
        else:
            intensity = intensity - baseline_algo(intensity, **configs[baseline_algo.__name__])[0]
    
    # Resampling step
    if resampled_mz is not None:
        intensity = np.interp(resampled_mz, mz, intensity, left=intensity.ravel()[0], right=intensity.ravel()[-1])
        mz = resampled_mz
    
    # Aligning step
    if msalign_configs.get('align_peaks', None) is not None:
        intensity = msalign(mz, intensity, **msalign_configs)
    
    return mz, intensity

def peakpicking_base(
        mz: np.ndarray,
        intensity: np.ndarray,
        idx: int,
        configs: "Configs | PipelineConfigurator",
        **internal_configs) -> np.ndarray:
    """Base peak picking function used as adapter for the peakpicker function.

    Parameters
    ----------
    mz : np.ndarray
        The m/z values of the spectrum.
    intensity : np.ndarray
        The intensity values of the spectrum.
    idx : int
        The index of the spectrum in the data set.
    configs : Configs | PipelineConfigurator
        The configuration dictionary.
    
    Returns
    -------
    np.ndarray
        The peak picking result.
    """
    configs = configs['peakpicker']  # Directly get the config dict for the peakpicker function.
    return peakpicker(mz, 
                      intensity, 
                      idx, 
                    #   discret_coeffs=internal_configs['discret_coeffs'],
                      headers=internal_configs['headers'],
                      **configs)


###########################################
#   Base processing functions             #
###########################################
### Code from __init__.py msalign (https://github.com/lukasz-migas/msalign)
"""Signal calibration and alignment by reference peaks - copy of MSALIGN function from MATLAB bioinformatics library."""
from .align import Aligner
from typing import List
__all__ = ["msalign", "Aligner"]

def msalign(
    x: np.ndarray,
    array: np.ndarray,
    align_peaks: List | None = None, # if None - return array
    align_method: str = "cubic",
    align_width: float = 10,
    align_ratio: float = 2.5,
    align_resolution: int = 100,
    align_iterations: int = 5,
    align_grid_steps: int = 20,
    align_shift_range: List = [-0.95,0.95],
    align_pweights: List = None,
    return_shifts: bool = False,
    align_by_index: bool = False,
    only_shift: bool = False,
):
    """Signal calibration and alignment by reference peaks

    A simplified version of the MSALIGN function found in MATLAB (see references for link)

    This version of the msalign function accepts most of the parameters that MATLAB's function accepts with the
    following exceptions: GroupValue, ShowPlotValue. A number of other parameters is allowed, although they have
    been renamed to comply with PEP8 conventions. The Python version is 8-60 times slower than the MATLAB
    implementation, which is mostly caused by a really slow instantiation of the
    `scipy.interpolate.PchipInterpolator` interpolator. In order to speed things up, I've also included several
    other interpolation methods which are significantly faster and give similar results.

    References
    ----------
    Monchamp, P., Andrade-Cetto, L., Zhang, J.Y., and Henson, R. (2007) Signal Processing Methods for Mass
    Spectrometry. In Systems Bioinformatics: An Engineering Case-Based Approach, G. Alterovitz and M.F. Ramoni, eds.
    Artech House Publishers).
    MSALIGN: https://nl.mathworks.com/help/bioinfo/ref/msalign.html

    Parameters
    ----------
    x : np.ndarray
        1D array of separation units (N). The number of elements of xvals must equal the number of elements of
        zvals.shape[1]
    array : np.ndarray
        2D array of intensities that must have common separation units (M x N) where M is the number of vectors
        and N is number of points in the vector
    align_peaks : list / None
        list of reference peaks that must be found in the xvals vector. Default: None. If None - return array as is
    align_method : str
        interpolation method. Default: 'cubic'. MATLAB version uses 'pchip' which is significantly slower in Python
    align_pweights: list (optional)
        list of weights associated with the list of peaks. Must be the same length as list of peaks
    align_width : float (optional)
        width of the gaussian peak in separation units. Default: 10
    align_ratio : float (optional)
        scaling value that determines the size of the window around every alignment peak. The synthetic signal is
        compared to the input signal within these regions. Default: 2.5
    align_resolution : int (optional)
        Default: 100
    align_iterations : int (optional)
        number of iterations. Increasing this value will (slightly) slow down the function but will improve
        performance. Default: 5
    align_grid_steps : int (optional)
        number of steps to be used in the grid search. Default: 20
    align_shift_range : list or numpy.ndarray, optional
    The maximum allowed shift values in the m/z axis. If `align_by_index` or `only_shift` is set to True, 
    these shifts are measured in data points (indices) instead. Default: [-0.95, 0.95].
    only_shift : bool
        determines if signal should be shifted (True) or rescaled (False). Default: True
    return_shifts : bool
        decide whether shift parameter `shift_opt` should also be returned. Default: False
    align_by_index : bool
        decide whether alignment should be done based on index rather than `xvals` array. Default: False
    """
    if align_peaks is None:
        return array
    
    aligner = Aligner(
        x,
        array,
        align_peaks,
        method=align_method,
        width=align_width,
        ratio=align_ratio,
        resolution=align_resolution,
        iterations=align_iterations,
        grid_steps=align_grid_steps,
        shift_range=align_shift_range,
        weights=align_pweights,
        return_shifts=return_shifts,
        align_by_index=align_by_index,
        only_shift=only_shift,
    )
    aligner.run()
    return aligner.apply()


msalign.__doc__ = Aligner.__doc__

def resample_mz_scale(mz_min: float,
                      mz_max: float,
                      resample_mz_step: float = None,
                      resample_num_points: int = None):
    """Build a uniform m/z scale between ``mz_min`` and ``mz_max``.

    Exactly one of ``resample_mz_step`` or ``resample_num_points`` must be
    provided to define the resulting grid.

    Parameters
    ----------
    mz_min : float
        Lower bound of the m/z range.
    mz_max : float
        Upper bound of the m/z range.
    resample_mz_step : float, optional
        Step size between consecutive m/z points. When given,
        ``resample_num_points`` is computed from it. Default ``None``.
    resample_num_points : int, optional
        Total number of points in the resulting scale. Default ``None``.

    Returns
    -------
    np.ndarray or None
        A uniformly spaced m/z scale, or ``None`` if neither parameter is
        provided.

    Raises
    ------
    ValueError
        If both ``resample_mz_step`` and ``resample_num_points`` are given.
    """
    if resample_mz_step is not None and resample_num_points is not None:
        raise ValueError("Provide exactly one of 'resample_mz_step' or 'resample_num_points'.")
    if resample_mz_step is not None:
        resample_num_points = int(np.round((mz_max - mz_min)/resample_mz_step)) + 1
    if resample_num_points is not None:
        return np.linspace(mz_min, mz_max, resample_num_points)
    else:
        return
    
### Smoothing
def smoothing(y: np.ndarray, 
              smooth_algo: str = None, 
              smooth_window: int = 7, 
              smooth_cycles: int = 1) -> np.ndarray:
    if len(y) == 0:
        return np.array([])
    
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    if smooth_algo == 'MA':
        return _movaver(y, smooth_window, smooth_cycles)
    elif smooth_algo == 'GA':
        return _gaussian_filter(y, smooth_window, smooth_cycles)
    elif smooth_algo == 'SG':
        return _savgol(y, smooth_window, smooth_cycles)
    else:
        print("Smoothing algorithm not recognized")
        return y

def _movaver(y, window, cycles):
    if window < 3 or len(y) < window:
        return y.copy()
    
    pad_size = window // 2
    kernel = np.ones(window) / window
    
    for _ in range(cycles):
        padded = np.pad(y, (pad_size, pad_size), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        y = smoothed
    
    return y

def _gaussian_filter(y, window, cycles, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    
    if window < 3 or len(y) < window:
        return y.copy()
    
    for _ in range(cycles):
        y = gaussian_filter1d(y, sigma=sigma, mode='mirror')
    
    return y

def _savgol(y, window, cycles, order=3):
    if window <= order or len(y) < window:
        return y.copy()
    
    for _ in range(cycles):
        y = savgol_filter(y, window_length=window, polyorder=order, mode='mirror')
    
    return y
def peakpicker(mz: np.ndarray,
               intens: np.ndarray,
               spectra_ind: int,
               fwhm_filter: float | tuple[float, float] | list[float, float]
                          | np.ndarray[float, float] | None = None,
               fwhm_merge_factor: float | None = None,
               heightfilter: float | None = None,
               rel_heightfilter: float | None = None,
               peaklocation: float = 1,
               noise_est_iter: int = 3,
               SNR_threshold: float | None = 3.5,
               noise_mz_width: float = 9.0,
               return_areas: bool = True,
               headers: list[str] = ["spectra_ind", "mz", "Intensity", "Area",
                                     "SNR", "FWHM", "PextL", "PextR",
                                     "Noise", "Mean noise"]
               ) -> np.ndarray:
    """Detect and characterise peaks in a profile mass spectrum.

    Locates every peak in the spectrum and produces a feature-rich peak
    list. For each detected peak the following properties are computed:

    * **m/z and intensity** of the peak apex.
    * **Peak area** — trapezoidal integration between the left and right
      peak-base boundaries.
    * **FWHM** — the interpolated m/z width of the peak at half its
      maximum intensity.
    * **Peak base points** (``PextL`` / ``PextR``) — the m/z of the
      valleys (local minima) delimiting the peak on both sides.
    * **Signal-to-noise ratio** (``SNR``), the estimated **noise level**
      and the **mean noise** intensity, when an ``SNR_threshold`` is
      given. Note: the current SNR filtering uses a z-score criterion.

    Each property can be refined or filtered through the corresponding
    parameter. If a filtering or calculation parameter is ``None``, the
    related step is skipped and the property is not added to the output,
    which saves time and memory.

    Notes
    -----
    The heavy computation is offloaded to a JIT-compiled Numba kernel
    (:func:`_peakpicker_core`). This wrapper performs input validation
    and result formatting only, and adds negligible overhead.

    Parameters
    ----------
    mz : np.ndarray
        The m/z scale.
    intens : np.ndarray
        The intensity values.
    spectra_ind : int
        Index of the current spectrum, written into the output peak list.
    fwhm_filter : float or tuple or None, optional
        Peak-width filter based on the full width at half maximum (FWHM).
        A scalar keeps peaks wider than it; a ``(min, max)`` tuple keeps
        peaks within that range. Default is ``None``.
    fwhm_merge_factor : float or None, optional
        Merges peaks closer than ``FWHM * fwhm_merge_factor``. Default is ``None``.

        Practical zones for spurious-peak removal:
        - ``< 0.85``: guaranteed artefact — always merge (Rayleigh limit).
        - ``0.85–1.70``: uncertainty zone — merge only after shape inspection.
        - ``≥ 1.70`` (up to ~2.5): independence zone — do not merge; peaks are resolved.
        
        Key reference values:
        - 1.00: IUPAC standard (50% valley).
        - 1.18: 10% descent criterion (near baseline).
        - 1.70: 4σ, 98.5% valley — safe lower bound for de-duplication.
        - 2.55: 6σ, 99.9% valley — ideal upper bound (no mutual distortion).
        - 3.00: noise-limited limit (99.99% valley), useful only on high-dynamic instruments.

    heightfilter : float or None, optional
        Removes peaks whose absolute apex intensity is below this value.
        Default is ``None``.
    rel_heightfilter : float or None, optional
        Removes peaks whose apex intensity relative to the spectrum
        maximum is below this value. Default is ``None``.
    peaklocation : float, optional
        Fraction of the peak height used to compute the barycentric peak
        centre and the thresholds for the oversegmentation filter.
        Default is ``1``.
    noise_est_iter : int, optional
        Number of iterations used to estimate the noise. Higher values
        give a more stable estimate at the cost of runtime. Default is
        ``3``.
    SNR_threshold : float or None, optional
        Removes peaks whose signal-to-noise ratio is below this
        threshold. Default is ``3.5``.
    noise_mz_width : float, optional
        Width of the m/z window used to estimate the noise. Default is ``9.0``.
    return_areas : bool, optional
        Whether to compute and return the peak area. Default is ``True``.
    headers : list of str, optional
        Column headers used to order the returned peak properties.

    Returns
    -------
    np.ndarray
        The peak list, one row per detected peak, with the requested
        peak properties as columns (ordered according to *headers*).
    """

    props = {}
    pmz, pint, pareas, pSNR, pext_left, pext_right, pfwhm, pnoise, pmean_noise = _peakpicker_core(mz,
                                                                                                  intens,
                                                                                                  heightfilter,
                                                                                                  rel_heightfilter,
                                                                                                  fwhm_filter,
                                                                                                  SNR_threshold,
                                                                                                  noise_mz_width,
                                                                                                  fwhm_merge_factor,
                                                                                                  peaklocation,
                                                                                                  return_areas,
                                                                                                  noise_est_iter)
    n_peaks = pmz.size
    if n_peaks == 0:
        return None
    props["spectra_ind"] = np.ones(n_peaks, dtype=int) * spectra_ind
    props["mz"] = pmz
    props["Intensity"] = pint
    if return_areas:
        props["Area"] = pareas
    if SNR_threshold is not None:
        props["SNR"] = pSNR
        props["Noise"] = pnoise
        props["Mean noise"] = pmean_noise
    props["PextL"] = pext_left
    props["PextR"] = pext_right
    props["FWHM"] = pfwhm
    return np.column_stack([props[header] for header in headers])

def peakpicker_legacy(mz,
              intens,
              spectra_ind,
              fwhhfilter = None,
              oversegmentationfilter = None,
              heightfilter = None,
              rel_heightfilter = None,
              peaklocation = 1,
              noise_func = np.std,
              noise_est_iterations = 3, 
              SNR_threshold = 3, 
              Calc_peak_area = True, 
              headers = ["spectra_ind", "mz", "Intensity", "Area", "SNR", "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"]
              ) -> np.ndarray:
    """
    Detect peaks in a mass spectrum and compute their characteristics.

    The function locates every peak in the spectrum and characterises it,
    producing a feature-rich peak list. Beyond the peak position and apex
    intensity, it computes and stores the following properties for each
    detected peak:

    * **m/z and intensity** of the peak apex.
    * **Peak area** — computed by trapezoidal integration between the left
      and right peak-base boundaries.
    * **FWHM points** (``FWHML`` / ``FWHMR``) — the left and right m/z
      positions at which the peak intensity drops to half its maximum.
    * **Peak base points** (``PextL`` / ``PextR``) — the m/z of the valleys
      (local minima) delimiting the peak on both sides.
    * **Signal-to-noise ratio** (``SNR``) and the estimated **noise level**
      together with the **mean noise** intensity, when an ``SNR_threshold``
      is supplied. Note: The current SNR filtering implementation only 
      supports the fast mode. It estimates noise across the entire mass 
      spectrum, excluding only those regions already identified as peaks.

    All of these characteristics can be refined or filtered with the
    corresponding parameters. If a filtering or calculation parameter is
    ``None``, the corresponding step is skipped and the related peak
    property is not added to the output, which saves time and memory.

    Parameters
    ----------
    mz : np.ndarray
        The m/z scale.
    intens : np.ndarray
        The intensity values.
    spectra_ind : int
        Index of the current spectrum, written into the output peak list.
    fwhhfilter : float or tuple or None, optional
        Peak-width filter based on the full width at half height (FWHH).
        A scalar keeps peaks wider than it; a ``(min, max)`` tuple keeps
        peaks within that range. Default ``None``.
    oversegmentationfilter : float or str or None, optional
        Filter for peaks that are too close to each other (merges
        oversegmented peaks). If a string is given, the median FWHH is used
        as the threshold. Default ``None``.
    heightfilter : float or None, optional
        Removes peaks whose absolute apex intensity is below this value.
        Default ``None``.
    rel_heightfilter : float or None, optional
        Removes peaks whose apex intensity relative to the spectrum maximum
        is below this value. Default ``None``.
    peaklocation : float, optional
        Fraction of the peak height used to compute a barycentric peak
        centre and the thresholds for the oversegmentation filter.
        Default ``1``.
    noise_func : callable, optional
        Function used to estimate the noise level (e.g. ``np.std``). The
        whole spectrum is processed over several iterations; after each
        iteration the noise vs signal points are re-classified.
        Default ``np.std``.
    noise_est_iterations : int, optional
        Number of iterations used to estimate the noise. More than three
        iterations is recommended. Default ``3``.
    SNR_threshold : float or None, optional
        Removes peaks whose signal-to-noise ratio is below this threshold.
        Default ``3``.
    Calc_peak_area : bool, optional
        Whether to compute the peak area. Default ``True``.
    headers : list of str, optional
        Column headers used to order the returned peak properties.

    Returns
    -------
    np.ndarray
        The peak list, one row per detected peak, with the requested peak
        properties as columns (ordered according to *headers*).
    """
    # TODO: implement a hybrid fast/slow SNR filter. In the last one or two cycles,
    # estimate noise within a window around each peak, where the window is the std
    # of purely-noise points left and right of the peak (peak points excluded) and
    # its size is measured in points (not m/z). Consider using numba.
    xsize = mz.size
    props={}
    # Robust valley finding
    valley_dots = np.where(np.diff(intens) !=0)[0] 
    valley_dots = np.concatenate((valley_dots, [xsize-1]))    
    loc_min = np.diff(intens[valley_dots])
    loc_min = (np.array([True,*(loc_min < 0)])) & np.array(([*(loc_min > 0),True]))
    left_min = np.concatenate([[-1],valley_dots[:-1]])[loc_min][:-1] + 1
    right_min = valley_dots[loc_min][1:]

    # Compute max for every peak
    size = left_min.shape
    val_max = np.empty(size)
    pos_peak = np.empty(size,dtype=int)
    for idx, [lm, rm] in enumerate(zip(left_min, right_min)):
        val_max[idx] = np.max(intens[lm:rm])
        pos_peak[idx] = lm + np.argmax(intens[lm:rm])
    # Remove peaks below the height, relative height
    if heightfilter and rel_heightfilter:
        k = (val_max >= heightfilter) & (val_max/max(intens) >= rel_heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]
        pos_peak = pos_peak[k]
    elif heightfilter:
        k = (val_max >= heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]
        pos_peak = pos_peak[k]
    elif rel_heightfilter:
        k = (val_max/max(intens) >= rel_heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]
        pos_peak = pos_peak[k]

    # Remove peaks below the SNR thresholds
    if SNR_threshold:
        noise_bool = np.ones(xsize, dtype = bool) # Zero iteration
       
        for it in range(noise_est_iterations):
            noise_func_full = noise_func(intens[noise_bool])
            noise_mean_full = np.mean(intens[noise_bool])
            for idx in np.where(((val_max-noise_mean_full)/noise_func_full>=SNR_threshold))[0]: # Essentially a plain z-score.
                sl = slice(left_min[idx],right_min[idx]+1)
                noise_bool[sl] = False
        
        props["Noise"] = noise_func(intens[noise_bool])
        props["Mean noise"]= np.mean(intens[noise_bool])
        k = (val_max-props["Mean noise"])/props["Noise"]>=SNR_threshold
        
        val_max=val_max[k]
        left_min=left_min[k]
        right_min=right_min[k]
        pos_peak = pos_peak[k]
    # Compute FWHH for every peak
    size = left_min.shape
    props["FWHML"] = np.empty(size)
    props["FWHMR"]  = np.empty(size)
    for idx, [lm, rm, vm, pp] in enumerate(zip(left_min, right_min, val_max, pos_peak)):
        props["FWHML"][idx] = np.interp(vm/2,intens[lm:pp+1], mz[lm:pp+1])
        props["FWHMR"][idx] = np.interp(vm/2,intens[pp:rm+1][::-1], mz[pp:rm+1][::-1])
    # Remove peaks with FWHH thresholds
    if fwhhfilter:
        if isinstance(fwhhfilter,tuple):
            k = ((props["FWHMR"] - props["FWHML"]) >= fwhhfilter[0]) & ((props["FWHMR"] - props["FWHML"]) <= fwhhfilter[1])
        else:
            k = (props["FWHMR"] - props["FWHML"]) >= fwhhfilter
        val_max = val_max[k]
        props["FWHML"] = props["FWHML"][k]
        props["FWHMR"] = props["FWHMR"][k]
        left_min = left_min[k]
        right_min = right_min[k]
        pos_peak = pos_peak[k]
    # Remove oversegmented peaks
    if oversegmentationfilter:
        if isinstance(oversegmentationfilter,str):
            oversegmentationfilter = np.median(props["FWHMR"]-props["FWHML"])
        while True:
            peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
            pkmz = np.empty(left_min.shape)
            
            for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
                mask = intens[lm:rm] >= th
                if not mask.any():
                    pkmz[idx]=np.nan
                else:
                    pkmz[idx] = np.sum(intens[lm:rm][mask] * mz[lm:rm][mask]) / np.sum(intens[lm:rm][mask])
            dpkmz = np.concatenate(([np.inf], np.diff(pkmz), [np.inf]))
            
            j = np.where((dpkmz[1:-1] <= oversegmentationfilter) & (dpkmz[1:-1] <= dpkmz[:-2]) & (dpkmz[1:-1] < dpkmz[2:]))[0]
            if j.size == 0:
                break
            left_min = np.delete(left_min, j + 1)
            right_min = np.delete(right_min, j)
            props["FWHML"] = np.delete(props["FWHML"], j + 1)
            props["FWHMR"] = np.delete(props["FWHMR"], j)
            ## New TODO: Test
            stack_j = np.vstack((j,j+1))
            pos_peak_oversegmentation = pos_peak[stack_j]
            val_max_oversegmentation = val_max[stack_j]
            max_idx = np.argmax(val_max_oversegmentation, axis = 0)
            range_j = np.arange(len(j))
            val_max[j] = val_max_oversegmentation[max_idx, range_j]
            pos_peak[j] = pos_peak_oversegmentation[max_idx, range_j]
            val_max = np.delete(val_max, j + 1)
            pos_peak = np.delete(pos_peak, j + 1)

            ## New
            ### OLD
            # val_max[j] = np.maximum(val_max[j], val_max[j + 1])
            # val_max = np.delete(val_max, j + 1)
            ### OLD
    else:
        peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
        pkmz = np.empty(left_min.shape)
        
        for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
            mask = intens[lm:rm] >= th
            if not mask.any():
                pkmz[idx] = np.nan
            else:
                pkmz[idx] = np.sum(intens[lm:rm][mask] * mz[lm:rm][mask]) / np.sum(intens[lm:rm][mask])

    signal_num = len(val_max)
    ## Area calculation
    if Calc_peak_area:
        props["Area"] = np.empty((signal_num,))
        for idx in range(signal_num):
            sl = slice(left_min[idx],right_min[idx]+1,1)
            if min(intens[sl])<0:
                props["Area"][idx] = np.trapezoid(intens[sl] - min(intens[sl]),mz[sl])
            else:
                props["Area"][idx] = np.trapezoid(intens[sl],mz[sl])
    if SNR_threshold:
        props["SNR"] = (val_max - props["Mean noise"])/props["Noise"]
        props["Noise"] = [props["Noise"]]*signal_num
        props["Mean noise"]= [props["Mean noise"]]*signal_num

    left_bool = np.array(pos_peak - left_min) > 2
    right_bool = np.array(right_min - pos_peak) > 2
    if left_bool.any():
        left_min[left_bool] += 1
    if right_bool.any():
        right_min[right_bool] -= 1
    props["PextL"] = mz[left_min] 
    props["PextR"] = mz[right_min]
    return np.column_stack(([spectra_ind]*signal_num,pkmz, val_max, *(props[key] for key in headers[3:])))


def modify_raw_spectrum(mz,
                        ints,
                        mz_discret_coeffs,
                        mz_crop_range = None,
                        zero_points_to_peaks_ext = False,
                        mz_segments_to_zero = None):
    """Modify raw spectrum data before further processing.

    Optionally:
    - Zero out the signal in specified m/z segments.
    - Crop the spectrum to a given m/z range.
    - Add zero-value points around peak edges when peaks are cropped.

    Parameters
    ----------
    mz : np.ndarray
        The m/z scale.
    ints : np.ndarray
        The intensity values.
    mz_discret_coeffs : array_like
        Polynomial coefficients describing the m/z discretisation step.
    zero_points_to_peaks_ext : bool, optional
        If ``True``, add zero points to peak extensions. Default ``False``.
    mz_segments_to_zero : list of tuple, optional
        A list of ``(mz_min, m/z_max)`` segments whose signal is set to zero.
        Default ``None``.
    mz_crop_range : tuple of float or None, optional
        If provided as ``(min_mz, max_mz)``, the spectrum is cropped to this
        range. If min_mz or max_mz is ``None``, the spectrum is not cropped from this side. 
        Points outside the range are removed. Default ``None`` (no cropping at all).

    Returns
    -------
    tuple of np.ndarray
        The modified ``(mz, ints)`` pair.
    """

    if mz_crop_range is not None:
        mz, ints = crop_spectrum(mz, ints, *mz_crop_range)
    if mz_segments_to_zero and mz_discret_coeffs is not None:
        mz, ints = reduce_signal_to_zero(mz, ints, mz_segments_to_zero)
    if zero_points_to_peaks_ext:
        mz, ints = add_zero_points_to_peaks(mz, ints, mz_discret_coeffs)
    return mz, ints

def crop_spectrum(mz, ints, low_mz=None, high_mz=None):
    left = 0
    right = mz.size
    if low_mz is not None:
        left = np.searchsorted(mz, low_mz, side='left')
    if high_mz is not None:
        right = np.searchsorted(mz, high_mz, side='right')
    return mz[left:right], ints[left:right]

def add_zero_points_to_peaks(mz: np.ndarray,
                             ints: np.ndarray, 
                             mz_discret_coeffs: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Add zero points to peaks
    """
    mz_discretion_model = np.poly1d(mz_discret_coeffs)

    diff_mz = np.diff(mz)
    mz_discr = mz_discretion_model(mz[:-1])
    big_gap_bool = diff_mz > 2.5*mz_discr
    small_gap_bool = (diff_mz > 1.25*mz_discr) ^ big_gap_bool
    new_val = {}
    if np.any(big_gap_bool):
        new_val['left'] = mz[np.append(big_gap_bool, [False])] + mz_discr[big_gap_bool]
        new_val['right'] = mz[np.append([False], big_gap_bool)] - mz_discr[big_gap_bool]

    if np.any(small_gap_bool):
        new_val['small'] = mz[np.append(small_gap_bool, [False])] + mz_discr[small_gap_bool]

    if new_val:
        new_val['borders'] = [mz[0]-mz_discr[0], mz[-1]+mz_discr[-1]]
        new_val = np.concatenate(list(new_val.values()), axis=None)
        idx = np.searchsorted(mz, new_val)
        mz = np.insert(mz, idx, new_val)
        ints = np.insert(ints, idx, 0)
        
    # sorting by mz
    idx_sort = np.argsort(mz)
    new_loc_mz = mz[idx_sort]
    new_loc_ints = ints[idx_sort]

    return new_loc_mz, new_loc_ints

def reduce_signal_to_zero(mz: np.ndarray,
                          ints: np.ndarray,
                          mz_segments_to_zero: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce signal to zero
    """
    artefacts_bool = np.zeros(len(mz), dtype=bool)
    for distortion in mz_segments_to_zero:
        artefacts_bool = artefacts_bool | ((mz > distortion[0]) & (mz < distortion[1]))
        ints[artefacts_bool] = 0
    return mz, ints

def _compute_KDE(peaklist: pd.DataFrame,
                discret_coeffs: np.ndarray,
                cpu_num: int = 1,
                KD_bandwidth: str | float = 'fwhm',
                bwc: float = 1.0,
                KD_kernel: str = 'gaussian',
                KDE_algo: str| None = None,
                split_mz_min: float = 10.0,
                split_peaks_min: int = 25,
                account_mzscale: bool = True):
    """Compute a kernel density estimate over the peaks of a peak list.

    The peak list is first split into segments separated by large m/z gaps.
    For every segment, the probability density of the peaks is estimated
    with a KDE (FFT or tree based) using a bandwidth selected per peak
    (e.g. derived from its FWHM). When ``account_mzscale`` is enabled, the
    bandwidth is clamped so that it never falls below the local m/z
    discretisation step (i.e. the smallest meaningful separation between two
    neighbouring m/z points is respected). The per-segment densities are
    then merged into a single m/z vs density grid.

    Parameters
    ----------
    peaklist : pd.DataFrame
        Peak list that must contain ``mz`` and ``FWHM`` columns.
    discret_coeffs : np.ndarray
        Polynomial coefficients describing the m/z discretisation step.
    cpu_num : int, optional
        Number of processes used for the per-segment density estimation.
        Default ``1``.
    KD_bandwidth : str or float, optional
        Bandwidth selection method (``'fwhm'``, ``'mz_discret'``) or a fixed
        float value. Default ``'fwhm'``.
    bwc : float, optional
        Multiplier applied to the selected/computed bandwidth.
        Default ``1.0``.
    KD_kernel : str, optional
        KDE kernel name. Default ``'gaussian'``.
    KDE_algo : str or None, optional
        KDE algorithm: ``'fft'`` or ``'tree'``. If ``None``, ``'tree'`` is
        used. Default ``None``.
    split_mz_min : float, optional
        Minimum m/z gap used to split peaks into separate segments.
        Default ``10.0``.
    split_peaks_min : int, optional
        Minimum number of peaks required per segment. Default ``25``.
    account_mzscale : bool, optional
        Whether to account for the m/z discretisation when computing the
        bandwidth. Default ``True``.

    Returns
    -------
    tuple of np.ndarray
        ``(X_plot, Y_plot)`` — the common m/z grid and the corresponding
        summed peak-density values.
    """
    FWHM2sigma = FWHM_TO_SIGMA_FACTOR/5
    mz_model = np.poly1d(discret_coeffs)
    if peaklist.empty:
        warnings.warn("Peak list is empty, returning empty arrays for peaks PDF")
        return np.array([]), np.array([])
    if KD_bandwidth == "fwhm":
        peaklist.loc[:,'KD_bandwidth'] = (peaklist['FWHM'])*FWHM2sigma*bwc
        if account_mzscale:
            mz_discret = mz_model(peaklist['mz'])
            
            low_band_bool = peaklist['KD_bandwidth'] < mz_discret
            peaklist.loc[low_band_bool,'KD_bandwidth'] = (mz_discret[low_band_bool]*bwc).astype(peaklist['KD_bandwidth'].dtype)
    elif KD_bandwidth == "mz_discret":
        peaklist.loc[:,'KD_bandwidth'] = mz_model(peaklist['mz'])*bwc
    else:
        peaklist.loc[:,'KD_bandwidth'] = KD_bandwidth*bwc
    KD_data = split_pdtable_by_peaks_gap(peaklist, split_mz_min = split_mz_min, split_peaks_min = split_peaks_min)
    n_segments = len(KD_data) 
    assert peaklist['mz'].count() == sum( len(t[0]) for t in KD_data)
    X_plot = [None] * n_segments
    Y_plot = [None] * n_segments
    
    KDE_algo = KDE_algo or 'tree'
    if KDE_algo.lower() == 'fft':# or (len(mz_discret_coeffs) == 1):
        KDE_func = FFTKDE
    elif KDE_algo.lower() == 'tree':
        KDE_func = TreeKDE
    else:
        raise ValueError('Unknown KDE function')
    partial_worker = partial(segment_probability_distribution, KDE_func, KD_kernel, discret_coeffs)
    with Pool(cpu_num) as p:
        Last_segment_max_mz = 0
        for n, (X_plot_segment, Y_plot_segment) in enumerate(tqdm(p.imap_unordered(partial_worker,KD_data), total = len(KD_data), unit = 'segment', desc = 'Peak PDF calculation')):
            assert Last_segment_max_mz < X_plot_segment.min() #TODO удалить при выкладывании
            X_plot[n] = X_plot_segment
            Y_plot[n] = Y_plot_segment
            # Y_plot[plot_slice] += result
    X_plot = np.hstack(X_plot)
    Y_plot = np.hstack(Y_plot)
    idx_sort = np.argsort(X_plot)
    X_plot = X_plot[idx_sort] 
    Y_plot = Y_plot[idx_sort]
    return X_plot, Y_plot

def segment_probability_distribution(KDE_func, KD_kernel, mz_discret_coeffs, KD_data):
    """Estimate the probability distribution of the peaks of one segment.

    Builds a uniform m/z grid around the segment (padded by six bandwidths
    on each side), then evaluates the KDE of the segment's peaks on it.

    Parameters
    ----------
    KDE_func : type
        KDE class to use (``FFTKDE`` or ``TreeKDE`` from KDEpy).
    KD_kernel : str
        KDE kernel name.
    mz_discret_coeffs : array_like
        Polynomial coefficients describing the m/z discretisation step.
    KD_data : tuple of np.ndarray
        ``(mz, KD_bandwidth)`` — the m/z values and per-peak bandwidths of
        the segment.

    Returns
    -------
    tuple of np.ndarray
        ``(X_plot_segment, Y_plot_segment)`` — the m/z grid and the
        corresponding estimated density values.
    """
    mz, KD_bandwidth = KD_data
    segment_min = mz[0] - KD_bandwidth[0]*6
    segment_max = mz[-1] + KD_bandwidth[-1]*6
    min_dist = np.poly1d(mz_discret_coeffs)(mz).min()
    if KDE_func.__name__ == 'FFTKDE':
        KD_bandwidth = np.median(KD_bandwidth)
    X_plot_segment = _set_KDE_X_plot(segment_min, segment_max, min_dist = min_dist)
    Y_plot_segment = KDE_func(kernel = KD_kernel, bw = KD_bandwidth).fit(mz)(X_plot_segment)*len(mz)
    return X_plot_segment, Y_plot_segment