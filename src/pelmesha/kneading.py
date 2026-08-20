import numpy as np
import math
import os
import warnings
from KDEpy import FFTKDE, TreeKDE
from pybaselines import Baseline
from pelmesha.utensils import split_pdtable_by_peaks_gap, _set_KDE_X_plot
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

FWHM_TO_SIGMA_FACTOR = 1 / np.sqrt(8 * np.log(2))  # Фактор пересчета FWHM в sigma
###########################################
#   Base pipeline functions               #
###########################################
def preprocess_configuration_base(
    datasource: "DataSource",
    roi,
    rmeta,
    configs: "Configs | PipelineConfigurator"):
    
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

    internal_configs['peakpick'] = {}
    # Configure headers conditionally based on processing flags 
    if configs['peakpicker']["SNR_threshold"] and configs['peakpicker']["Calc_peak_area"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "Area", "SNR",  
                        "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"]
        
    elif configs['peakpicker']["SNR_threshold"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "SNR",  
                        "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"]
    elif configs['peakpicker']["Calc_peak_area"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "Area",  
                        "PextL", "PextR", "FWHML", "FWHMR"]
    else: 
        headers_list = ["spectra_ind", "mz", "Intensity",  
                        "PextL", "PextR", "FWHML", "FWHMR"]
    configs.update({"headers": headers_list})
    return resampled_mz, headers_list, internal_configs


def process_spectra_base(
    mz: np.ndarray,
    intensity: np.ndarray,
    configs: "Configs | PipelineConfigurator",
    **internal_configs) -> np.ndarray:
    """Полный цикл предобработки масс-спектра:
    сглаживание → коррекция базовой линии → ресемпл →  выравнивание.
    mz - всегда вектор
    intensity - Если данные континуальны - матрица, если нет - вектор
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

    mzsize = mz.size
    configs = configs['peakpicker'] #Получаем непосредственно словарь конфигов для функции peakpicker
    return peakpicker(mz, intensity, mzsize, idx, **configs)


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
    if resample_mz_step is not None and resample_num_points is not None:
        raise ValueError("Укажите что-то одно: либо resample_mz_step, либо resample_num_points.")
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

def peakpicker(mz,
              intens,
              xsize, 
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
    Общее описание
    ----
    Функция для получения характеристик пиков спектра. Если в каком-то параметре стоит `None`, то функция не будет производить фильтрацию или расчёты и не будет добавлять свойство пиков на выходе, что может экономить время и память.
    :param mz: mz
    :param intens: Intensity
    :param valley_dots: numpy array того какие точки спектра явлюятся наклонными. На входе подаётся как `np.where(np.diff(intens) != 0)[0]` или как строка этого результата, если intens матрица. Этот параметр нужен скорее для универсальности функции и оптимизации кол-ва расчётов/обращений. 
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `None`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `None`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `None`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `None`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `None`
    :param noise_func: функция оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `np.std`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    
    :type mz: `np.array`
    :type intens: `np.array`
    :type valley_dots: `np.array`
    :type oversegmentationfilter: `float`
    :type fwhhfilter: `float`
    :type heightfilter: `float`
    :type peaklocation: `float` and =<1
    :type rel_heightfilter: `float`
    :type SNR_threshold: `float`
    :type noise_func: function
    :type noise_est_iterations: `int`

    :return: peaklist with peak properties
    :rtype: `np.array`
    """
    #TODO: сделать гибридфильтра по SNR быстрого и медленного варианта: последний или послдние два цикла - шум определяется по окну вокруг пика, 
    # возможно окно - это std точек справа и слева, которые чисто шумовые (точки пика исключены), а кол-во точек и есть размер окна (не по m/z). Постараться сделать с numba
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
        noise_points = np.array([True]*xsize) # Zero iteration

        for it in range(noise_est_iterations):
            for idx in np.where(((val_max-np.mean(intens[noise_points]))/noise_func(intens[noise_points])>=SNR_threshold))[0]: #По сути тут расчёт z-score в чистом виде TODO: оценить скорость рассчётов моего варианта и scipy.stats.zscore
                sl = slice(left_min[idx],right_min[idx]+1)
                noise_points[sl] = False
        
        props["Noise"] = noise_func(intens[noise_points])
        props["Mean noise"]= np.mean(intens[noise_points])
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
                        zero_points_to_peaks_ext = False,
                        mz_segments_to_zero = None):
    """
    Modify raw spectrum data. Reduce to zero segemnts and add zero points to peaks if their peaks are cropped (especially if peaks consists from 1-2 points)
    """
    if mz_segments_to_zero and mz_discret_coeffs is not None:
        mz, ints = reduce_signal_to_zero(mz, ints, mz_segments_to_zero)
    if zero_points_to_peaks_ext:
        mz, ints = add_zero_points_to_peaks(mz, ints, mz_discret_coeffs)
    return mz, ints

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
    
    FWHM2sigma = FWHM_TO_SIGMA_FACTOR/5
    mz_model = np.poly1d(discret_coeffs)
    
    if KD_bandwidth == "fwhm":
        peaklist.loc[:,'KD_bandwidth'] = (peaklist['FWHMR'] - peaklist['FWHML'])*FWHM2sigma*bwc
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

def segment_probability_distribution(KDE_func, KD_kernel, mz_discret_coeffs, KD_data): # Пока медленно
    mz, KD_bandwidth = KD_data
    segment_min = mz[0] - KD_bandwidth[0]*6
    segment_max = mz[-1] + KD_bandwidth[-1]*6
    min_dist = np.poly1d(mz_discret_coeffs)(mz).min()
    if KDE_func.__name__ == 'FFTKDE':
        KD_bandwidth = np.median(KD_bandwidth)
    X_plot_segment = _set_KDE_X_plot(segment_min, segment_max, min_dist = min_dist)
    Y_plot_segment = KDE_func(kernel = KD_kernel, bw = KD_bandwidth).fit(mz)(X_plot_segment)*len(mz)
    return X_plot_segment, Y_plot_segment

def _peaklist_stats(peaklist: pd.DataFrame,
                         sample: str,
                         roi: str,
                         draw_results: bool = True):
    temp_pivo = pd.pivot_table(peaklist,index=["spectra_ind","Peak"],values="mz",aggfunc=["count"])
    ms_num = peaklist.loc[:,'spectra_ind'].nunique()

    temp_pivo = temp_pivo.iloc[(temp_pivo["count"]>1)["mz"].values]

    if not temp_pivo.empty:
        duplicated_num=len(temp_pivo.index.value(level="Peak"))
        num_of_uniq_spectras = len(temp_pivo.droplevel('Peak').index.unique()) #Определяем кол-во спектров, где обнаружены дубликаты чисто для справки
        textw=f"At the specified peak grouping settings, {temp_pivo['count']['mz'].sum()-temp_pivo.shape[0]} duplicates were identified, of which {duplicated_num} were unique peaks in {num_of_uniq_spectras} of mass spectra ({num_of_uniq_spectras*100/(ms_num):.2f}% of the total spectra)."
        warnings.warn(textw)
        if temp_pivo['count']["mz"].value_counts().index.max() > 2 and draw_results:
            plt.figure(figsize=(3, 2))
            plt.bar(temp_pivo['count']["mz"].value_counts().index.astype(str),temp_pivo['count']["mz"].value_counts().astype(int))
            plt.xlabel('Quantity')
            plt.ylabel('Num of duplicates')
            plt.gca().set_title(f"Peaks occurence in one group in mass spectrum. Sample: {sample} {roi}")        
            plt.grid(visible=True,which="both",axis="y")