import pandas as pd
import numpy as np
import collections
from pelmesha.loaders import hdf5_Load, specdata_Load, hdf5_close, create_file_path, del_datasets_hdf5, del_hdf5, repack_hdf5, hdf5_metadata
from itertools import product, zip_longest
from threading import Thread
from pybaselines import Baseline
from scipy.stats import median_abs_deviation
from pelmesha.loaders import find_paths, logger
from pyimzml.ImzMLParser import ImzMLParser
import h5py
from h5py import File
import gc
import math
import os
import re
import glob
import copy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from math import sqrt
import warnings
import yaml
from typing import Callable, Union, Tuple
from scipy.signal import savgol_filter
from pyteomics import mzxml
# from pelmesha.cli import calculate
from functools import wraps
try:
    from torch.multiprocessing import Pool, cpu_count, Manager, Value, set_start_method

except Exception as error:
    warnings.warn(f"During import torch.multiprocessing package raised error: {error}. Using python package multiprocessing instead", stacklevel =3)
    from multiprocessing import Pool, cpu_count, Manager, Value, set_start_method
try:
    set_start_method('spawn')
except:
    pass

## pairwise for python versions below 10 
from sys import version_info
if version_info[0] < 3:
    raise Exception("Must be using Python 3")
else:
    if version_info[1]<10:
        from itertools import tee

        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)
    else:
        from itertools import pairwise


### Code from __init__.py msalign (https://github.com/lukasz-migas/msalign)
"""Signal calibration and alignment by reference peaks - copy of MSALIGN function from MATLAB bioinformatics library."""
from .align import Aligner
from typing import List
__all__ = ["msalign", "Aligner"]


def msalign(
    x: np.ndarray,
    array: np.ndarray,
    align_peaks: List,
    method: str = "cubic",
    width: float = 10,
    ratio: float = 2.5,
    resolution: int = 100,
    iterations: int = 5,
    grid_steps: int = 20,
    shift_range: List = None,
    align_pweights: List = None,
    return_shifts: bool = False,
    align_by_index: bool = False,
    only_shift: bool = False,
):
    aligner = Aligner(
        x,
        array,
        align_peaks,
        method=method,
        width=width,
        ratio=ratio,
        resolution=resolution,
        iterations=iterations,
        grid_steps=grid_steps,
        shift_range=shift_range,
        weights=align_pweights,
        return_shifts=return_shifts,
        align_by_index=align_by_index,
        only_shift=only_shift,
    )
    aligner.run()
    return aligner.apply()


msalign.__doc__ = Aligner.__doc__
### END of code from __init__.py msalign (https://github.com/lukasz-migas/msalign)

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

class Configs(dict): 
    initialized = False

    def __init__(self, functions_list, config_path = None,**kwargs):
        if not isinstance(functions_list,list):
            functions_list = [functions_list]
        self._init_functions_list = functions_list
        self._init_config_path = config_path
        
        ## Argument validation for functions
        # for key in kwargs:
        #     if isinstance(kwargs[key], AdaptiveParameter):
        #         kwargs[key] = kwargs[key].parameter
        if "peaklocation" in kwargs:
            if not isinstance(kwargs['peaklocation'], (int, float)) or not np.isscalar(kwargs['peaklocation']) or kwargs['peaklocation'] < 0 or kwargs['peaklocation'] > 1:
                raise ValueError("peaks_prop: Invalid peak location")
        if "fwhhfilter" in kwargs:
            if not isinstance(kwargs['fwhhfilter'], (int, float)) or not np.isscalar(kwargs['fwhhfilter']) or kwargs['fwhhfilter'] < 0:
                if not isinstance(kwargs['fwhhfilter'],type(None)):
                    raise ValueError("peaks_prop: Invalid FWHH filter")
        if "oversegmentationfilter" in kwargs:
            if not isinstance(kwargs['oversegmentationfilter'], (int, float)) or not np.isscalar(kwargs['oversegmentationfilter']):
                if isinstance(kwargs['oversegmentationfilter'], str):
                    kwargs['oversegmentationfilter'] = kwargs['oversegmentationfilter'].lower()
                elif isinstance(kwargs['oversegmentationfilter'], type(None)):
                    pass
                else:
                    raise ValueError("peaks_prop: Invalid oversegmentation filter")
            elif kwargs['oversegmentationfilter'] < 0:
                raise ValueError("peaks_prop: Invalid oversegmentation filter")
        if "heightfilter" in kwargs:
            if not isinstance(kwargs["heightfilter"], (int, float)) or not np.isscalar(kwargs["heightfilter"]) or kwargs["heightfilter"] < 0:
                if not isinstance(kwargs["heightfilter"], type(None)):
                    raise ValueError("peaks_prop: Invalid height filter")
        if "rel_heightfilter" in kwargs:
            if not isinstance(kwargs["rel_heightfilter"], (int, float)) or not np.isscalar(kwargs["rel_heightfilter"]) or kwargs["rel_heightfilter"] < 0 or kwargs["rel_heightfilter"] > 100:
                if not isinstance(kwargs["rel_heightfilter"], type(None)):
                    raise ValueError("peaks_prop: Invalid relative height filter")
        
        ## Getting functions arguments
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Base_configs.yaml")
        if not config_path.endswith('.yaml'):
            config_path+='.yaml'
        with open(config_path, 'rb') as file:
            configs = {}
            configs.update(yaml.load(file,Loader=yaml.FullLoader))
        configs = self._conf_param_recurs_replace(configs, kwargs)

        baseline_algo = configs.get("baseline_algo", None)
        if baseline_algo:
            if getattr(Baseline(), baseline_algo, None) is None:
                available = [m for m in dir(Baseline()) if not m.startswith('_')]
                raise ValueError(
                    f"Method '{baseline_algo}' not found. Available methods: {', '.join(available)}"
                ) from None
        
        configs['baseliner'] = AdaptiveParameter(baseline_algo, _baseliner_prep)
        ### Adjusting smoothing window and maximum shift parameters based on conditions for m/z-to-points conversion
        if "shift_range" in configs:
            if isinstance(configs['shift_range'],(int,float)):
                configs["shift_range"] = [-configs["shift_range"],configs["shift_range"]]
            elif isinstance(configs['shift_range'],AdaptiveParameter):
                configs["shift_range"] = configs["shift_range"].parameter
               
        if configs.get('align_peaks', None) is not None:

            if configs.get('only_shift',False):
                configs['align_by_index'] = True

            if configs.get('align_by_index',False):
                configs["shift_range"] = AdaptiveParameter(configs["shift_range"], 
                                                        adaptation_rule = _shift_range_to_dots)
            else: 
                configs["shift_range"] = AdaptiveParameter(configs["shift_range"], 
                                                        None)

        else:
            configs["shift_range"] = AdaptiveParameter(None, 
                                                        adaptation_rule = _shift_range_to_dots)
        
        if configs.get('smooth_algo',False):
            if configs["smooth_window"] is None:
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Base_configs.yaml"), 'rb') as file:
                    base_conf = {}
                    base_conf.update(yaml.load(file,Loader=yaml.FullLoader))
                
                configs["smooth_window"] = base_conf['smooth_window']
                warnings.warn(f'"smooth_window" parameter is None, while smooth_algo is specified. "smooth_window" changed to base value from "Base_configs.yaml" file: { base_conf['smooth_window']}', stacklevel =3)
            if isinstance(configs["smooth_window"], AdaptiveParameter):
                configs["smooth_window"] = configs["smooth_window"].parameter
            configs["smooth_window"] = AdaptiveParameter(configs["smooth_window"], 
                                                            adaptation_rule = _smooth_window_to_dots)
        else:
            configs["smooth_window"] = AdaptiveParameter(None, 
                                                            _smooth_window_to_dots)


        ### Configuring spectral noise estimation parameters
        if 'noise_func' in configs:
            if configs["noise_func"] == "MAD":
                configs["noise_func"] = MAD
            elif configs["noise_func"] == "std":
                configs["noise_func"] = np.std
        ### Default Configuration Initialization  
        # Ensures presence of essential processing parameters by setting defaults if omitted.
        if 'align_peaks' not in configs:
            configs["align_peaks"] = None # Peak alignment flag (None = disabled)  
        if 'smooth_algo' not in configs:
            configs["smooth_algo"] = None # Smoothing algorithm selector (None = no smoothing)  

        ### Dynamic Peak List Header Configuration  
        # Generates dataset column headers based on SNR thresholding and peak area calculation flags.  
        # - Includes SNR, noise metrics, and peak area columns when corresponding parameters are enabled.  
        # - Default headers contain core peak characteristics: m/z, intensity, peak boundaries, and FWHM.  
        if "SNR_threshold" not in configs:
            configs["SNR_threshold"] = 3.5 # Default SNR cutoff for peak validation  
        if "Calc_peak_area" not in configs:
            configs["Calc_peak_area"] = True # Enable/disable peak area calculation  
        # Configure headers conditionally based on processing flags 
        if configs["SNR_threshold"] and configs["Calc_peak_area"]:
            configs["headers"] = DatasetHeaders([  
                                            "spectra_ind", "mz", "Intensity", "Area", "SNR",  
                                            "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"  
                                        ])
        elif configs["SNR_threshold"]:
            configs["headers"] = DatasetHeaders([  
                                            "spectra_ind", "mz", "Intensity", "SNR",  
                                            "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"  
                                        ])
        elif configs["Calc_peak_area"]:
            configs["headers"] = DatasetHeaders([  
                                            "spectra_ind", "mz", "Intensity", "Area",  
                                            "PextL", "PextR", "FWHML", "FWHMR"  
                                        ])
        else: 
            configs["headers"] = DatasetHeaders([  
                                            "spectra_ind", "mz", "Intensity",  
                                            "PextL", "PextR", "FWHML", "FWHMR"  
                                        ])

        ## Rearranging and hierarchical configuration setup for nested function scopes
        if functions_list[0] is not None:
            configs = self._rearrange_conf(configs,functions_list)
        self.initialized = True
        super().__init__(**configs)

    def _rearrange_conf(self,configs,functions_list):
        ## Rearranging arguments for baseliner into a dictionary
        self.param_nests_path = {}
        try:
            configs["baseline_configs"]={}
            func_code = configs['baseliner']([1]).__dict__["__wrapped__"].__code__
            for arg in tuple(configs.keys()):
                if arg in func_code.co_varnames[:func_code.co_argcount]:
                    configs["baseline_configs"][arg] = configs.pop(arg)
                    
                    
        except Exception as e:
            
            if isinstance(e,(TypeError,AttributeError)):
                configs['baseliner'] =  AdaptiveParameter(None, _baseliner_prep)
                configs['baseline_configs'] = {}
            else:
                print(e.__class__,e)
        ## Rearranging arguments by functions into a dictionary
        for func in functions_list:
            configs[func.__name__.split("_")[0]+"_configs"]={}

        for arg in tuple(configs.keys()):
            for func in functions_list:
                func_name_conf = func.__name__.split("_")[0]+"_configs"
                if arg in func.__code__.co_varnames[:func.__code__.co_argcount] and not arg.endswith("_configs"):
                    configs[func_name_conf][arg] = configs.pop(arg)
                    self.param_nests_path[arg] = [func_name_conf]

        ## Hierarchical configuration setup for nested function scopes
        for nested in tuple(configs.keys()):
            if nested.endswith("_configs"):
                for func in functions_list:
                    func_name_conf = func.__name__.split("_")[0]+"_configs"
                    if nested in func.__code__.co_varnames:
                        for arg in configs[nested].keys():
                            if self.param_nests_path.get(arg, False):
                                self.param_nests_path[arg] = [func_name_conf] + self.param_nests_path[arg]
                            else:
                                self.param_nests_path[arg] = [func_name_conf, nested]
                        configs[func_name_conf][nested] = configs.pop(nested)
        return configs

    def _conf_param_recurs_replace(self,configs, param_dict):
        for item in param_dict:
            try:
                if isinstance(param_dict[item], dict):
                    # Recursively replace parameters in nested dictionaries
                    self._conf_param_recurs_replace(configs[item], param_dict[item])
                else:
                    # Update configuration with non-dictionary parameters
                    configs[item] = param_dict[item] 
            except KeyError:
                print(f"Parameter '{item}' is not in configs")
        return configs
    
    def __getitem__(self, key):
        if dict.__contains__(self,key): 
            return dict.__getitem__(self, key)
        else:
            flatten_dict = self.flatten()
            if key in flatten_dict:
                return flatten_dict[key]
            else:
                raise KeyError(f"Key '{key}' not found in self")
            # if not hasattr(self, 'param_nests_path'):
            #     raise KeyError(key)
            # if key not in self.param_nests_path:
            #     raise KeyError(key)
            # path = self.param_nests_path[key]
            # item = self
            # for nest in path:
            #     # Используем super().get(), если вложенные элементы — это тоже экземпляры вашего класса
            #     # или проверяем, есть ли у item метод __getitem__
            #     # item = super().__getitem__(nest)
            #     item = super().__getitem__(nest)
            # return item[key]
            # except KeyError:
            #     raise KeyError(f"Key '{key}' not found in self or param_nests_path")
            # item = self
            # for nest in self.param_nests_path[key]:
            #     item = item[nest]
            # return item[key]
    def __setitem__(self, key, value):
        if not self.initialized:
            super().__setitem__(key, value)
        else:
            if key in ['baseliner', 'baseline_algo'] and self:
                
                temp = dict.__getitem__(self,'baseline_algo')
                try:
                    super().__setitem__('baseline_algo', value)
                    flatten_dict = self.flatten()
                    functions_list = self._init_functions_list 
                    config_path = self._init_config_path
               
                    # self["baseline_configs"]={}
                    # func_code = self['baseliner']([1]).__dict__["__wrapped__"].__code__
                    # for arg in tuple(self.keys()):
                    #     if arg in func_code.co_varnames[:func_code.co_argcount]:
                    #         self["baseline_configs"][arg] = self.pop(arg)
                    self.__init__(functions_list, config_path, **flatten_dict) #TODO: изменить этот упрощённый костыль. Необходимо реинициализировать только baseliner с перебором всех параметров, а не весь конфиг 

                except Exception as e:

                    super().__setitem__('baseline_algo', temp)
                    raise e
            if dict.__contains__(self, key):
                dict.__setitem__(self, key, value)
            else:     
                path = dict.get(self.param_nests_path, key)
                if path is not None:
                    item = self
                    for nest in path:
                        item = dict.__getitem__(item, nest)
                        if not isinstance(item, dict):
                            raise KeyError(f"Cannot traverse path at '{nest}', got {type(item).__name__}")
                    if key in ['shift_range', 'smooth_window']:
                        item[key].parameter = value
                    elif dict.__contains__(item,key):
                        dict.__setitem__(item, key, value)
                    else:
                        dict.__setitem__(self, key, value)
                else:
                    dict.__setitem__(self, key, value)
                
    def __getstate__(self):
        """
        Возвращает состояние объекта для сериализации.
        """

        state = {
            'dict_data': dict(self), 
            'param_nests_path': self.param_nests_path,
            '_init_functions_list': self._init_functions_list,
            '_init_config_path': self._init_config_path,
            'initialized': self.initialized
        }
        return state
    def __setstate__(self, state):
        """
        Восстанавливает состояние объекта после десериализации.
        """
        self.param_nests_path = dict(state.get('param_nests_path', {}))
        self._init_functions_list = state.get('_init_functions_list', [])
        self._init_config_path = state.get('_init_config_path', None)

        dict.update(self, state.get('dict_data', {}))
        self.initialized = state.get('initialized',False)
        
        
        # print(self)
    def flatten(self, data = None):
        flattened = {}
        if data is None:
            data = self
        for key in data.keys():
            if isinstance(data[key],dict):
                flattened.update(self.flatten(data[key]))
            else:
                flattened[key] = data[key]
        return flattened

    # def open(path):
    #     if not path.endswith('.yaml'):
    #         if os.path.exists(path + '.yaml'):
    #             path = path + '.yaml'
    #         else:
    #             path = find_paths(path, file_end='.yaml')
    #             if len(path) > 1:
    #                 raise ValueError(f"Multiple file paths were found:\n{path}\nPlease specify the direct path to the desired file.")
    #             elif not path:
    #                 raise FileNotFoundError('Файл не найден.')
    #             else:
    #                 path = path[0]
    #     return Configs(None, config_path=path)
    
    def save(self, path2dir = "", file_end = "_proccesing_settings"):
        self.dump(path2dir, file_end)
    def dump(self, path2dir = "", file_end = "_proccesing_settings"):
        if not path2dir.endswith(".yaml"):
            if path2dir.endswith(file_end):
                path = path2dir + ".yaml"
            else:
                path = path2dir + file_end +".yaml"
        else:
            path = path2dir
        with open(path,"w") as file:
            self._dump_nest(self,file)
    def _dump_nest(self,nest,file):
        baseliner_temp = False
        if 'baseliner' in nest:
            baseliner_temp = nest.pop('baseliner')

        for key in nest.keys():
            if isinstance(nest[key],dict):
                self._dump_nest(nest[key],file)
            elif isinstance(nest[key],AdaptiveParameter):
                par_temp = copy.copy(nest[key].parameter)
                yaml.safe_dump({key: par_temp}, file, default_flow_style=False, sort_keys=False)
            elif isinstance(nest[key],DatasetHeaders):
                pass
            else:
                if callable(nest[key]):
                    yaml.safe_dump({key: nest[key].__name__},file, default_flow_style=False, sort_keys=False)
                else:
                    try:
                        yaml.safe_dump({key: nest[key]}, file, default_flow_style=False, sort_keys=False)
                    except:
                        yaml.safe_dump({key: list(nest[key])}, file, default_flow_style=False, sort_keys=False)
        
        if isinstance(baseliner_temp,AdaptiveParameter):
            nest["baseliner"] = baseliner_temp
def _set_local_proc_configs(DataProc_configs, data_mz):
    dots_distance = np.median(np.diff(data_mz))
    local_configs = copy.deepcopy(DataProc_configs)
    local_configs['smoothing_configs']['smooth_window'](dots_distance) 
    local_configs['msalign_configs']['shift_range'](dots_distance)
    local_configs['baseliner'](data_mz)
    return local_configs

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


def _noaln_sequence(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        if kwargs.get('msalign_configs'):
            kwargs['msalign_configs']['align_peaks'] = None
        elif kwargs.get('align_peaks'):
            kwargs['align_peaks'] = None
        elif kwargs['DataProc_configs'].get('msalign_configs'):
            kwargs['DataProc_configs']['msalign_configs']['align_peaks'] = None
        logger.log('No align processing started')
        func(*args, **kwargs, dataset_name = 'peaklists_noaln')
    return wrapper
# class DataSource: TODO: Develop a source class that we can work with. It doesn’t matter which data source we use — IMZML or HDF5 or maybe MZXML; the approach will be the same.
#     """
#     Класс для работы с различными источниками масс-спектрометрических данных (IMZML, HDF5, MZXML).
#     Обеспечивает унифицированный интерфейс для получения данных m/z шкалы и интенсивностей спектра.
#     """

#     def __init__(self, path, dtypeconv):
#         """
#         Инициализация источника данных.
#         :param path: путь к файлу данных
#         """
#         self.dtypeconv = dtypeconv
#         self.path = path
#         if path.lower().endswith(".imzml"):
#             self.source = ImzMLParser(path)
#             self._loader = _load_imzml_mz

#         elif path.lower().endswith(".mzxml"):
#             self.source = mzxml.MzXML(path)
#             self._loader_intensity =
#             self._loader_mz =
#         elif path.lower().endswith(".hdf5"): !!!!!!!!!!!!!!!!!!! Разработать через idxroi. где определяем какой именно roi и sample используется
#             # self.source = File(path, 'r', libver='latest')
#             # self._loader_intensity =
#             # self._loader_mz =
#             raise ValueError("Неподдерживаемый формат файла")
#         else:
#             raise ValueError("Неподдерживаемый формат файла")

#     def _load_imzml(self, idx):
#         """
#         Загрузка данных из источника.
#         В зависимости от формата файла, данные загружаются разными способами.
#         """
#         mz, intens = zip(*(self.source.getspectrum(i) for i in idx))
#         return np.array(mz).astype(self.dtypeconv), np.array(intens).astype(self.dtypeconv)
#     def _load_hdf5(self, idx):
#         """
#         Загрузка данных из источника.
#         В зависимости от формата файла, данные загружаются разными способами.
#         """
#         return self.source
#         elif isinstance(self.source, h5py._hl.files.File):
#             # Для HDF5 структура может отличаться, пример предполагает стандартные группы
#             self.mz_scales = self.source['mz_scale']
#             self.intensities = self.source['intensities']
#         elif isinstance(self.source, mzxml.MzXML):
#             self.spectra = [spectrum for spectrum in self.source]
    
#     def __getitem__(self, index):
#         """
#         Получение спектра по индексу.
#         :param index: индекс спектра
#         :return: кортеж (mz_scale, intensities) для заданного спектра
#         """
#         if not hasattr(self, 'mz_scales') or not hasattr(self, 'intensities') and not hasattr(self, 'spectra'):
#             self._load_data()

#         if isinstance(self.source, ImzMLParser):
#             mz_scale, intensities = self.source.get_spectrum(index)
#         elif isinstance(self.source, h5py._hl.files.File):
#             # Предполагаем, что данные хранятся в виде массивов, где каждый спектр — это строка
#             mz_scale = self.mz_scales[index]
#             intensities = self.intensities[index]
#         elif isinstance(self.source, mzxml.MzXML):
#             spectrum = self.spectra[index]
#             mz_scale = spectrum['m/z array']
#             intensities = spectrum['intensity array']
#         return mz_scale, intensities

#     def close(self):
#         """
#         Закрытие источника данных.
#         """
#         self.source.close()

def _shift_range_to_dots(window_shift_mz, dots_distance):

    if window_shift_mz:
        return tuple(int(shift_mz/dots_distance) for shift_mz in window_shift_mz)
    else:
        return None

def _smooth_window_to_dots(smooth_window_mz, dots_distance):
    if smooth_window_mz:
        return int(smooth_window_mz/dots_distance)
    else:
        return None
def _baseliner_prep(baseliner,mz_scale):
    if baseliner:
        return getattr(Baseline(mz_scale),baseliner)
    else:
        return None
### Base functions

def imzml2hdf5(path_list, dtypeconv='single', chunk_rowsize = "Auto", chunk_bsize = 10000000, reconv = False): 
    # Очень важный TODO. ПОЗВОЛИТ РАБОТАТЬ С METASPACE ДАННЫМИ!!!: превратить эту функцию в конвертацию центроидных (и уже обработанных) данных imzml в hdf5 (типа в пиклист)
    """
    Description
    ----
    Conversion of raw data from imzML to hdf5 according to a list of file paths/root folders from path_list.

    If a folder path rather than a file path is provided, a search for imzML files in subfolders is performed.

    The conversion of imzML file data is written to the folder above the root folder for the imzML file. 
    If the upper folder is common for several files, their data is written to a single hdf5 file but into different hdf5 datasets.
    The dataset name is taken from the name of the folder containing the imzML file.

    :param path_list: list of str or paths to folder or `imzML` file
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"double"`
    :param chunk_rowsize: chunking hdf5 datasets for partial and efficient loading data to RAM. The default is `"Auto"`

        `"Auto"` - автоматический подбор кол-ва строк записи матрицы в hdf5 на основе размера chunk_bsize,

        `"Full"` - датасет сохраняется и выгружается целиком (занимает много RAM), 

        `int` - датасет дробится по заданному числу строк, где каждая строка это данные спектра.
    :param chunk_bsize: the chunk size for hdf5 datasets in bytes. Optional, works only with chunk_rowsize =`"Auto"`. The default is `10000000`
    
    :type path_list: list or str
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type chunk_rowsize: {`"Auto"`,`"Full"`, `int`}
    :type chunk_bsize: `int`, optional

    :return: None
    """
    logger("imzml2hdf5",{**locals()})
    if isinstance(path_list, str):
        path_list=[path_list]
    sample_imzmlpath_list=[] # Словарь списка sample для каждого слайда (отдельный путь в path_list)
    
    logger.log("Finding paths to imzml")
    imzmlpath_list = find_paths(path_list) ## Поиск файлов imzml и создание списка корневых папок с файлами ".imzml"
    sample_tot_num = len(imzmlpath_list)  # счётчик общего количества sample, используется для создания количества процессов не более этого значения (не критично, но оптимально вдруг, чтобы не создавать пул нерабочих процессов, что возможно ест ресурс компа)
    logger.log(f"Paths to imzml:{imzmlpath_list}")
    if sample_tot_num ==0:
        logger.warn("Sample total num is - 0. Couldn't find imzML files")
        return
    ##
    ## Создание списков наименований слайдов, roi и рассчёт общего количества roi
    logger.log(f"Preparation paths for multiprocessing")
    for path in imzmlpath_list:
        
        Slides_path=os.path.dirname(os.path.dirname(path)) #Определяем путь в root директорию. Это директория, где будут храниться данные обработки в hdf5 файле
        Slide_name=os.path.basename(Slides_path) #Определяем имя слайда
         
        if all('_rawdata.hdf5' not in item for item in os.listdir(Slides_path)): # условие для выгрузки из imzml и конвертации данных в hdf5     
            sample_imzmlpath_list.append([Slides_path,path])
        else:
            if reconv:
                os.remove(os.path.join(Slides_path,Slide_name)+"_rawdata.hdf5")
                sample_imzmlpath_list.append([Slides_path,path])
                logger.log(f"Old data on {Slides_path} deleted")
            else:
                logger.log(f"Data on the path {path} has hdf5 file for raw data. Change argument 'reconv' to True, if needed to reconvert")
                print(f"Data on the path {path} has hdf5 file for raw data. Change argument 'reconv' to True, if needed to reconvert")

    ##
    ## Определение количества пула процессов
    cpu_num = cpu_count()-1
    if cpu_num > sample_tot_num:
        cpu_num = sample_tot_num
    logger.log(f"Num of CPU for usage {cpu_num}")
    corenum_counter = Value('i',0) 
    ## Выгрузка данных с помощью ImzMLParser'a и их конвертация в hdf5 (в дальнейшем работаем с hdf5)
    logger.log(f"Creating Queue for getting information and controling  pool processes")
    manager = Manager()
    print_queue = manager.Queue()
    queue = manager.Queue()
    queue.put(True)
    logger.log(f"Creating thread for print and bar progress")
    t = Thread(target=printer,args=[print_queue])
    t.start()
    print_queue.put(len(sample_imzmlpath_list))
    logger.log(f"Starting conversion imzml to hdf5")
    with Pool(cpu_num,initializer=init_worker,initargs=['hdf5_writer',corenum_counter]) as p:
        p.starmap(hdf5_writer,product(sample_imzmlpath_list,[queue],[print_queue],[dtypeconv],[chunk_rowsize],[chunk_bsize]))
    p.join()
    logger.log(f"Conversion ended")
    print_queue.put(Sentinel()) # Остановка работы функции printer
    t.join() # Wait for all printing to complete
    ##
    logger.ended()
    return None

def Raw2proc(data_obj_path,
             hdf5_save_folder = None,
             hdf5_end = "_specdata", 
             draw = True, 
             plot_mz_range = None, 
             sample_spectra_idx = None,
             Ram_GB = 3, 
             h5chunk_size_MB = 10,
             dtypeconv='single',
             free_cores=1,
             configs = None,
             dataset_name = "int",
             eval_align = False,
             **kwargs):
    """
    Общее описание
    ----
    Функция обработки сырых спектров из `imzML` только до обработанных спектров и после их записи в файл hdf5 под названием "Slidename_specdata.hdf5" в датасет "int". Конечным результатом является обработанный спектр. Требуется много места на жёстком диске.

    :param data_obj_path: list of paths to root folders where to search imzml files in subfolders 
    :param baseliner: Baseline class for baseline correction
    :param baseline_algo: Algorithm of baseline correction. Default: `"asls"`

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    :param params2baseline_correction: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html). Default: `{}`

        .. Example: {"lam" : 500000, "diff_order" : 1}
    :param align_peaks: list of reference peaks for align. Default: `None`
    :param weights_list: list of weights for reference peaks in aligning. Default: `None`
    :param max_shift_mz: max spectrum shift at aligning in mz. Default: `0.95`
    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`). Default: `{}`

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param smooth_algo: spectrum smoothing algorithm. Default is `"None"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param smooth_window: window size in mz for smooth. Default:`0.075`
    :param smooth_cycles: Number of iterations for spectrum smooth. Default: `1`
    :param draw: Draw example graphs of raw and proccessed random spectrum of image. Default: `True`
    :param plot_mz_range: Range for graphs draw. Default: `None`
    :param rewrite: delete old hdf5 before writing new spectra data. Default: `False`
    :param Ram_GB: Determine max sizes in GB of the data proccesing on CPU cores at moment.
    :param h5chunk_size_MB: the chunk size for hdf5 datasets writing in MB
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param free_cores: Number of CPU cores don't used in multiproccessing

    :type data_obj_path: `list`
    :type max_shift_mz: `float`
    :type resample_to_dots: `int`
    :type baseliner: `Baseline` class
    :type baseline_algo: `str`
    :type params2baseline_correction: `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`,`None`}
    :type smooth_window: `float`
    :type smooth_cycles: `int`
    :type draw: `bool`
    :type plot_mz_range: `list` or `None`
    :type rewrite: `bool` #deprecated
    :type Ram_GB: `float`
    :type h5chunk_size_MB: `float`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type free_cores: `1`

    :return: `None`
    :rtype: `NoneType`
    """
    
    
    # Process args
    if isinstance(configs,Configs):
        configs.__init__([msalign,smoothing,DataProc_array],**configs)
    else:
        configs = Configs([msalign,smoothing,DataProc_array], configs,**kwargs)

    resample_to_dots = configs["resample_to_dots"]
    # DataProc_configs = configs['DataProc_configs']
    #Create thread for printing text in multiprocessing
    logger("Raw2proc",{**locals()})
    # Process args
    # defaults parametres for align
    #Create thread for printing text in multiprocessing
    manager = Manager()
    print_queue = Manager().Queue()
    queue = manager.Queue()
    queue.put(True)
    t = Thread(target=printer,args=[print_queue])  
    t.start()
    if isinstance(data_obj_path,str):
        data_obj_path=[data_obj_path]

    # Определение количества пула процессов
    cpu_num = cpu_count()-free_cores
    Ram_GB = Ram_GB*1e+9
    batch_bsize = Ram_GB/cpu_num
    bytes_flsize = BYTES_FLOAT_SIZE[dtypeconv]

    ###I. Finding slide directory for rawdata of samples (imzml)
    path_dict=find_imzml_roots(data_obj_path)
    logger.log(f'Path to files dictionary: {path_dict}')
    
    # Working with slides
    for folder_path in path_dict.keys():
        slide = os.path.basename(folder_path)  
        print(f"The {slide} raw spectra data is on progress.")
        data_obj_coord={}
        sample_list = path_dict[folder_path]
        if hdf5_save_folder is None:
            hdf5_save_folder = folder_path
        ###II. Extracting spectra coordinates, roi indexes, other metadata for proccessing slide samples from poslog and _info text files. Input arguments organization for "intprocc_parbatched_imzml" function (spectra processing batched and parallelized)
        print(f"Slide's {slide} spectra coordinates and metadata extraction for preparation parallel proccessing")
        logger.log(f"Slide's {slide} spectra coordinates and metadata extraction for preparation parallel proccessing")

        chunk_size_dict = {}
        if resample_to_dots:
            chunk_size_gen = max(1,np.ceil(h5chunk_size_MB*1e+6/(bytes_flsize*resample_to_dots)))
            for sample_file in sample_list.copy():
                chunk_size_dict[sample_file] = chunk_size_gen
        else:
            for sample_file in sample_list.copy():
                with ImzMLParser(sample_file) as file:
                    if "continuous" in file.metadata.pretty().get("file_description"):
                        dcont = file.metadata.pretty().get("file_description").get("continuous")
                    elif "processed" in file.metadata.pretty().get("file_description"):
                        dcont = not file.metadata.pretty().get("file_description").get("processed")
                    else:
                        dcont = False
                    if dcont:
                        chunk_size_dict[sample_file] = max(1,np.ceil(h5chunk_size_MB*1e+6/(bytes_flsize*file.mzLengths[0])))
                    else:
                        print(f"Discontinues data of sample {sample_file} requires resampling for HDF5 export. \nSolutions:\n- Set `resample_to_dots` value\n- Use `Raw2peaklist` for peak-based output")
                        sample_list.remove(sample_file)
                        continue
        if not sample_list:
            print(f"No suitable files found in the specified path: {folder_path}")
            continue
        hdf5_save_path = create_file_path(hdf5_save_folder, slide, hdf5_end)
        config_path2save = os.path.join(os.path.dirname(hdf5_save_path),slide)
        with Pool(cpu_num) as p:
            data_obj_temp = p.starmap(setup_spectra_batching,
                                      list(product(sample_list,
                                                   [hdf5_save_path],
                                                   [config_path2save],
                                                   [batch_bsize],
                                                   [dtypeconv],
                                                   [print_queue],
                                                   [cpu_num],
                                                   [configs],
                                                   [queue],
                                                   [chunk_size_dict],
                                                   [dataset_name]
                                                   )
                                                   )
                                                   )
        p.join()    
        args_batches=[]
        logger.log(f"Preparing arguments and data for parallelization")
        for data in data_obj_temp:
            data_obj_coord.update(data[0])
            args_batches.extend(data[1])

        del data_obj_temp
        gc.collect()
        ##II. Coordinates, metadata and organization of input arguments for parallelized and batched proccessing of spectra - DONE

        logger.log(f"Slide's {slide} spectra metadata writing")
        configs.dump(config_path2save)
        del_hdf5(hdf5_save_path)
        hdf5_metadata(hdf5_save_path, data_obj_coord, chunk_size_dict)

        logger.log(f"Data processing started")
        multiproc_processing(int2procc_parbatched,print_queue,cpu_num,args_batches, dataset_name, eval_align)

        ##IV. Drawing example result
        if draw:
            draw_data(hdf5_save_path,
                        sample_spectra_idx = sample_spectra_idx,
                        plot_mz_range = plot_mz_range,
                        imzml_source = True,
                        dtypeconv = dtypeconv,
                        **configs)
            if eval_align:
                _noaln_sequence(draw_data)(hdf5_save_path,
                        sample_spectra_idx = sample_spectra_idx,
                        plot_mz_range = plot_mz_range,
                        imzml_source = True,
                        dtypeconv = dtypeconv,
                        **configs)
        
    # Closing threads
    print_queue.put(Sentinel())
    t.join()
    logger.ended()
    gc.collect()
    return

def Raw2peaklist(data_obj_path, 
                 hdf5_save_folder = None,
                 hdf5_end = "_specdata", 
                 draw = True, 
                 plot_mz_range = None, 
                 sample_spectra_idx = None,  
                 Ram_GB = 3, 
                 h5chunk_size_MB = 10, 
                 dtypeconv='single',
                 free_cores=1, 
                 configs = None,
                 eval_align = False,
                 dataset_name = 'peaklists',
                 **kwargs):
    """
    Общее описание
    ----
    Функция обработки сырых спектров из `imzML` до пиклиста и после их записи в файл hdf5 под названием "Slidename_specdata.hdf5" в датасет "peaklists". Конечным результатом является полный пиклист-матрица имаджа.

    :param data_obj_path: list of paths to root folders where to search imzml files in subfolders 
    :param baseliner: Baseline class for baseline correction
    :param baseline_algo: Algorithm of baseline correction. Default: `"asls"`

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    :param params2baseline_correction: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html). Default: `{}`

        .. Example: {"lam" : 500000, "diff_order" : 1}
    :param align_peaks: list of reference peaks for align. Default: `None`
    :param weights_list: list of weights for reference peaks in aligning. Default: `None`
    :param max_shift_mz: max spectrum shift at aligning in mz. Default: `0.95`
    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`). Default: `{}`

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param smooth_algo: spectrum smoothing algorithm. Default is `"None"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `0`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `0`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `0`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `0`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `3.5`
    :param noise_est: алгоритм оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `"std"`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    :param smooth_window: window size in mz for smooth. Default:`0.075`
    :param smooth_cycles: Number of iterations for spectrum smooth. Default: `1`
    :param draw: Draw example graphs of raw and proccessed random spectrum of image. Default: `True`
    :param plot_mz_range: Range for graphs draw. Default: `None`
    :param rewrite: delete old hdf5 before writing new spectra data. Default: `False`
    :param Ram_GB: Determine max sizes in GB of the data proccesing on CPU cores at moment.
    :param h5chunk_size_MB: the chunk size for hdf5 datasets writing in MB
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param free_cores: Number of CPU cores don't used in multiproccessing

    :type data_obj_path: `list`
    :type max_shift_mz: `float`
    :type resample_to_dots: `int`
    :type baseliner: `Baseline` class
    :type baseline_algo: `str`
    :type params2baseline_correction: `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`,`None`}
    :type oversegmentationfilter: `float`
    :type fwhhfilter: `float`
    :type heightfilter: `float`
    :type peaklocation: `float` and =<1
    :type rel_heightfilter: `float`
    :type SNR_threshold: `float`
    :type noise_est: {`"std"`,`"mad"`}
    :type noise_est_iterations: `int`
    :type smooth_window: `float`
    :type smooth_cycles: `int`
    :type draw: `bool`
    :type plot_mz_range: `list` or `None`
    :type rewrite: `bool`
    :type Ram_GB: `float`
    :type h5chunk_size_MB: `float`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type free_cores: `1`

    :return: `None`
    :rtype: `NoneType`
    """
    if isinstance(configs,Configs):
        configs.__init__(FUNCTIONS_FOR_PROCCESING,**configs)
    else:
        configs = Configs(FUNCTIONS_FOR_PROCCESING, configs,**kwargs)

    logger("Raw2peaklist",{**locals()})
    #Create thread for printing text in multiprocessing
    manager = Manager()
    print_queue = Manager().Queue()
    queue = manager.Queue()
    queue.put(True)
    t = Thread(target = printer,args = [print_queue])  
    t.start()
    if isinstance(data_obj_path,str):
        data_obj_path=[data_obj_path]

    # Определение параметров по процессингу
    cpu_num = cpu_count()-free_cores
    Ram_GB = Ram_GB*1e+9
    batch_bsize = Ram_GB/cpu_num
    bytes_flsize = BYTES_FLOAT_SIZE[dtypeconv]
    chunk_size = max(1,np.ceil(h5chunk_size_MB*1e+6/(bytes_flsize*len(configs['peaks_configs']["headers"]))))

    ###I. Finding slide directory for rawdata of samples (imzml)
    path_dict=find_imzml_roots(data_obj_path)
    logger.log(f'Path to files dictionary: {path_dict}')

    # Processing
    for folder_path in path_dict.keys():
        slide = os.path.basename(folder_path)  
        print(f"The {slide} raw spectra data is on progress.")
        data_obj_coord={}
        sample_list = path_dict[folder_path]
        if hdf5_save_folder is None:
            hdf5_save_folder = folder_path
        ###II. Extracting spectra coordinates, roi indexes, other metadata for proccessing slide samples from poslog and _info text files. Input arguments organization for "intprocc_parbatched_imzml" function (spectra processing batched and parallelized)
        print(f"Slide's {slide} spectra coordinates and metadata extraction for preparation parallel proccessing")
        logger.log(f"Slide's {slide} spectra coordinates and metadata extraction for preparation parallel proccessing")
        hdf5_save_path = create_file_path(hdf5_save_folder, slide, hdf5_end)
        config_path2save = os.path.join(os.path.dirname(hdf5_save_path),slide)

        with Pool(cpu_num) as p:
            data_obj_temp = p.starmap(setup_spectra_batching,
                                      list(product(
                                          sample_list,
                                          [hdf5_save_path],
                                          [config_path2save],
                                          [batch_bsize],
                                          [dtypeconv],
                                          [print_queue],
                                          [cpu_num],
                                          [configs],
                                          [queue],
                                          [chunk_size],
                                          [dataset_name]
                                          )
                                          )
                                          )
        p.join()    
        args_batches=[]

        logger.log(f"Preparing arguments and data for parallelization")
        for data in data_obj_temp:
            data_obj_coord.update(data[0])
            args_batches.extend(data[1])

        del data_obj_temp
        # del temp
        gc.collect()
        logger.log(f"Slide's {slide} spectra coordinates writing")
        
        ##III. Writing metadata
        print(f"Slide's {slide} spectra coordinates writing")
        del_hdf5(hdf5_save_path)
        configs.dump(config_path2save)
        hdf5_metadata(hdf5_save_path,data_obj_coord,chunk_size)
        ##IV. Proccessing, peakpicking and writing to hdf5
        print(f"Slide's {slide} spectra parallel proccessing")
        
        multiproc_processing(int2proc2peaklist_parbatched, print_queue, cpu_num, args_batches, dataset_name, eval_align)
        # print_queue.put(0) #closing old tqdm bar
        
        if draw: #Секция для отрисовки полученных результатов
            if draw:
                draw_data(hdf5_save_path,
                            sample_spectra_idx = sample_spectra_idx,
                            plot_mz_range = plot_mz_range,
                            imzml_source = True,
                            dtypeconv = dtypeconv,
                            **configs)
                if eval_align:
                    _noaln_sequence(draw_data)(hdf5_save_path,
                                sample_spectra_idx = sample_spectra_idx,
                                plot_mz_range = plot_mz_range,
                                imzml_source = True,
                                dtypeconv = dtypeconv,
                                **configs)
                
    # if eval_align:
        #TODO: внести сравнение пиков
    # Closing threads and hdf5 object

    print_queue.put(Sentinel())

    t.join()
    logger.ended()
    gc.collect()


    return

def proc2peaklist(data_obj_path, 
                  file_end = '_specdata',
                  draw = True, 
                  plot_mz_range = None,
                  sample_spectra_idx = None, 
                  Ram_GB = 3, 
                  h5chunk_size_MB = 10, 
                  dtypeconv='single', 
                  free_cores=1, 
                  configs = None, 
                  eval_align = False,
                  dataset_name = 'peaklists',
                  **kwargs):
    """
    Общее описание
    ----
    Функция для получения пиклистов из обработанных спектров и после их записи в файл hdf5 под названием "Slidename_specdata.hdf5" в датасет "peaklists". Конечным результатом является полный пиклист-матрица имаджа.

    :param data_obj_path: list of paths to root folders where to search imzml files in subfolders 

    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `0`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `0`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `0`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `0`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `3.5`
    :param noise_est: алгоритм оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `"std"`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    :param draw: Draw example graphs of raw and proccessed random spectrum of image. Default: `True`
    :param plot_mz_range: Range for graphs draw. Default: `None`
    :param rewrite: delete old hdf5 before writing new spectra data. Default: `False`
    :param Ram_GB: Determine max sizes in GB of the data proccesing on CPU cores at moment.
    :param h5chunk_size_MB: the chunk size for hdf5 datasets writing in MB
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param free_cores: Number of CPU cores don't used in multiproccessing

    :type data_obj_path: `list`
    :type oversegmentationfilter: `float`
    :type fwhhfilter: `float`
    :type heightfilter: `float`
    :type peaklocation: `float` and =<1
    :type rel_heightfilter: `float`
    :type SNR_threshold: `float`
    :type noise_est: {`"std"`,`"mad"`}
    :type noise_est_iterations: `int`
    :type draw: `bool`
    :type plot_mz_range: `list` or `None`
    :type rewrite: `bool`
    :type Ram_GB: `float`
    :type h5chunk_size_MB: `float`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type free_cores: `1`

    :return: `None`
    :rtype: `NoneType`
    """
    if isinstance(configs,Configs):
        configs.__init__([peaks_prop_array],**configs)
    else:
        configs = Configs([peaks_prop_array], configs,**kwargs)

    PeakPicking_configs = configs['peaks_configs']
    logger("proc2peaklist",{**locals()})
    manager = Manager()
    print_queue = Manager().Queue()
    queue = manager.Queue()
    queue.put(True)
    t = Thread(target=printer,args=[print_queue])
    t.start()
    if isinstance(data_obj_path,str):
        data_obj_path=[data_obj_path]
    # Определение количества пула процессов
    cpu_num = cpu_count()-free_cores
    Ram_GB = Ram_GB*1e+9
    h5chunk_size_MB=h5chunk_size_MB
    bytes_flsize = BYTES_FLOAT_SIZE[dtypeconv]  
    ##
    chunk_size = max(1,np.ceil(h5chunk_size_MB*1e+6/(bytes_flsize*len(configs['peaks_configs']["headers"]))))
    if not file_end.endswith(".hdf5"):
        file_end+=".hdf5"
    path_list=find_paths(data_obj_path,file_end = file_end)

    for file_path in path_list:        
        slide = os.path.basename(file_path).replace(file_end,'')
        print(f"The {slide} processed spectra data is loaded from the hdf5 file.")
        config_path_base = file_path.replace(file_end,'')
        if os.path.exists(config_path_base+'_proccesing_settings.yaml'): 
            prev_configs = Configs(FUNCTIONS_FOR_PROCCESING , config_path_base + '_proccesing_settings.yaml', **kwargs)
            prev_configs.dump(config_path_base,file_end = "_proccesing_settings")
        else:
            logger.warn(
                        "Previous file configurations were not found. "
                        "As a result, mass spectra processing parameters were not inherited in the new processing settings file configuration."
                    )
            configs.dump(config_path_base,file_end = "_proccesing_settings")
            
        
        ##III. Peakpicking and writing to hdf5
        data_obj_feat = File(file_path,"a")
        if eval_align:
            datasets_to_del = [dataset_name, dataset_name + "_noaln"]
        else:
            datasets_to_del = [dataset_name]
        hdf5_repack_bool = del_datasets_hdf5(data_obj_feat, dataset_to_del = datasets_to_del)
        args_batches=[]
        print(f"Slide's {slide} spectra parallel peak picking")
        ## Ram managment
        num_of_processes_works=0
        for sample in data_obj_feat.keys():
            for roi in data_obj_feat[sample].keys():
                try:
                    roi_chunks = len(list(data_obj_feat[sample][roi]["int"].iter_chunks()))
                    chunk_size_MB = data_obj_feat[sample][roi]["int"].nbytes/roi_chunks
                    num_batch4chunks = int(np.ceil(data_obj_feat[sample][roi]["int"].nbytes/Ram_GB))
                    iter_chunks=[None]*roi_chunks
                    for n,chunk in enumerate(data_obj_feat[sample][roi]["int"].iter_chunks()):
                        iter_chunks[n]=chunk[0]
                except:
                    roi_chunks = int(np.ceil(data_obj_feat[sample][roi]["int"].nbytes/h5chunk_size_MB))+1
                    chunk_size_MB=h5chunk_size_MB
                    if roi_chunks <cpu_num*2:
                        chunk_size_MB=h5chunk_size_MB*roi_chunks/cpu_num*2
                        roi_chunks=cpu_num*2+1
                    if np.ceil(Ram_GB/chunk_size_MB)<cpu_num or np.ceil(data_obj_feat[sample][roi]["int"].nbytes/chunk_size_MB)<cpu_num:
                        chunk_size_MB=[Ram_GB/cpu_num]
                        chunk_size_MB.append(data_obj_feat[sample][roi]["int"].nbytes/cpu_num)
                        roi_chunks = int(np.ceil(data_obj_feat[sample][roi]["int"].nbytes/min(chunk_size_MB)))+1

                    num_batch4chunks = np.ceil(data_obj_feat[sample][roi]["int"].nbytes/Ram_GB)             
                    iter_chunks = list(slice(*x) for x in list(pairwise(np.linspace(0,data_obj_feat[sample][roi]["int"].shape[0],roi_chunks,dtype=int))))
                num_of_processes_works =num_of_processes_works+len(iter_chunks)
                batches = (np.array_split(np.array(iter_chunks),num_batch4chunks))
                args_batches += list(product(batches,[sample],[roi]))
        data_obj_feat.close()
        print_queue.put(num_of_processes_works)
        headers = PeakPicking_configs['headers']
        corenum_counter = Value('i',0) 
        with Pool(cpu_num,initializer = init_worker, initargs=['proc2peaklist_parbatched',corenum_counter]) as p:
            for args_batch in args_batches:

                sample=args_batch[1]
                roi = args_batch[2]
                args_batch = list(product(args_batch[0],[sample],[roi],[file_path],[PeakPicking_configs],[dtypeconv],[print_queue]))
                results = np.vstack(p.starmap(proc2peaklist_parbatched,args_batch))

                with File(file_path,'a') as data_obj_feat:
                    
                    try:
                        start_row = data_obj_feat[sample][roi][dataset_name].shape[0]
                        npeaks = results.shape[0]
                        data_obj_feat[sample][roi][dataset_name].resize(start_row + npeaks ,0)
                        data_obj_feat[sample][roi][dataset_name][start_row:(start_row+ npeaks),:] = results
                    except:
                        data_obj_feat.create_dataset(sample + "/" + roi + f"/{dataset_name}",results.shape, maxshape = (None, len(headers)), chunks=(chunk_size, len(headers)))

                        data_obj_feat[sample][roi][dataset_name][:] = results
                        data_obj_feat[sample][roi][dataset_name].attrs["Column headers"] = headers
        if hdf5_repack_bool:
            repack_hdf5(file_path)
        p.join()
        ##III. Peakpicking and writing to hdf5 - Done
        ##IV. Writing results
        print(f"Slide's {slide} spectra writing feature results")
        if draw: #Секция для отрисовки полученных результатов
            draw_data(file_path,
                        sample_spectra_idx = sample_spectra_idx,
                        plot_mz_range = plot_mz_range,
                        **PeakPicking_configs)

        print_queue.put(0) # закрываем tqdm
        # if eval_align:
            #TODO: addproc. 
            # 1. Отобрать пути семплов Done
            # 2. Получить список параметров с отключением align_peaks в none
            # 3. Чистим датасет
            # 4. Проводим подготовку параметров как в Raw2peaklists c помощью setup... 
            # 5. Запускаем обработку
            # 6. Сделано.
            # 7. Проводим оценку выравнивания кодом Игоря

    gc.collect()
    # Closing threads and hdf5 object
    logger.ended()
    print_queue.put(Sentinel())
    t.join()
    
    return

### Utility functions for multiprocessing
class Sentinel: 
    """Утилитный класс-заглушка для прекращения цикла while в функции printer и настройки атрибутов"""
    pass

def printer(print_queue):
    '''
    Вспомогательная функция используется для отображения сообщений в дочерних процессах, включая отображение прогресса с помощью tqdm.
    Еcли с помощью put на входе str, то печатает сообщение, если на входе число первый раз, то создаёт tqdm объект и отображает прогресс, если повторно число - то удаляет tqdm объект, если True, то отображает продвижение процесса.
    :param print_queue: is a multiproccesing.Manager.Queue() proxy object
    :type print_queue: multiproccesing.Manager.Queue() proxy object
    '''
    while True:
        msg = print_queue.get()
        if isinstance(msg, Sentinel):
            try:
                pbar.close()
            except:
                pass
            break
        elif isinstance(msg,str):
            print(msg, flush=True)
        elif msg is True:
            pbar.update(1)
        else:
            try:
                pbar.close()
                del pbar
            except:
                pbar = tqdm(total = msg,desc="Batches progress",smoothing = 0.005)       

def hdf5_writer(foldersample_path, queue,print_queue, dtypeconv,chunk_rowsize,chunk_bsize):
    """
    Общее описание
    ----
    Вспомогательная функция для imzml2hdf5
    
    :param foldersample_path: list of str with paths `imzML` file
    :param queue: Менеджер для задержки работы процесса перед записью данных датасета. Необходимо для однопоточной записи данных, так как запись данных в hdf5 невозможна в мультипоточном режиме
    :param print_queue: Менеджер для печати сообщений на экран с процесса.
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param chunk_rowsize: chunking hdf5 datasets for partial and efficient loading data to RAM. The default is `"Auto"`

        `"Auto"` - автоматический подбор кол-ва строк записи матрицы в hdf5 на основе размера chunk_bsize,

        `"Full"` - датасет сохраняется и выгружается целиком (занимает много RAM), 

        `int` - датасет дробится по заданному числу строк, где каждая строка это данные спектра.
    :param chunk_bsize: the chunk size for hdf5 datasets in bytes. Optional, works only with chunk_rowsize =`"Auto"`. The default is `10000000`
    
    :type foldersample_path: list
    :type queue: Manager.Queue()
    :type print_queue: Manager.Queue()
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type chunk_rowsize: {`"Auto"`,`"Full"`, `int`}
    :type chunk_bsize: `int`, optional

    :return: None
    :rtype: Nonetype
    """
    logger(f"hdf5_writer_on_core_{hdf5_writer.name}",{**locals()})
    Slide_folder_path = foldersample_path[0]
    Slide_name = os.path.basename(Slide_folder_path)
    sample_path2imzml = foldersample_path[1]
    folder_path2imzml = os.path.dirname(sample_path2imzml)
    sample_name = os.path.splitext(os.path.basename(sample_path2imzml))[0]
    logger.log(f"Path to slide folder {Slide_folder_path}\nPath to imzml {sample_path2imzml}\nSample name {sample_name}")
    ## Извлечение из poslog физических координат
    sample_data={}
    count=0
    idx_first=0
    roi_idx = {}
    
    try:
        sample_imzml=ImzMLParser(sample_path2imzml)
    except FileNotFoundError: #Если нет imzML файла в папке - пропуск
        print_queue.put(f'No {sample_path2imzml} file in directory {Slide_folder_path}')
        return
    
    try:
        dcont = sample_imzml.metadata.pretty()["file_description"]["continuous"]
    except KeyError:
        dcont = not sample_imzml.metadata.pretty()["file_description"]["processed"]

    poslog_specdata = [None]*len(sample_imzml.coordinates) #Данные строк в poslog с записью roi и координат снятого спектра.
    roi_list = []
    dots_num={}
    logger.log("Loading data and metadata")
    try:
        with open(os.path.join(folder_path2imzml,sample_name)+"_poslog.txt") as f:
            data = f.readlines()
            
            ##первая итерация записи координат начиная с третьей строки
            coords =  data[2].split(' ')
            roi_num = re.search('R(.+?)X', data[2]).group(1)
            roi_list.append(roi_num)
            poslog_specdata[count]=(roi_num,float(coords[-3]), float(coords[-2]))
            
            sample_data[roi_num]={}
            roi_idx[roi_num] = idx_first
            
            sample_data[roi_num]["mz"] = np.array(sample_imzml.getspectrum(roi_idx[roi_num])[0],dtype=dtypeconv)
            sample_data[roi_num]["z"] = np.array(float(coords[-1]),dtype=dtypeconv)
            
            count+=1
            ## продолжение итераций    
            for i in range(2,len(data)-1):
                coords =  data[i+1].split(' ')
                
                if(coords[-4]!='__'):
                    roi_num = re.search('R(.+?)X', data[i+1]).group(1)
                    poslog_specdata[count]=(roi_num,float(coords[-3]), float(coords[-2]))
                    
                    if roi_num not in roi_list[-1]:
                        roi_list.append(roi_num)
                        sample_data[roi_num]={}
                        roi_idx[roi_num] = []
                        roi_idx[roi_list[-2]] = (idx_first, count-idx_first)
                        
                        idx_first=count
                        sample_data[roi_num]["mz"] = np.array(sample_imzml.getspectrum(roi_idx[roi_num])[0],dtype=dtypeconv)
                        sample_data[roi_num]["z"] = float(coords[-1])
                    
                    count +=1
            roi_idx[roi_num] = (idx_first, count-idx_first)
            
            ## Preallocating координаты и int
            for roi in roi_list:
                dots_num[roi] = len(sample_data[roi]["mz"])
                sample_data[roi]["xy"] = np.empty((roi_idx[roi][1],2))
                sample_data[roi]["int"] = np.empty((roi_idx[roi][1],dots_num[roi]), dtype=dtypeconv)
                ### Определим Chunksize при автоматическом определениии размера
                if chunk_rowsize == "Auto":
                    bytes_flsize = BYTES_FLOAT_SIZE[dtypeconv]
                    chunk_rowsize = np.ceil(chunk_bsize/(bytes_flsize*dots_num[roi]))
                ###
            
            ##
            ## Заполнение пустых матриц координат и int 
            for idx, (roi,x,y)  in enumerate(poslog_specdata):
                sample_data[roi]["xy"][idx,:] = [x, y]
                sample_data[roi]["int"][idx-roi_idx[roi][0],:] = sample_imzml.getspectrum(idx)[1]
        del coords, idx_first
    except FileNotFoundError:
        roi_num = "00"
        roi_list.append(roi_num)
        sample_data[roi_num]={}
        
        numspectra = len(sample_imzml.coordinates)

        sample_data[roi_num]["xy"] = np.empty((numspectra,2))
        
        if dcont:
            sample_data[roi_num]["mz"] = np.array(sample_imzml.getspectrum(0)[0],dtype=dtypeconv)
            dots_num[roi_num] = sample_data[roi_num]["mz"].shape[1]
            sample_data[roi_num]["int"] = np.empty((numspectra,dots_num[roi_num]), dtype=dtypeconv)
            for idx in range(numspectra):
                sample_data[roi_num]["xy"][idx,:] = sample_imzml.get_physical_coordinates(idx)
        else:
            print_queue.put(f"Sample: {sample_path2imzml}\nThe data in the imzml file is not continuous. It will not be recorded in HDF5 format.")
            print_queue.put(True)
            return # Заглушка. Нет идей как грамотно впихнуть данные в hdf5, где надо пихать матрицы, а не листы с произвольным размером 
            sample_data[roi_num]["mz"] = [0]*numspectra
            sample_data[roi_num]["int"] = [0]*numspectra
            for idx in range(numspectra):
                sample_data[roi_num]["mz"][idx],sample_data[roi_num]["int"][idx] = sample_imzml.getspectrum(idx)
                sample_data[roi_num]["xy"][idx,:] = sample_imzml.get_physical_coordinates(idx)

            sample_data[roi_num]["mz"]=np.array(list(zip_longest(*sample_data[roi_num]["mz"], fillvalue=0))).astype(dtypeconv).T
            sample_data[roi_num]["int"]=np.array(list(zip_longest(*sample_data[roi_num]["int"], fillvalue=0))).astype(dtypeconv).T
            dots_num[roi_num] = sample_data[roi_num]["mz"].shape[1]
        sample_data[roi_num]["z"] = sample_imzml.coordinates[0][-1]

    if chunk_rowsize == "Auto":
        bytes_flsize = BYTES_FLOAT_SIZE[dtypeconv]
        chunk_rowsize = chunk_bsize/(bytes_flsize*dots_num[roi_num])
    del roi_num, dots_num
    ##
    
    logger.log("Sample dataset name estimation")
    ## Автоматическое определение имени датасета
    folder_name=os.path.basename(folder_path2imzml)
    if sample_name == folder_name:
        ds_name = folder_name
    else:
        ds_name=folder_name+"_"+sample_name
    logger.log(f"Name is {ds_name}")
    ##
    ## Запись в hdf5
    logger.log(f"Waiting queue for writing")
    temp = queue.get()
    logger.log(f"Writing started")
    print_queue.put(f"Slide {Slide_name} with sample {sample_name} is waiting queue")

    hdf5_raw=File(os.path.join(Slide_folder_path,Slide_name)+"_rawdata.hdf5","a")
    
    if chunk_rowsize == "Full":
        
        for roi in roi_list:
            logger.log(f"Writing data without chunks for roi {roi}")
            print_queue.put(f"Slide {Slide_name} with sample {sample_name}"+" roi "+roi+" data writing is in progress")
            for type in  ['/mz','/int','/xy','/z']:
                hdf5_raw.create_dataset(ds_name+'/'+roi+type, data=sample_data[roi][type.replace("/","")])
            hdf5_raw[ds_name][roi].attrs['continues'] = dcont   
        hdf5_raw.close()
    else:
        
        for roi in roi_list:
            logger.log(f"Writing data with chunks rowsize {chunk_rowsize} for roi {roi}")
            print_queue.put(ds_name+" roi "+roi+" data writing is in progress")
            
            ### chunked version
            
            hdf5_raw.create_dataset(ds_name+'/'+roi+'/int', data=sample_data[roi]['int'],chunks=(chunk_rowsize,sample_data[roi]['int'].shape[1]))
            hdf5_raw.create_dataset(ds_name+'/'+roi+'/xy', data=sample_data[roi]['xy'],chunks=(chunk_rowsize,sample_data[roi]['xy'].shape[1]))
            hdf5_raw.create_dataset(ds_name+'/'+roi+'/mz', data=sample_data[roi]['mz'])
            hdf5_raw.create_dataset(ds_name+'/'+roi+'/z', data=sample_data[roi]['z'])
            hdf5_raw[ds_name][roi].attrs['continues'] = dcont #Data points type       
        hdf5_raw.close()
    logger.log(f"Writing ended")
    print_queue.put(f"{sample_path2imzml} data writing is finished")
    print_queue.put(True)
    queue.put(True)
    logger.ended()

def int2procc_parbatched(sample_file_path, 
                         hdf5_save_path, 
                         sample,
                         roi,
                         interval,
                         resampled_mz,
                         print_queue,
                         DataProc_configs={}, 
                         queue = None, 
                         chunk_size = 10000, 
                         dataset_name = "int"):
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой обработки спектров. Используется в функции `Raw`2proc.

    :param sample_file_path: path to spectra source `imzML`
    :param sample: Sample name
    :param roi: Roi name
    :param interval: spectrum idx range used for proccesing
    :param dots_num: number of dots in spectrum
    :param dcont: spectra type of the data continues or discontinues
    :param print_queue: Менеджер для отображения сообщений на экран с процесса.
    :param discon_resample_range: mz range for data resample if imzML data is discontinuous
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param args2procc: params packaged in dictionary for proccessing
    :param queue: Number of iterations for spectrum smooth. Default: `1`
    :param chunk_size: number of rows per chunk in hdf5
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    
    :type data_obj_path: `str`
    :type sample: `str`
    :type roi: `str`
    :type interval: `list`
    :type dots_num: `int`
    :type dcont: `bool`
    :type print_queue: `Manager.Queue()`
    :type discon_resample_range: `tuple`
    :type resample_to_dots: `int`
    :type args2procc: `dict`
    :type queue: `Manager.Queue()`
    :type chunk_size: `int`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `None`
    :rtype: `NoneType`
    """
    logger(f"Raw2proc_on_core_{int2procc_parbatched.name}",{**locals()})
    idx_range = range(*interval)
    sample_imzml=ImzMLParser(sample_file_path)
    temp = queue.get()
    logger.log(f"Reading data")
    try:
        with File(hdf5_save_path,'r', libver='latest') as hdf5:
            dtypeconv = hdf5[sample][roi].attrs['dtype']
            dcont = hdf5[sample][roi].attrs['continuous']
    except Exception as error:
        logger.warn(f"During reading data raised folowing exception:{error}")
    queue.put(True)

    if dcont:
        if resampled_mz is not None:
            data_mz = sample_imzml.getspectrum(0)[0].astype(dtypeconv)
            # Векторизованная интерполяция через numpy
            data_int = np.array(tuple(
                np.interp(resampled_mz, data_mz, intens, 
                        left=intens[0], right=intens[-1])
                for intens in (sample_imzml.getspectrum(idx)[1].astype(dtypeconv) for idx in idx_range)
            ))
            data_mz = resampled_mz
        else:
            data_mz = sample_imzml.getspectrum(0)[0].astype(dtypeconv)
            data_int = np.array(tuple(sample_imzml.getspectrum(idx)[1].astype(dtypeconv) for idx in idx_range))
        ## proccessing array
        data_int = DataProc_array(data_int,data_mz,**DataProc_configs)
        logger.log(f"Processing continues data ended")
    else:
        if resampled_mz is not None:
            data_mz = resampled_mz
            # Подготовка всех спектров сразу
            all_mz, all_intens = zip(*(sample_imzml.getspectrum(idx) for idx in idx_range))

            # Векторизованная интерполяция через numpy
            data_int = np.array(tuple(
                np.interp(data_mz, mz, intens, 
                        left=intens[0], right=intens[-1])
                for mz, intens in zip(all_mz, all_intens)
            ))

            data_int = DataProc_array(data_int,data_mz,**DataProc_configs)
            logger.log(f"Processing discontinues data with resampling is ended")
        else:
            
            print_queue.put(f"Discontinues data of sample {sample} roi {roi} requires resampling for HDF5 export. \nSolutions:\n- Set `resample_to_dots` value\n- Use `Raw2peaklist` for peak-based output")
            return 

    ### Запись пиков в hdf5 очередью
    temp = queue.get()

    dots_num = len(data_mz)
    logger.log(f"Writing processed mass spectra")
    with File(hdf5_save_path,"a") as hdf5:
        idx_start,numspec = hdf5[sample][roi].attrs['idxroi']
        sl = range(interval[0]-idx_start,interval[1]-idx_start)
        if rf'{sample}/{roi}/int' in hdf5:
            hdf5[sample][roi]["int"][sl,:] = data_int
        else:
            logger.log(f"Creating datasets")
            hdf5.create_dataset(sample + "/" + roi + f"/{dataset_name}", (numspec, dots_num), chunks=(chunk_size, dots_num))
            logger.log(f"Intensity dataset created succesfully")
            hdf5.create_dataset(sample + "/" + roi + "/mz", data = data_mz)
            logger.log(f"mz dataset created succesfully")
            hdf5[sample][roi]["int"][sl,:] = data_int
    logger.log(f"Batch data is saved")
    logger.ended()
    print_queue.put(True)

    queue.put(True)
    
    return

def int2proc2peaklist_parbatched(sample_file_path,
                                 hdf5_file_path,
                                 sample,
                                 roi,
                                 interval, 
                                 resampled_mz, 
                                 print_queue,
                                 configs,
                                 queue = None, 
                                 chunk_size = 10000, 
                                 dataset_name = 'peaklists'):
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой обработки сырыах спектров до пиклистов, без сохранения промежуточных результатов. Используется в функции Raw2peaklist.

    :param sample_file_path: path to spectra source `imzML`
    :param sample: Sample name
    :param roi: Roi name
    :param interval: spectrum idx range used for proccesing
    :param dots_num: number of dots in spectrum
    :param dcont: spectra type of the data continues or discontinues
    :param print_queue: Менеджер для отображения сообщений на экран с процесса.
    :param discon_resample_range: mz range for data resample if imzML data is discontinuous
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param args2procc: params packaged in dictionary for spectra proccessing
    :param args2peakpicking: params packaged in dictionary for peakpicking
    :param queue: Number of iterations for spectrum smooth.
    :param chunk_size: number of rows per chunk in hdf5
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    
    :type data_obj_path: `str`
    :type sample: `str`
    :type roi: `str`
    :type interval: `list`
    :type dots_num: `int`
    :type dcont: `bool`
    :type print_queue: `Manager.Queue()`
    :type discon_resample_range: `tuple`
    :type resample_to_dots: `int`
    :type args2procc: `dict`
    :type args2peakpicking: `dict`
    :type queue: `Manager.Queue()`
    :type chunk_size: `int`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `None`
    :rtype: `NoneType`
    """
    logger(f"Raw2peaklist_on_core_{int2proc2peaklist_parbatched.name}",{**locals()})
    ## Пояснение, что какого-то файла не удалось найти/открыть при массовой обработке данных
    idx_range = range(*interval)
    sample_imzml=ImzMLParser(sample_file_path)
    DataProc_configs = configs["DataProc_configs"]
    PeakPicking_configs = configs["peaks_configs"]
    temp = queue.get()
    logger.log(f"Reading data")
    try:
        with File(hdf5_file_path,'r', libver='latest') as hdf5:
            idx_start = hdf5[sample][roi].attrs['idxroi'][0]
            dtypeconv = hdf5[sample][roi].attrs['dtype']
            dcont = hdf5[sample][roi].attrs['continuous']
    except Exception as error:
        logger.warn(f"During reading data raised folowing exception:{error}")
    queue.put(True)
    nspec_range=range(interval[0]-idx_start,interval[1]-idx_start)
    logger.log(f"Processing data")

    if dcont:
        if resampled_mz is not None:
            data_mz = sample_imzml.getspectrum(0)[0].astype(dtypeconv)
            # Векторизованная интерполяция через numpy
            data_int = np.array(tuple(
                np.interp(resampled_mz, data_mz, intens, 
                        left=intens[0], right=intens[-1])
                for intens in (sample_imzml.getspectrum(idx)[1].astype(dtypeconv) for idx in idx_range)
            ))
            data_mz = resampled_mz
        else:
            data_mz = sample_imzml.getspectrum(0)[0].astype(dtypeconv)
            data_int = np.array(tuple(sample_imzml.getspectrum(idx)[1].astype(dtypeconv) for idx in idx_range))
        ## proccessing array
        data_int = DataProc_array(data_int,data_mz,**DataProc_configs)
        
        ## Получение пиков
        peaklists = peaks_prop_array(data_mz,data_int,nspec_range,**PeakPicking_configs)
        logger.log(f"Processing continues data ended")
    else:
        if resampled_mz is not None:
            data_mz = resampled_mz
            # Подготовка всех спектров сразу
            all_mz, all_intens = zip(*(sample_imzml.getspectrum(idx) for idx in idx_range))

            # Векторизованная интерполяция через numpy
            data_int = np.array(tuple(
                np.interp(data_mz, mz, intens, 
                        left=intens[0], right=intens[-1])
                for mz, intens in zip(all_mz, all_intens)
            ))

            data_int = DataProc_array(data_int,data_mz,**DataProc_configs)
            peaklists = peaks_prop_array(data_mz,data_int,nspec_range,**PeakPicking_configs)
        else:
            peaklists={}
            ## Получение пиков
            for n, idx in enumerate(idx_range):
                data_mz, data_int = sample_imzml.getspectrum(idx)
                
                ## faster variant of function _set_local_proc_configs. This variant is without copy.deepcopy of configs, because this is not needed here
                dots_distance = np.median(np.diff(data_mz))
                DataProc_configs['smoothing_configs']['smooth_window'](dots_distance) 
                DataProc_configs['msalign_configs']['shift_range'](dots_distance)
                DataProc_configs['baseliner'](data_mz)

                # data_int = DataProc_1d(data_int,data_mz,**_set_local_proc_configs(DataProc_configs,data_mz)) #TODO вероятно забыл удалить _set_local_proc_configs. Надо проверить работоспособность кода без него
                data_int = DataProc_1d(data_int,data_mz,**DataProc_configs) #TODO вероятно забыл удалить _set_local_proc_configs. Надо проверить работоспособность кода без него

                peaklists[n] = peaks_prop_infunc(data_mz, data_int, np.where(np.diff(data_int) != 0)[0], len(data_mz),
                                           nspec_range[n], **PeakPicking_configs)
            
            peaklists = np.vstack(tuple(peaklists.values()))
        logger.log(f"Processing discontinues data ended")
        logger.log(f"Peaklists shape: {peaklists.shape}")
    ### Запись пиков в hdf5 очередью
    temp = queue.get()
    headers = PeakPicking_configs['headers']
    
    logger.log(f"Writing data")
    with File(hdf5_file_path,"a", libver='latest') as hdf5:
        logger.log(f"File {hdf5_file_path} is opened")
        if rf"{sample}/{roi}/{dataset_name}" in hdf5:
            start_row = hdf5[sample][roi][dataset_name].shape[0]
            hdf5[sample][roi][dataset_name].resize(start_row + peaklists.shape[0],0)
            hdf5[sample][roi][dataset_name][start_row:(start_row+peaklists.shape[0]),:] = peaklists
            logger.log(f"Added {dataset_name} data")
        else:
            logger.log(f"Creating {dataset_name} dataset for {sample} {roi}")
            n_heads = len(headers)
            hdf5.create_dataset(sample + "/" + roi + "/"+dataset_name,peaklists.shape, maxshape = (None, n_heads), chunks=(chunk_size, n_heads))
            hdf5[sample][roi][dataset_name][:] = peaklists
            hdf5[sample][roi][dataset_name].attrs["Column headers"] = headers
            if resampled_mz is not None:
                if rf"{sample}/{roi}/mz" not in hdf5:
                    hdf5.create_dataset(sample + "/" + roi + "/mz", data=resampled_mz)

    logger.log(f"Writing for {sample} {roi} in idx range {interval} ended succesfully")
    print_queue.put(True)
    queue.put(True)
    logger.ended()
    return
            
def proc2peaklist_parbatched(sl, 
                             sample,
                             roi,
                             sample_file_path, 
                             PeakPicking_configs={},
                             dtypeconv='single',
                             print_queue=None):
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессингового пикпикинга из обработанных спектров и сохранением полученных пиклистов в hdf5 под названием "[Slidename]_specdata.hdf5" в датасет "peaklists". Используется в функции proc2peaklist.

    :param sample_file_path: path to spectra source `imzML`
    :param sample: Sample name
    :param roi: Roi name
    :param interval: spectrum idx range used for proccesing
    :param dots_num: number of dots in spectrum
    :param dcont: spectra type of the data continues or discontinues
    :param print_queue: Менеджер для отображения сообщений на экран с процесса.
    :param discon_resample_range: mz range for data resample if imzML data is discontinuous
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param args2procc: params packaged in dictionary for spectra proccessing
    :param args2peakpicking: params packaged in dictionary for peakpicking
    :param queue: Number of iterations for spectrum smooth.
    :param chunk_size: number of rows per chunk in hdf5
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    
    :type data_obj_path: `str`
    :type sample: `str`
    :type roi: `str`
    :type interval: `list`
    :type dots_num: `int`
    :type dcont: `bool`
    :type print_queue: `Manager.Queue()`
    :type discon_resample_range: `tuple`
    :type resample_to_dots: `int`
    :type args2procc: `dict`
    :type args2peakpicking: `dict`
    :type queue: `Manager.Queue()`
    :type chunk_size: `int`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `None`
    :rtype: `NoneType`
    """
    ## Пояснение, что какого-то файла не удалось найти/открыть при массовой обработке данных
    with File(sample_file_path,'r', libver='latest', swmr=True) as data_obj:
        idx_range = range(sl.start,sl.stop)
        peaklists = peaks_prop_array(data_obj[sample][roi]['mz'][:].astype(dtypeconv), data_obj[sample][roi]['int'][idx_range,:].astype(dtypeconv),idx_range,**PeakPicking_configs)
    print_queue.put(True)
    return peaklists

def setup_spectra_batching(sample_file, 
                           hdf5_save_path,
                           config_path2save, 
                           batch_bsize, 
                           dtypeconv, 
                           print_queue,
                           cpu_num, 
                           configs, 
                           queue, 
                           chunk_size, 
                           dataset_name): #resample_to_dots, shift_range, baseliner, smooth_window): 
    """
    General description
    ----
    Parallel batch preparation pipeline for imaging data

    Processing workflow:
    1. ROI segmentation - Identifies regions of interest (ROI) from poslog metadata
    2. Coordinate extraction - Extracts spatial coordinates and Z-plane information
    3. Batch packaging - Prepares memory-efficient batches for parallel processing

    :param sample_file: Path to .imzML file
    :param batch_bsize: Batch size in bytes (auto-calculated based on RAM constraints)
    :param print_queue: Multiprocessing progress reporting queue
    :param cpu_num: Available CPU cores for parallelization
    :param shift_range: Maximum allowed shift on mz scale for msalign function
    :param baseliner: Initialized baseline correction object
    :param smooth_window: Smoothing wondow size in m/z units
    :param resample_to_dots: Target number of spectral points after resampling. Default: `None`
    :param dtypeconv: Float precision type: "half" (16b), "single" (32b), "double" (64b)

    :type sample_file: `str`
    :type batch_bsize: `float`
    :type cpu_num: `int`
    :type print_queue: `Manager.Queue()`
    :type shift_range: `float`
    :type baseliner: `str`
    :type smooth_window: `float`
    :type resample_to_dots: `int`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `tuple` with contains dict with coordinates and packaged params for multiproccessing
    :rtype: `tuple`
    """ 
    ### File path handling and validation
    
    sample_folder_path = os.path.dirname(sample_file)
    sample_name = os.path.splitext(os.path.basename(sample_file))[0]
    folder_name = os.path.basename(sample_folder_path)
    base_path = os.path.join(sample_folder_path, sample_name)
    poslog_err = sample_name
    if folder_name == sample_name:
        sample=sample_name
    else:
        sample = folder_name+"_"+sample_name

    try:
        sample_imzml=ImzMLParser(sample_file)
        data_obj={} 
        data_obj[sample]={}
        ### Get spectrum type storage (continuous or disctontinuous)
        if "continuous" in sample_imzml.metadata.pretty().get("file_description"):
            dcont = sample_imzml.metadata.pretty().get("file_description").get("continuous")
        elif "processed" in sample_imzml.metadata.pretty().get("file_description"):
            dcont = not sample_imzml.metadata.pretty().get("file_description").get("processed")
        else:
            dcont = False
    except FileNotFoundError as e: #Если нет imzML файла в папке - пропуск
        print_queue.put(f"File error: {str(e)}")
        return
    
    #### Stage 2. Get spectra sizes
    try:
        with open(base_path+"_info.txt") as f:
            data_info = f.readlines()
            spectrum_points = int(data_info[12].split(' ')[1]) # Информация по кол-ву точек спектра
            specnum = int(data_info[2].split(' ')[-1]) # Информация по кол-ву спектров в sample
    except:   
        dpoints = sample_imzml.mzLengths
        if dcont:
            spectrum_points = dpoints[0]
        else:
            spectrum_points = int(np.quantile(dpoints, 0.95))
        specnum = len(dpoints)
    resample_to_dots = configs.get("resample_to_dots", None)
    if resample_to_dots:
        spectrum_points = resample_to_dots
    poslog_path = base_path + "_poslog.txt"
    ### Stage 3. Data extraction
    if os.path.exists(poslog_path): ### Extraction from _poslog and _info text files
        roi_list, roi_idx, poslog_specdata = _poslog_parser(poslog_path, specnum)
        
        for roi in roi_list:
            data_obj[sample][roi]={}
            data_obj[sample][roi]["xy"] = np.empty((roi_idx[roi][1],2))
            data_obj[sample][roi]["z"] = np.empty((roi_idx[roi][1],1))
        for idx, (roi,x,y,z) in enumerate(poslog_specdata):            
            data_obj[sample][roi]["xy"][idx-roi_idx[roi][0],:] = [x, y]
            data_obj[sample][roi]["z"][idx-roi_idx[roi][0]]  = z
         
    else: ### If there is no poslog file in the folder, take coordinates from imzml
        print_queue.put(f'The {poslog_err+"_poslog.txt"} file is not in directory {sample_folder_path}, the coordinate data is taken from the imzML file')
        
        #### Stage 3 from imzml. Get base info and coordinates
        #### Initialization
        roi = "00" # Only one roi
        roi_list = [roi]
        roi_idx = {}
        data_obj[sample][roi]={}
        
        coords = np.empty((specnum,2))
        roi_idx[roi] = (0,specnum)
        try:
            for idx in range(specnum):
                coords[idx,:] = sample_imzml.get_physical_coordinates(idx)
            data_obj[sample][roi]['xy'] = coords
        except:
            data_obj[sample][roi]['xy'] = np.array(sample_imzml.coordinates)[:,[0,1]]
        data_obj[sample][roi]["z"] = np.array([0]*specnum) # Заглушка z- координаты нигде не узнать
    #### Stage 4. Batching and organization spectra data for parallel proccesing
    bytes_flsize = BYTES_FLOAT_SIZE[dtypeconv]
    spectra_chunksize = max(1, int(batch_bsize/(bytes_flsize*spectrum_points)))
    roi_count = len(roi_list)
    n_int = int(specnum/(roi_count*spectra_chunksize)+1)
    if n_int*roi_count<cpu_num*2:
        n_int = int(cpu_num*2/roi_count)+1

    par_args = []
    #### Stage 5. Setting some mz scale dependent parametres
    for roi in roi_list:
        local_configs = {}
        for key in configs.keys():
            if key.endswith("_configs"):
                local_configs[key] = configs[key].copy() 
        indexes = roi_idx[roi]
        data_obj[sample][roi]["continuous"] = dcont
        data_obj[sample][roi]["idxroi"] = indexes
        data_obj[sample][roi]["source"] = sample_file
        data_obj[sample][roi]["dtype"] = dtypeconv
        data_obj[sample][roi]["N_resampled"] = resample_to_dots
        data_obj[sample][roi]['configs'] = config_path2save
        
        #### 
        if dcont:
            data_mz = sample_imzml.getspectrum(indexes[0])[0].astype(dtypeconv)
            min_mz = min(data_mz)
            max_mz = max(data_mz)
        else:
            data_mz = sample_imzml.getspectrum(indexes[0])[0].astype(dtypeconv)
            min_mz = min(data_mz)
            max_mz = max(data_mz)
            for idx in range(indexes[0]+1,indexes[0]+indexes[1]):
                data_mz = sample_imzml.getspectrum(idx)[0].astype(dtypeconv)
                min_mz = min([min_mz,min(data_mz)])
                max_mz = max([max_mz,max(data_mz)])
        data_obj[sample][roi]["mz_range"] = (min_mz,max_mz)
        
        if resample_to_dots:
            resampled_mz = np.linspace(min_mz,max_mz,resample_to_dots).astype(dtypeconv)
            data_mz = resampled_mz
            
        else:
            resampled_mz = None
        
        if (resample_to_dots or dcont) and local_configs.get("DataProc_configs", False):
            # smooth_window, shift_range, baseliner = _adapt_proccesing_parameters(data_mz, smooth_window, shift_range, baseliner)
            # dots_distance = np.median(np.diff(data_mz))
            # local_configs["DataProc_configs"]['smoothing_configs'] = local_configs["DataProc_configs"]['smoothing_configs'].copy()
            # local_configs["DataProc_configs"]['msalign_configs'] = local_configs["DataProc_configs"]['msalign_configs'].copy()
            # local_configs["DataProc_configs"] = local_configs["DataProc_configs"].copy()
            # local_configs["DataProc_configs"]['smoothing_configs']['smooth_window'] = local_configs["DataProc_configs"]['smoothing_configs']['smooth_window'](dots_distance) 
            # local_configs["DataProc_configs"]['msalign_configs']['shift_range'] = local_configs["DataProc_configs"]['msalign_configs']['shift_range'](dots_distance)
            # local_configs["DataProc_configs"]['baseliner'] = local_configs["DataProc_configs"]['baseliner'](data_mz) 
            local_configs["DataProc_configs"] = _set_local_proc_configs(local_configs["DataProc_configs"], data_mz)
        
        linspace_values = np.linspace(
            roi_idx[roi][0],
            roi_idx[roi][0] + roi_idx[roi][1],
            n_int,
            dtype=int
            )
        pairwise_values = pairwise(linspace_values)
        if len(local_configs.keys()) == 1:
            local_configs = local_configs[list(local_configs.keys())[0]]
        par_args.extend(list(args) for args in product(
                [sample_file],
                [hdf5_save_path],
                [sample],
                [roi], 
                pairwise_values,
                [resampled_mz],
                [print_queue],
                [local_configs],
                [queue], 
                [chunk_size[sample_file] if isinstance(chunk_size, dict) else chunk_size], 
                [dataset_name]
                )
                )
    return (data_obj, par_args) #, shift_range, baseliner, smooth_window)

### Utility functions for processing
def _poslog_parser(poslog_path,specnum):
    idx=0
    roi_idx = {} # Информация sample по индексам спектров roi=(индекс первого спектра, кол-во спектров roi)
    roi_list = []
    current_roi = None
    poslog_specdata = [None]*specnum
    roi_pattern = re.compile(r'R(.+?)X')

    with open(poslog_path) as f:
        data = f.readlines()[2:]
    for line in data:
        line_search = roi_pattern.search(line)
        if not line_search:
            continue
        roi_num = line_search.group(1)
        coords =  line.split(' ')[-3:]
        try:
            x, y, z = map(float, coords)
        except (ValueError, IndexError):
            continue 
        if roi_num != current_roi:
            if current_roi is not None:
                roi_idx[current_roi] = (start_idx, idx - start_idx)
            current_roi = roi_num
            start_idx = idx
            roi_list.append(roi_num)
        poslog_specdata[idx]=(roi_num, x, y, z)
        idx += 1
    
    # Final ROI update
    if current_roi:
        roi_idx[current_roi] = (start_idx, idx - start_idx)
    return roi_list, roi_idx, poslog_specdata


def init_worker(func_name,corenum_counter):
    with corenum_counter.get_lock():
        eval(func_name).name = corenum_counter.value
        corenum_counter.value += 1
def multiproc_processing(func, print_queue, cpu_num, args_batches, dataset_name = None, eval_align = False):
    if eval_align is not None:
        logger.log(f"Data processing started")
        print_queue.put(len(args_batches))
        corenum_counter = Value('i',0) 
        with Pool(cpu_num, initializer = init_worker, initargs=[func.__name__, corenum_counter]) as p:
            p.starmap(func, args_batches)
        p.join()
        print_queue.put(0)
    if (eval_align is None) or eval_align:
        print(f"Processing peaklists without aligning to \"peaklists_noaln\" dataset")
        for args_batch in args_batches:
            local_conf = args_batch[-4]
            if local_conf.get('DataProc_configs', False):
                local_conf['DataProc_configs']['msalign_configs']['align_peaks'] = None
            elif local_conf.get("msalign_configs", False):
                local_conf['msalign_configs']['align_peaks'] = None
            if not dataset_name.endswith("_noaln"):
                args_batch[-1] = dataset_name + "_noaln"
        multiproc_processing(func,print_queue,cpu_num,args_batches, dataset_name, eval_align = False)

def DataProc_array(y,x, baseliner = None, baseline_configs = {},
                       msalign_configs={}, smoothing_configs = {}): 
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой предобработки спектров без ресемплинга. Function works only with continual data or with one dimensional array/list.
    
    :param y: array of spectra intensities with shape (n,d), where each row (n) corresponds to intensities of spectrum and column (d) corresponds to dots of spectra   
    :param x: array of spectra mz with shape (1,d)
    :param baseliner: Baseline class for baseline correction
    :param baseline_algo: Algorithm of baseline correction.

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    

    :param params2baseline_correction: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html)
        
        .. Example: {"lam" : 500000, "diff_order" : 1}

    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`).

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param align_peaks: list of reference peaks for align
    :param weights_list: list of weights for reference peaks in aligning
    :param dots_shift: max spectrum shift in dots
    :param smooth_algo: spectrum smoothing algorithm. Default is `"GA"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param smooth_window: window size for smooth
    :param smooth_cycles: Number of iterations for spectrum smooth

    :type y: `array`
    :type x: `array` or `list`
    :type baseliner: `Baseline` class
    :type baseline_algo: `str`
    :type params2baseline_correction:  `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`}
    :type smooth_window: `float`
    :type smooth_cycles: `int`

    :return: array of proccessed spectra
    :rtype: `np.array`

    """

    # Spectral Alignment
    if msalign_configs['align_peaks']:
        y = msalign(x,y,**msalign_configs)
    # Baseline Correction and Smoothing
    if baseliner:
        if smoothing_configs['smooth_algo']:
            return np.array(tuple(smoothing(spectrum - baseliner(spectrum,**baseline_configs)[0], **smoothing_configs) for spectrum in y))
        else:
            return y - np.array(tuple(baseliner(spectrum,**baseline_configs)[0] for spectrum in y))
    else:
        if smoothing_configs['smooth_algo']:
            return np.array(tuple(smoothing(spectrum,**smoothing_configs) for spectrum in y))

    return np.array(y)

def DataProc_1d(y,x,baseliner= None, baseline_configs={},
                       msalign_configs={}, smoothing_configs = {}): # Data preprocessing without resampling one dimensional
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой предобработки спектров без ресемплинга. Function works only with continual data or with one dimensional array/list.
    
    :param y: array of spectrum intensities with shape (1,d) or list   
    :param x: array of mz with shape (1,d)
    :param baseliner: Baseline class for baseline correction
    :param baseline_algo: Algorithm of baseline correction.

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    

    :param params2baseline_correction: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html)
        
        .. Example: {"lam" : 500000, "diff_order" : 1}

    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`).

        .. Example: {"iterations" : 2, "only_shift" : False}

    :param align_peaks: list of reference peaks for align
    :param weights_list: list of weights for reference peaks in aligning
    :param dots_shift: max spectrum shift in dots
    :param smooth_algo: spectrum smoothing algorithm. Default is `"GA"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param smooth_window: window size for smooth
    :param smooth_cycles: Number of iterations for spectrum smooth
    
    :type y: `array` or `list`
    :type x: `array` or `list`
    :type baseliner: `Baseline` class
    :type baseline_algo: `str`
    :type params2baseline_correction:  `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`}
    :type smooth_window: `float`
    :type smooth_cycles: `int`

    :return: array of proccessed spectra
    :rtype: `np.array`

    .. todo:: code refactoring with class
    """
    # Spectral Alignment
    if msalign_configs['align_peaks'] is not None: # TODO Задуматься о том, чтобы поставить выравнивание в конце обработки
        y = msalign(x,y,**msalign_configs).squeeze()
    # Baseline Correction and Smoothing
    if baseliner:
        if smoothing_configs['smooth_algo']:
            return smoothing(y - baseliner(y,**baseline_configs)[0], **smoothing_configs)
        else:
            return y - baseliner(y,**baseline_configs)[0]
    else:
        if smoothing_configs['smooth_algo']:
            return smoothing(y,**smoothing_configs)

    return np.array(y)

def find_imzml_roots(paths):
    path_dict={}
    if isinstance(paths,str):
        paths = [paths]
    for path in paths:
        if path.lower().endswith('.imzml'):
            Slide_folder = os.path.dirname(os.path.dirname(path))
            path_dict.setdefault(Slide_folder,[])
            path_dict[Slide_folder].append(path)
        else:
            for root, dirs, files in os.walk(path):
                for file in files: 
                    if file.lower().endswith('.imzml'):
                        path_dict.setdefault(os.path.dirname(root),[])
                        path_dict[os.path.dirname(root)].append(os.path.join(root,file))
                del root, dirs, files
    for key in path_dict.keys():
        path_dict[key]=list(set(path_dict[key]))
    return path_dict

def smoothing(y, smooth_algo, smooth_window, smooth_cycles):
    if len(y) == 0:
        return np.array([])
    
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    if smooth_algo == 'MA':
        return movaver(y, smooth_window, smooth_cycles)
    elif smooth_algo == 'GA':
        return gaussian_filter(y, smooth_window, smooth_cycles)
    elif smooth_algo == 'SG':
        return savgol(y, smooth_window, smooth_cycles)
    else:
        print("Smoothing algorithm not recognized")
        return y

def movaver(y, window, cycles):
    if window < 3 or len(y) < window:
        return y.copy()
    
    pad_size = window // 2
    kernel = np.ones(window) / window
    
    for _ in range(cycles):
        padded = np.pad(y, (pad_size, pad_size), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        y = smoothed
    
    return y

def gaussian_filter(y, window, cycles, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    
    if window < 3 or len(y) < window:
        return y.copy()
    
    for _ in range(cycles):
        y = gaussian_filter1d(y, sigma=sigma, mode='mirror')
    
    return y

def savgol(y, window, cycles, order=3):
    if window <= order or len(y) < window:
        return y.copy()
    
    for _ in range(cycles):
        y = savgol_filter(y, window_length=window, polyorder=order, mode='mirror')
    
    return y

def MAD(y,nan_policy):
    return sqrt(2*math.log(len(y)))*median_abs_deviation(y,nan_policy)/0.6745 # from matlab "mad" algorithm noise description (but this is for y_h to filter out noisy components in the first high-band decomposition of DCWT peak picking)

### Functions for peakpicking
def peaks_prop_array(X, 
                     Y_array,
                     spectra_ind, 
                     fwhhfilter = None,
                     oversegmentationfilter = None,
                     heightfilter = None,
                     rel_heightfilter = None,
                     peaklocation = 1,
                     noise_func = np.std,
                     noise_est_iterations = 3, 
                     SNR_threshold = None, 
                     Calc_peak_area = True,
                     headers = None):
    """
    Общее описание
    ----
    Функция для получения пиклиста с характеристиками пиков из одного спектра. Оптимизирована под использование в получении пиклистов в дисконтинуальных данных без resample'а данных.

    :param X: mz
    :param Y: Intensity
    :param spectra_ind: индекс спектра
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `None`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `None`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `None`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `None`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `None`
    :param noise_func: функция оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `np.std`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    
    :type X: `np.array`
    :type Y: `np.array`
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
    
    xsize = X.size
    peaklists = {}     
    for ns, ind in enumerate(spectra_ind):
        Y=Y_array[ns,:]
        peaklists[ns]=peaks_prop_infunc(X,
                                        Y,
                                        np.where(np.diff(Y) !=0)[0],
                                        xsize, 
                                        ind,
                                        fwhhfilter,
                                        oversegmentationfilter,
                                        heightfilter,
                                        rel_heightfilter,
                                        peaklocation,
                                        noise_func,
                                        noise_est_iterations,
                                        SNR_threshold, 
                                        Calc_peak_area,
                                        headers)

    return np.vstack(tuple(peaklists.values()))

def peaks_prop_infunc(X,
                      Y,
                      valley_dots,
                      xsize, 
                      spectra_ind,
                      fwhhfilter,
                      oversegmentationfilter,
                      heightfilter,
                      rel_heightfilter,
                      peaklocation,
                      noise_func,
                      noise_est_iterations, 
                      SNR_threshold, 
                      Calc_peak_area, 
                      headers = ["spectra_ind", "mz", "Intensity", "Area", "SNR", "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"]):
    """
    Общее описание
    ----
    Функция для получения характеристик пиков спектра. Если в каком-то параметре стоит `None`, то функция не будет производить фильтрацию или расчёты и не будет добавлять свойство пиков на выходе, что может экономить время и память.
    :param X: mz
    :param Y: Intensity
    :param valley_dots: numpy array того какие точки спектра явлюятся наклонными. На входе подаётся как `np.where(np.diff(Y) != 0)[0]` или как строка этого результата, если Y матрица. Этот параметр нужен скорее для универсальности функции и оптимизации кол-ва расчётов/обращений. 
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `None`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `None`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `None`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `None`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `None`
    :param noise_func: функция оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `np.std`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    
    :type X: `np.array`
    :type Y: `np.array`
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
    valley_dots = np.concatenate((valley_dots, [xsize-1]))    
    loc_min = np.diff(Y[valley_dots])
    loc_min = (np.array([True,*(loc_min < 0)])) & np.array(([*(loc_min > 0),True]))
    left_min = np.concatenate([[-1],valley_dots[:-1]])[loc_min][:-1] + 1
    right_min = valley_dots[loc_min][1:]

    # Compute max for every peak
    size = left_min.shape
    val_max = np.empty(size)
    for idx, [lm, rm] in enumerate(zip(left_min, right_min)):
        val_max[idx] = np.max(Y[lm:rm])
    
    # Remove peaks below the height, relative height
    if heightfilter and rel_heightfilter:
        k = (val_max >= heightfilter) & (val_max/max(Y) >= rel_heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]
    elif heightfilter:
        k = (val_max >= heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]
    elif rel_heightfilter:
        k = (val_max/max(Y) >= rel_heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]

    # Remove peaks below the SNR thresholds
    if SNR_threshold:
        noise_points = np.array([True]*xsize) # Zero iteration

        for it in range(noise_est_iterations):
            for idx in np.where(((val_max-np.mean(Y[noise_points]))/noise_func(Y[noise_points])>=SNR_threshold))[0]: #По сути тут расчёт z-score в чистом виде TODO: оценить скорость рассчётов моего варианта и scipy.stats.zscore
                sl = slice(left_min[idx],right_min[idx]+1)
                noise_points[sl] = False
        
        props["Noise"] = noise_func(Y[noise_points])
        props["Mean noise"]= np.mean(Y[noise_points])
        k = (val_max-props["Mean noise"])/props["Noise"]>=SNR_threshold
        
        val_max=val_max[k]
        left_min=left_min[k]
        right_min=right_min[k]

    # Compute FWHH for every peak
    size = left_min.shape
    props["FWHML"] = np.empty(size)
    props["FWHMR"]  = np.empty(size)
    pos_peak = np.empty(size)
    for idx, [lm, rm, vm] in enumerate(zip(left_min, right_min,val_max)):
        pp = lm + np.argmax(Y[lm:rm])
        pos_peak[idx] = pp
        props["FWHML"][idx] = np.interp(vm/2,Y[lm:pp+1], X[lm:pp+1])
        props["FWHMR"][idx] = np.interp(vm/2,Y[pp:rm+1][::-1], X[pp:rm+1][::-1])
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
    # Remove oversegmented peaks
    if oversegmentationfilter:
        if isinstance(oversegmentationfilter,str):
            oversegmentationfilter = np.median(props["FWHMR"]-props["FWHML"])
        while True:
            peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
            pkX = np.empty(left_min.shape)
            
            for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
                mask = Y[lm:rm] >= th
                if np.sum(mask) == 0:
                    pkX[idx]=np.nan
                else:
                    pkX[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
            dpkX = np.concatenate(([np.inf], np.diff(pkX), [np.inf]))
            
            j = np.where((dpkX[1:-1] <= oversegmentationfilter) & (dpkX[1:-1] <= dpkX[:-2]) & (dpkX[1:-1] < dpkX[2:]))[0]
            if j.size == 0:
                break
            left_min = np.delete(left_min, j + 1)
            right_min = np.delete(right_min, j)
            props["FWHML"] = np.delete(props["FWHML"], j + 1)
            props["FWHMR"] = np.delete(props["FWHMR"], j)
            
            val_max[j] = np.maximum(val_max[j], val_max[j + 1])
            val_max = np.delete(val_max, j + 1)
    else:
        peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
        pkX = np.empty(left_min.shape)
        
        for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
            mask = Y[lm:rm] >= th
            if np.sum(mask) == 0:
                pkX[idx]=np.nan
            else:
                pkX[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])

    signal_num = len(val_max)
    ## Area calculation
    if Calc_peak_area:
        props["Area"] = np.empty((signal_num,))
        for idx in range(signal_num):
            sl = slice(left_min[idx],right_min[idx]+1,1)
            if min(Y[sl])<0:
                props["Area"][idx] = np.trapezoid(Y[sl] - min(Y[sl]),X[sl])
            else:
                props["Area"][idx] = np.trapezoid(Y[sl],X[sl])
    if SNR_threshold:
        props["SNR"] = (val_max - props["Mean noise"])/props["Noise"]
        props["Noise"] = [props["Noise"]]*signal_num
        props["Mean noise"]= [props["Mean noise"]]*signal_num
    big_peak_bool = (np.array(right_min) - np.array(left_min))>=5
    left_min[big_peak_bool] += 1
    right_min[big_peak_bool] -= 1
    props["PextL"] = X[left_min] 
    props["PextR"] = X[right_min]

    return np.column_stack(([spectra_ind]*signal_num,pkX, val_max, *(props[key] for key in headers[3:])))
### Utility
def draw_data(
        data_sources: list,
        plot_mz_range = None,
        sample_spectra_idx = None,
        imzml_source = False,
        dataset_name = 'peaklists', 
        **kwargs):
    """
    Общее описание
    ----
    Функция для построения графиков интенсивностей и пик-листов из hdf5 в заданном диапазоне и в определённом спектре.

    :param data_obj_list: list of hdf5 objects
    :param plot_mz_range: list of min and max range to draw graphs
    :param num_specst: num of the spectrum to draw. Otherwise it will be choosed randomly
    
    :type data_obj_list: list of paths to hdf5
    :type plot_mz_range: `tuple` or `list`
    :type num_specst: `int`
    """
    randomized_spec = False
    diapcalc = lambda mz, plot_mz_range: (np.array(mz>plot_mz_range[0]) & np.array(mz<plot_mz_range[1])) if plot_mz_range is not None else range(len(mz))
    
    if not isinstance(data_sources, (list,tuple)):
        data_sources = [data_sources]
    for source in data_sources:
        if isinstance(source,str):
            data_obj = hdf5_Load(source)
        elif isinstance(source, (dict,File)):
            data_obj = source

        for slide in data_obj.keys():
            if isinstance(data_obj[slide],File):
                Raw_bool = data_obj[slide].filename.endswith('_rawdata.hdf5') #Deprecated
            else:
                Raw_bool = False
            for sample in data_obj[slide].keys():
                for roi in data_obj[slide][sample].keys():
                    if sample_spectra_idx is None or randomized_spec:
                        randomized_spec = True
                        sample_spectra_idx=np.random.randint(0,data_obj[slide][sample][roi]['xy'].len())
                    plt.figure().set_figwidth(25)
                    plt.gcf().set_figheight(5)
                    ### raw
                    
                    print(f'Spectrum number: {sample_spectra_idx}')
                    dtypeconv = kwargs.get("dtypeconv")
                    if Raw_bool:
                        mz_raw = data_obj[slide][sample][roi]["mz"][:]
                        intens_raw = data_obj[slide][sample][roi]["int"][sample_spectra_idx,:]
                        Label = ["Raw mass spectrum"]
                    else:
                        source = data_obj[slide][sample][roi].attrs['source']
                        if source.endswith('.imzML'):
                            sample_imzml = ImzMLParser(source)
                            idx_roi = data_obj[slide][sample][roi].attrs['idxroi']
                            idx_roi = range(idx_roi[0], idx_roi[0] + idx_roi[1])
                            
                            if  not (dtypeconv == "double" or isinstance(dtypeconv, np.float64)) and dtypeconv is not None:
                                mz_raw, intens_raw = (array.astype(dtypeconv) for array in sample_imzml.getspectrum(idx_roi[sample_spectra_idx]))
                            else:
                                mz_raw, intens_raw = sample_imzml.getspectrum(idx_roi[sample_spectra_idx])
                            Label = ["Raw mass spectrum from imzml file"]
                        elif source.endswith('*.mzXML'):
                            sample_spectra_file_idx = data_obj[slide][sample][roi].attrs['idxroi'][sample_spectra_idx]
                            with mzxml.MzXML(glob.glob(source)[sample_spectra_file_idx]) as sample_mzXML:
                                if dtypeconv:
                                    mz_raw = sample_mzXML[0]['m/z array'].astype(dtypeconv)
                                    intens_raw = sample_mzXML[0]['intensity array'].astype(dtypeconv)
                                else:
                                    mz_raw = sample_mzXML[0]['m/z array']
                                    intens_raw = sample_mzXML[0]['intensity array']
                            Label = ["Raw mass spectrum from mzXML file"]
                        else:
                            raise
                    if plot_mz_range is None:
                        plot_mz_range = [min(mz_raw),max(mz_raw)]
                    diap_raw = diapcalc(mz_raw, plot_mz_range)
                    plt.plot(mz_raw[diap_raw], intens_raw[diap_raw],alpha=0.75)

                    ### proccessed
                    mz = None
                    intens = None
                    if not Raw_bool and not imzml_source:
                        if rf"{sample}/{roi}/mz" in data_obj[slide]:
                            mz = data_obj[slide][sample][roi]["mz"][:]
                        if rf"{sample}/{roi}/int" in data_obj[slide]:    
                            intens = data_obj[slide][sample][roi]["int"][sample_spectra_idx,:]
                            Label.append("Processed mass spectrum from hdf5")
                    
                    if kwargs.get('DataProc_configs',False):
                        if kwargs.get("resample_to_dots",False) or kwargs.get("resampled_mz",False):
                            if rf"{sample}/{roi}/mz" not in data_obj[slide] or imzml_source:
                                mz = kwargs.get("resampled_mz", np.linspace(min(mz_raw),max(mz_raw), kwargs.get("resample_to_dots"), dtype=mz_raw.dtype))
                            if rf"{sample}/{roi}/int" not in data_obj[slide] or imzml_source:
                                intens_raw = np.interp(mz,mz_raw,intens_raw, left = intens_raw[0], right = intens_raw[-1]).astype(intens_raw.dtype)
                        else:
                            mz = mz_raw
                    if mz is not None:
                        if intens is None or imzml_source:
                            DataProc_configs = kwargs.get('DataProc_configs')
                            dots_distance = np.median(np.diff(mz))
                            DataProc_configs['smoothing_configs']["smooth_window"](dots_distance)
                            DataProc_configs["msalign_configs"]["shift_range"](dots_distance)
                            DataProc_configs["baseliner"](mz)
                            intens = DataProc_1d(intens_raw, mz,**DataProc_configs)
                        
                        diap = diapcalc(mz, plot_mz_range)
                        Label.append("Processed mass spectrum")
                        plt.plot(mz[diap], intens[diap],alpha=0.75)
                    
                    ### peaklists
                    DataFeat=None
                    if data_obj[slide].get(f"{sample}/{roi}/{dataset_name}", False):
                        DataFeat = pd.DataFrame(data_obj[slide][sample][roi][dataset_name][:].T, data_obj[slide][sample][roi][dataset_name].attrs["Column headers"]).T
                    else:
                        if 'peaks_configs' in kwargs:
                            if mz is None:
                                mz = mz_raw
                            if intens is None:
                                intens = intens_raw
                            PeakPicking_configs = kwargs["peaks_configs"]
                            peaklists = peaks_prop_infunc(mz, intens, np.where(np.diff(intens) != 0)[0], len(intens),sample_spectra_idx, **kwargs["peaks_configs"])
                            DataFeat = pd.DataFrame(peaklists.T, PeakPicking_configs['headers']).T
                            

                    if DataFeat is not None:
                        DataFeat = DataFeat.astype({"spectra_ind": int})
                        # print(DataFeat.shape)
                        DataFeat.query("mz>@plot_mz_range[0] and mz<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx").plot(x="mz",y="Intensity",ax = plt.gca(),style = "x", color = "k")
                        left_intens=[]
                        for left_base in DataFeat.query("PextL>@plot_mz_range[0] and PextL<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx")['PextL']:
                            left_intens.append(intens[mz>=left_base][0])
                        
                        right_intens = []
                        for right_base in DataFeat.query("PextR>@plot_mz_range[0] and PextR<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx")['PextR']:
                            right_intens.append(intens[mz<=right_base][-1])
                        plt.plot(DataFeat.query("PextL>@plot_mz_range[0] and PextL<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx")['PextL'],
                        left_intens,'v')
                        plt.plot(DataFeat.query("PextR>@plot_mz_range[0] and PextR<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx")['PextR'],
                        right_intens,'^')
                        if data_obj[slide].get(f"{sample}/{roi}/{dataset_name}", False):
                            Label=Label+["Peaks from dataset " + dataset_name, 'Left peak base','Right peak base']
                        else:
                            Label=Label+[f'Processed peaks for spectrum {sample_spectra_idx} (dataset {dataset_name} is missing)', 'Left peak base','Right peak base']
                            
                    plt.grid(visible=True,which="both")
                    plt.xlim(plot_mz_range)
                    plt.legend([*Label])
                    plt.minorticks_on()
                    plt.xlabel("m/z")
                    plt.ylabel("Intensity")
                    plt.title(f"Slide: {slide}, sample: {sample}, roi: {roi}, spectrum idx: {sample_spectra_idx}")
                    plt.show()
def audit_processing_quality(
        input_data_paths: list,
        plot_mz_range = None,
        sample_spectra_idx = None,
        config_path = None,
        dtypeconv = "double",
        **kwargs):
    """
    Общее описание
    ----
    Функция позволяет визуально оценить результат обработки спектров и пикпикинга для выбранных параметров по обработке одного спектра перед запуском обработки для всех спектров. Функция работает аналогично Raw2peaklist, но только обрабатывает один случайный спектр во всех sample и roi и после строит график для оценки.

    :param data_obj_path: list of paths to root folders where to search imzml files in subfolders 
    :param baseline_algo: Algorithm of baseline correction. Default: `"asls"`

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    :param params2baseline_correction: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html). Default: `{}`

        .. Example: {"lam" : 500000, "diff_order" : 1}
    :param align_peaks: list of reference peaks for align. Default: `None`
    :param weights_list: list of weights for reference peaks in aligning. Default: `None`
    :param max_shift_mz: max spectrum shift at aligning in mz. Default: `0.95`
    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`). Default: `{}`

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param smooth_algo: spectrum smoothing algorithm. Default is `"None"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay 
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `0`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `0`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `0`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `0`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `3.5`
    :param noise_est: алгоритм оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `"std"`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    :param smooth_window: window size in mz for smooth. Default:`0.075`
    :param smooth_cycles: Number of iterations for spectrum smooth. Default: `1`
    :param plot_mz_range: Range for graphs draw. Default: `None`
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`

    :type data_obj_path: `list`
    :type max_shift_mz: `float`
    :type resample_to_dots: `int`
    :type baseline_algo: `str`
    :type params2baseline_correction: `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`,`None`}
    :type oversegmentationfilter: `float`
    :type fwhhfilter: `float`
    :type heightfilter: `float`
    :type peaklocation: `float` and =<1
    :type rel_heightfilter: `float`
    :type SNR_threshold: `float`
    :type noise_est: {`"std"`,`"mad"`}
    :type noise_est_iterations: `int`
    :type smooth_window: `float`
    :type smooth_cycles: `int`
    :type plot_mz_range: `list` or `None`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `None`
    :rtype: `NoneType`
    """
    path_dict=find_imzml_roots(input_data_paths)
    configs = Configs([msalign,smoothing,peaks_prop_array,DataProc_array],config_path=config_path,**kwargs)
    for key in list(path_dict.keys()):
        slide = os.path.basename(key)
        path_dict[slide] = {}
        for path in path_dict[key]: 
            sample_name = os.path.splitext(os.path.basename(path))[0]
            folder_name = os.path.basename(os.path.dirname(path))
            if folder_name == sample_name:
                sample=sample_name
            else:
                sample = folder_name+"_"+sample_name
            path_dict[slide][sample] = {}
            path_dict[slide][sample]['No_roi'] = Sentinel()
            path_dict[slide][sample]['No_roi'].attrs = {}
            path_dict[slide][sample]['No_roi'].attrs['source'] = path
            path_dict[slide][sample]['No_roi'].attrs['idxroi'] = (0,len(ImzMLParser(path).mzLengths))
            path_dict[slide][sample]['No_roi'].attrs['dtype'] = dtypeconv
            path_dict[slide][sample]['No_roi'].attrs['N_resampled'] = configs.get("resample_to_dots",None) 
        path_dict.pop(key)   

    draw_data(path_dict, plot_mz_range=plot_mz_range, sample_spectra_idx=sample_spectra_idx, **configs)
    return

def source_search(hdf5_object):
    if isinstance(hdf5_object, str):
        hdf5_object = File(hdf5_object, 'r')
    hdf5_metadata = {}
    if isinstance(hdf5_object, h5py.Group):
        source = hdf5_object.attrs.get('source', False)
        if source:
            hdf5_metadata[source] = {}
            for attr_name, attr_value in hdf5_object.attrs.items():
                if not attr_name == 'source':
                    hdf5_metadata[source][attr_name] = attr_value
    for name, obj in hdf5_object.items():
        if isinstance(obj, h5py.Group):
            source = obj.attrs.get('source', False)
            if source:
                hdf5_metadata[source] = {}
                for attr_name, attr_value in obj.attrs.items():
                    if not attr_name == 'source':
                        hdf5_metadata[source][attr_name] = attr_value
            else:
                hdf5_metadata.update(source_search(obj))
    return hdf5_metadata

def add_procc_data(hdf5_object, 
                   func, 
                   configs_source = None, 
                   dataset_name = None, 
                   free_cores = 1, 
                   Ram_GB = 3, 
                   h5chunk_size_MB= 10, 
                   datasets_list = None, 
                   eval_align = False,
                   draw = True,
                   plot_mz_range = None,
                   sample_spectra_idx = None):
    if isinstance(hdf5_object, str):
        hdf5_save_path = hdf5_object
        hdf5_object = specdata_Load(hdf5_object)

    if isinstance(hdf5_object, dict):
        for slide in hdf5_object.keys():
            if isinstance(hdf5_object[slide],h5py._hl.files.File):
                add_procc_data(hdf5_object[slide], 
                               configs_source = configs_source, 
                               func = func, 
                               dataset_name = dataset_name, 
                               free_cores = free_cores, 
                               Ram_GB = Ram_GB, 
                               h5chunk_size_MB = h5chunk_size_MB,
                               datasets_list = datasets_list,
                               eval_align = eval_align,
                               draw = draw,
                               plot_mz_range = plot_mz_range,
                               sample_spectra_idx = sample_spectra_idx)
            else:
                add_procc_data(slide, 
                               configs_source = configs_source, 
                               func = func, 
                               dataset_name = dataset_name, 
                               free_cores = free_cores, 
                               Ram_GB = Ram_GB, 
                               h5chunk_size_MB = h5chunk_size_MB, 
                               datasets_list = hdf5_object[slide],
                               eval_align = eval_align,
                               draw = draw,
                               plot_mz_range = plot_mz_range,
                               sample_spectra_idx = sample_spectra_idx)
    else:
        if isinstance(hdf5_object, h5py.File):
            hdf5_save_path = hdf5_object.filename
        else:
            hdf5_save_path = hdf5_object.__filename__
        logger("add_procc_data_with_function_{func.name}",{**locals()})
        peakpicking = False
        manager = Manager()
        print_queue = Manager().Queue()
        queue = manager.Queue()
        queue.put(True)
        t = Thread(target=printer,args=[print_queue])  
        t.start()
        # Определение количества пула процессов
        cpu_num = cpu_count()-free_cores
        Ram_GB = Ram_GB*1e+9
        h5chunk_size_MB = h5chunk_size_MB*1e+6
        batch_bsize = Ram_GB/cpu_num

        if configs_source is None:
            configs_source = os.path.join(os.path.dirname(hdf5_save_path), 
                                    (os.path.basename(os.path.dirname(hdf5_save_path)) + "_proccesing_settings.yaml"))
             
        if isinstance(configs_source,str):
            if os.path.exists(configs_source):
                funcs_list = FUNCTIONS_FOR_SETTINGS.get(func.__name__, FUNCTIONS_FOR_PROCCESING)
                for f in funcs_list:
                    if f.__name__ == 'peaks_prop_array':
                        peakpicking = True
                configs = Configs(funcs_list, config_path = configs_source)
            else:
                raise FileExistsError(f"Configs file on path {configs_source} doesn't exist")
        elif isinstance(configs_source, Configs):
            configs = configs_source
            
        if dataset_name.endswith('_noaln'):
            eval_align = None
        # if datasets_list is not None:
        #     datasets_dict = {}
        #     for dataset in datasets_list:
        #         if len(dataset)>1:
        #             datasets_dict[dataset[0]] = dataset[1]
        #         else:
        #             datasets_dict.setdefault(dataset[0], None)
        args_batches = []
        for sample, roi, roi_idx, dtypeconv, source_path, dcont, resample_to_dots, mz_range in _get_local_metadata(hdf5_object, datasets_list = datasets_list): # (sample, roi, roi_idx, dtypeconv, source_path, dcont, resample_to_dots)
            if sample is None:
                continue
            # #Booling data for dataset 
            # if datasets_list is None:
            #     pass
            # elif sample in datasets_dict:
            #     if (roi in datasets_dict[sample]) or (datasets_dict[sample] is None) or (datasets_dict[sample] == [None]):
            #         pass
            #     else:
            #         continue
            # else:
            #     continue
            
            sample_imzml = ImzMLParser(source_path)
            bytes_flsize = BYTES_FLOAT_SIZE[dtypeconv]
            
            if peakpicking:
                data_columns_num = len(configs['peaks_configs']["headers"])
            else:
                if resample_to_dots:
                    data_columns_num = resample_to_dots
                elif dcont:
                    data_columns_num = sample_imzml.mzLengths[0]
                else:
                    logger.warn('') #TODO Дописать варнинг и как будет определяться chunk_size
            spectra_chunksize = max(1,np.ceil(batch_bsize/(bytes_flsize*data_columns_num)))
            hdf5_chunksize = max(1,np.ceil(h5chunk_size_MB/(bytes_flsize*data_columns_num)))
            specnum = len(sample_imzml.mzLengths)
            n_int = int(specnum/(spectra_chunksize)+1)
            if n_int<cpu_num*2:
                n_int = int(cpu_num*2)+1

            if resample_to_dots is not None:
                resampled_mz = np.linspace(*mz_range, resample_to_dots, dtype= dtypeconv)
                data_mz = resampled_mz
            else:
                resampled_mz = None
                data_mz = sample_imzml.getspectrum(roi_idx[0])[0].astype(dtypeconv)
            
            if (resample_to_dots or dcont) and configs.get("DataProc_configs", False):
                local_configs = configs.copy()
                local_configs['DataProc_configs'] = _set_local_proc_configs(local_configs['DataProc_configs'], data_mz)
            
            if len(local_configs.keys()) == 1:
                local_configs = local_configs[list(local_configs.keys())[0]]

            linspace_values = np.linspace(
                roi_idx[0],
                roi_idx[0] + roi_idx[1],
                n_int,
                dtype=int
                )
            pairwise_values = pairwise(linspace_values)
            args_batches.extend(list(args) for args in product(
                [source_path],
                [hdf5_save_path],
                [sample],
                [roi],
                pairwise_values,
                [resampled_mz],
                [print_queue],
                [local_configs],
                [queue],
                [hdf5_chunksize], 
                [dataset_name]
                )
                )
        hdf5_object.close()
        multiproc_processing(func, print_queue, cpu_num, args_batches, dataset_name, eval_align = eval_align)
        print_queue.put(Sentinel())
        t.join()
        logger.ended()
        if draw:
            # TODO: Выяснить почему может неправильно отрисовываться пики
            draw_data(hdf5_save_path, 
                      plot_mz_range = plot_mz_range,
                      sample_spectra_idx = sample_spectra_idx, 
                      imzml_source = True, 
                      dataset_name = dataset_name, 
                      **configs)
    gc.collect()
def _get_local_metadata(obj, datasets_list = None):
    for _, local_obj in obj.items():
        if isinstance(local_obj,(h5py.Group, h5py.File)):
            for sample, roi, roi_idx, dtypeconv, source_path, dcont, resample_to_dots, mz_range in _get_local_metadata(local_obj,datasets_list):
                yield sample, roi, roi_idx, dtypeconv, source_path, dcont, resample_to_dots, mz_range
        else:
            if isinstance(datasets_list, list):
                rois = []
                samples = {}
                for dataset in datasets_list:
                    if isinstance(dataset, (list,tuple)):
                        if len(dataset) > 1:
                            rois = dataset[1]
                        else:
                            rois = None
                        samples[dataset[0]] =  rois
            _, sample, roi = obj.name.split("/")
            
            if (sample in samples) or (datasets_list is None):

                if (roi in samples[sample]) or (samples[sample] is None):
                    roi_idx = obj.attrs['idxroi']
                    dtypeconv = obj.attrs['dtype']
                    source_path = obj.attrs['source']
                    dcont = obj.attrs['continuous']
                    resample_to_dots = obj.attrs.get('N_resampled', None)
                    mz_range = obj.attrs['mz_range']
                    yield sample, roi, roi_idx, dtypeconv, source_path, dcont, resample_to_dots, mz_range
                    break
            
            yield None, None, None, None, None, None, None, None
            break     
                
       
                    
    
def _find_dots_process(specdata_sources, save_results = True, **kwargs):
    # TODO: Если файл yaml для peaklists_noaln совпадает с нынешними настройками без выравнивания, то делается варнинг и получается уже повторная обработка не происходит (УЧЕСТЬ И DTYPECONV!).
    # TODO: Если удаляется peaklists_noaln - удаляется и файл с настройками
    """
    Run the main data processing pipeline.

    The pipeline reads raw and aligned spectra from HDF5, computes KDEs,
    performs peak picking, aligns peak lists, and computes
    descriptive and inferential statistics. Results are emitted via the
    `result` signal as a tuple of render instructions and statistics.

    Notes
    -----
    Emits
        - ``create_pbar``: tuple of (min, max) for a progress bar.
        - ``progress``: updates during dataset iteration.
        - ``result``: composite payload for UI updates.
        - ``finished``: upon completion or on handled exception.
        - ``error``: formatted traceback on exception.
    """
    # Использование:
    # 1) После основной обработки данных, где сразу происходит абсолютно та же обработка данных через Raw2peaklist, но без выравнивания.
    # Проблема использования: Необходима запись нового массива данных в HDF5 или написание новой/рефакторинг функции обработки или освоение декораторов.
    # 2) Отдельно после обработки. По сути тоже, что и первое, но параметры обработки берутся строго из yaml файла.

    ### Data_loading
    logger("_find_dots_process",{**locals()})
    grouped_images_DF=pd.DataFrame()
    Coords = pd.DataFrame(columns=['x','y'], dtype = float)

    if isinstance(specdata_sources,list):
        source_list=specdata_sources
        for source in source_list:
            Slide_data = specdata_Load([source])
            for slide in Slide_data.keys():
                _find_dots_process(Slide_data[slide], save_results = save_results, **kwargs)
        source_list = []

    elif isinstance(specdata_sources, dict): #TODO: Добавить поддержку вносимых hdf5 файлов
        source_keys=list(specdata_sources.keys())
        if isinstance(specdata_sources[source_keys[0]], (h5py._hl.files.File,h5py.File, h5py.Group, h5py.Dataset)):
            # new_specdata_sources = {}
            # sample_dict = {}
            for source in source_keys:
                _find_dots_process(specdata_sources[source], save_results = save_results, **kwargs)
            source_list = []
        else:
            source_list = specdata_sources.keys()
            samples = True
    elif isinstance(specdata_sources, (h5py._hl.files.File,h5py.File, h5py.Group, h5py.Dataset)):
        new_specdata_sources = {}
        sample_dict = {}

        if isinstance(specdata_sources, (h5py._hl.files.File,h5py.File)):
            path_to_file = specdata_sources.filename
        else:
            path_to_file = specdata_sources.__filename__
        
        sample_dict = new_specdata_sources.setdefault(path_to_file, {})
        names = specdata_sources.name.split("/")
        if names == [''] or names == ['','']:
            new_specdata_sources[path_to_file] = None
        elif len(names)>1:
            sample_dict.setdefault(names[0], [])
            if len(names)>2:
                sample_dict[names[0]].append(names[1])
            else:
                sample_dict[names[0]].append(None)
        else: 
            new_specdata_sources[path_to_file] = None
        for source in new_specdata_sources.keys():
            if new_specdata_sources[source]:
                new_specdata_sources[source] = list(new_specdata_sources[source].items())
        specdata_sources = new_specdata_sources
        source_list = specdata_sources.keys()
        samples = True
    elif isinstance(specdata_sources,str):
        source_list = [specdata_sources]
        samples = False
    else:
        raise TypeError("specdata_source must be a string, list or a dictionary") 
    
    ## Поверка на наличие _noaln, составляем доработку
    datasets4_noaln = {}
    do_new_noaln = False
    for source in source_list:
        Slide_data = specdata_Load([source])
        ### samples to load
        datasets4_noaln.setdefault(source, {})
        for slide in list(Slide_data.keys()):
            if samples:
                s_iter = specdata_sources[source]
                if s_iter is None:
                    s_iter = Slide_data[slide].keys()
            else:
                s_iter = Slide_data[slide].keys()
            # TODO: написать функцию для получения списков итерируемых данных (то что используется в данном коде с s_iter)
            for sample in s_iter:
                datasets4_noaln[source].setdefault(sample, [])
                if len(sample)>1 and isinstance(sample,(tuple,list)):
                    rois = sample[1]
                    if rois is None:
                        rois = Slide_data[slide][sample].keys()
                    sample = sample[0]
                else:
                    rois = Slide_data[slide][sample].keys()
                for roi in rois:
                    # peaklists_aln =  Slide_data[slide][sample][roi]['peaklists']
                    # headers = Slide_data[slide][sample][roi].attrs['Column headers']
                    if f"{sample}/{roi}/peaklists_noaln" not in Slide_data[slide]: 
                        datasets4_noaln[source][sample].append(roi)
                        do_new_noaln = True
                        # peaklists_noaln = Slide_data[slide][sample][roi]['peaklists_noaln']

                    

    hdf5_close()
    if do_new_noaln:
        for source in datasets4_noaln.keys():
            datasets4_noaln[source] = list(datasets4_noaln[source].items())      
        add_procc_data(datasets4_noaln, int2proc2peaklist_parbatched, dataset_name = "peaklists_noaln", **kwargs)
    else:
        logger.log("No new peaklists_noaln datasets to process")
    
    for source in source_list:
        Slide_data = specdata_Load([source])
        # TODO: s_iter переписать в виде функции, чтобы не было дублирования
        for slide in list(Slide_data.keys()):
            path_to_hdf5 = Slide_data[slide].filename
            if samples:
                s_iter = specdata_sources[source]
                if s_iter is None:
                    s_iter = list(Slide_data[slide].keys())
            else:
                s_iter = list(Slide_data[slide].keys())
            Slide_data[slide].close()
            for sample in s_iter:
                with h5py.File(path_to_hdf5, 'r') as hdf5:
                    if len(sample)>1 and isinstance(sample,(tuple,list)):
                        rois = sample[1]
                        if rois is None:
                            rois = list(hdf5[sample].keys())
                        sample = sample[0]
                    else:
                        rois = list(hdf5[sample].keys())
                for roi in rois:
                    # TODO 1) Сделать автоматическим подбор Bandwidth, возможно лучше переписать код из pfeats сделав получение KDE общим по данным из hdf5
                    # TODO 1.2) Исправить ошибку, если bandwidth слишком большой (а может и маленький??). Так как возникают пустые листы в списке пиков и как следствие есть возникновение NaN чисел при mean функции, что приводит к ошибке
                    # TODO 2) Разобраться в необходимости ref и dev
                    # TODO 3) Сделать автоматическим подбор n_dots
                    # TODO 4) Убрать вопросы записи данных и куда сохранять метаданные
                    # TODO 5) СДелать возможность отрисовки графиков результатов
                    # TODO 6) Разобраться в применяемой статистике определения что всё гуд
                    # TODO 7) Разобраться с работой венгерского алгоритма в данной штуке и в его необходимости
                    calculate(path_to_hdf5,f"{sample}/{roi}/peaklists_noaln",f"{sample}/{roi}/peaklists", ref = 0, dev=0.15, bandwidth=0.025, n_dots=100000, save_results = save_results)
    
### Constants and base configs
BYTES_FLOAT_SIZE = {"single": 4, "double": 8, "half": 2}
FUNCTIONS_FOR_PROCCESING = [msalign,smoothing,DataProc_array,peaks_prop_array]

FUNCTIONS_FOR_SETTINGS= {"proc2peaklist" : FUNCTIONS_FOR_PROCCESING[-1:-2:-1],
                            "Raw2proc" : FUNCTIONS_FOR_PROCCESING[0:-1],
                            "Raw2peaklist" : FUNCTIONS_FOR_PROCCESING,
                            "int2proc2peaklist_parbatched" : FUNCTIONS_FOR_PROCCESING,
                            "int2procc_parbatched" : FUNCTIONS_FOR_PROCCESING[0:-1],
                            "proc2peaklist" : FUNCTIONS_FOR_PROCCESING[-1:-2:-1]
                            }