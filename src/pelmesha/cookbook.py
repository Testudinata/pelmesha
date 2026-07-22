import pandas as pd
import numpy as np
from pelmesha.dough import AdaptiveParameter, DatasetHeaders
from pelmesha.filling import hdf5_Load, specdata_Load, hdf5_close, create_file_path, del_datasets_hdf5, del_hdf5, repack_hdf5, hdf5_metadata, _hdf5_get_metadata, find_paths, logger, source_search
from itertools import product, zip_longest, batched
from threading import Thread
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter, medfilt
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
from pyteomics import mzxml
import yaml
from pybaselines import Baseline
from abc import ABC, abstractmethod
# TODO рефакторизация ужасно написанного класса Configs, особенно с костылём initialized:
# Основная цель класса Configs: упрощение настроек конфигов простым указанием лишь части параметров, которые хочется поменять по сравнению с базовыми настройками,
#  либо путём к уже готовому конфигу, автоматизация их валидации и подтягивании остальных параметров, упрощение пользования одновременно как nested словарём с разбитыми 
# по подгруппам/функциям параметров, так и как плоским супом сугубо из самих параметров для упрощения ручной их замены
# Фичи: автоматический сбор докстринга параметров функций
#  класс при инициализации, которого сначала подтягивает параметры из указанного конфига, либо базового, если не указан путь, 
# а при передаче в начале через kwargs (или откуда созданные динамические параметры с описаниями)/или замене значений параметров впоследствии через __setitem__ производится повторная валидация параметра, 
# выделяет из них адаптивные параметры, распределяет параметры по функциям
# 0) Прописать докстринг для всех параметров в классе как на примере ниже, то есть заранее. Также у каждого параметра будет две "группы" 1-ая, обработка масс спектров или пикпикинг, а 2-ая к конкретно какой функции применять:
#:  а) from typing import TypedDict, Unpack

# class ConnectionParams(TypedDict, total=False):
#     db_name: str
#     """Имя целевой базы данных."""
    
#     timeout: int
#     """Максимальное время ожидания."""

# class DatabaseConnection:
#     def __init__(self, **kwargs: Unpack[ConnectionParams]):
#         pass
#   б) Или вариант с pydantic, есть преимущество - есть компьютед филд, который подойдёт для адаптивных параметров, но тогда нужны ещё поля "локальных параметров" (использовать PrivateAttr или SerializationInfo c контекстом, но кажется лучше последнее - меньше будет занимать ram) 
# , от которых и идёт рассчёт :
# from pydantic import BaseModel, Field

# class ParamDef(BaseModel):
#     db_name: str = Field("default_db", description="Имя целевой базы данных.")
#     timeout: int = Field(30, description="Максимальное время ожидания в секундах.")
# 0.1) Upd. Попытаться внедрить динамическое создание параметров из функций:
#   Видимо заранее создать всевозможные варианты? Типа, если ввёл один из аргументов, то он автоматически подтягивает именно эти?
#   Или просто проигнорировать вариант с pybaselines и просто вывести какие параметры для какого метода/класса коррекции имеется в целом с максимально кратким описание или без
#   ИЛИ!!! Попробовать испольовать кэшированные записи (нужно создать файл?). НО! Нужно учесть, что если что-то меняется в пакете (версия???), то должна быть пересборка 
# 1) Инициализация:
#   а) Создание класса инициализации базовых параметров из пидантик, где пидантик записаны из какой функции взяты параметры
# 2) ВХодные параметры:
#   а) Превращает плоский суп входных параметров и распределяет их в pydantic, заменяя базовые значения (вероятно, если использовать pydantic или dataclass, можно обойтись без плоского супа кажется, но с базовым хранилищем данных):
#   б) Если входной параметр путь к файлу yaml, то выгружает данные из него
# 3) Необходимо продумать чёткое разделение имён параметров, чтобы не пересекались
# 4) Разобраться есть ли возможность сделать параметры pydantic динамическими, в зависимости от функций в пайплайне и/или фиксированными. 
# 5) Сделать возможным, чтобы выдавались подсказки при вводе в функции/методе, в котором этот пидантик файл используется и/или подаётся целиком
# 6) Помимо обычного получения указанных значений параметров, есть дополнительная возможность вызова группы параметров, принадлежащих только к определённой группе/подгруппе 
# 7) Запись в YAML по указанному пути

class Configs_legacy(dict): 
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
def _set_local_proc_configs(DataProc_configs, data_mz): #TODO: сделать адаптивное окно??? со знанием локального dots_distance
    dots_distance = np.median(np.diff(data_mz))
    local_configs = copy.deepcopy(DataProc_configs)
    local_configs['smoothing_configs']['smooth_window'](dots_distance) 
    local_configs['msalign_configs']['shift_range'](dots_distance)
    local_configs['baseliner'](data_mz)
    return local_configs


    
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

class Pipeline():
    '''WIP
        Нужно сохранить все функции, которые будут использоваться в пайплайне
        Настроить для каждой функции конфиги: создать автоматизированный конфиг класс, который вытащит базовые значения функции и перепишет часть их значений, 
        которые пользователь хочет поменять (автоматически находит):
        1) Создание экземпляра: передача используемых функций. Создаётся класс конфига. Во время инициализации - для метода сет_конфиг и ран_процесс - компанует динамический докстринг из описаний функций и 
        дефолтных значений, чтобы по итогу при запуске или сете - можно было почитать подсказки по аргументам функций
        
        2) Set_конфиг: передача конфигов в виде плоского супа
        3) Run_процесс: Передача DataSource и обработка данных (возможно создать два метода в зависимости от типа данных - контиунальных или неконтиунальных)
        В конце сохраняет результат в hdf5, сохраняет конфиги
        '''
    def __init__(self, processing_functions_list, peakpicking_function):
        
        self.processing_functions_list = processing_functions_list
        self.peakpicking_function = peakpicking_function
        if peakpicking_function:
            total_functions = processing_functions_list.append(peakpicking_function)
        else:
            total_functions = processing_functions_list
        self.configs = Configs( total_functions )
        self._resolve_docs()
    
    def set_configs(self, configs):
        self.configs(configs)
        self.configs = configs
    @abstractmethod
    def run_process(self, data_source, configs = None, free_cpu = 1):
        '''
        WIP На данный момент каждый кастомный пайплайн должен иметь свой run_process - оставляю это как затравку на будущее
        Разбивает данные на батчи и по кускам в мультипроцессинге обрабатывает
        Каждый процесс выполняет пайплайн из data_source поэтапно:
        1) Выполняются функции из processing_functions_list
        2) Записывает промежуточный результат, если пользователь это просит
        3) Выполняется функция peakpicking_function, если функция задана
        4) Записывает результат, если он есть
        5) Сохраняет конфиги
        '''
        if configs:
            self.configs(configs)
        # TODO дописать универсальный базовый вариант
    def _resolve_docs(self):
        '''
        WIP
        Создает докстринг для set_configs и run_process по функциям из processing_functions_list и peakpicking_function
        '''
        pass
        
