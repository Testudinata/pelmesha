import os
import re
import sys
import numpy as np
from h5py import File
import h5py
import gc
import warnings
import logging
import pandas as pd 
import xarray as xr
from pyteomics import mzxml
from pyimzml.ImzMLParser import ImzMLParser
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import math
from multiprocessing import Pool, cpu_count
from abc import ABC, abstractmethod
from itertools import product, pairwise
from functools import cached_property, partial
import yaml
from pybaselines import Baseline
from pelmesha.dough import Indexator, SliceIndexator

# Иерархия структур в наименованиях и в HDF5: 
# 1.Slide - слайд или пара слайдов, на котором/ых находятся образцы (Sample) (= одному пайплану эксперимента (образцы->стекло->нанесение матрицы->измерение), это может быть корневой папкой, в которой сохраняются все измерения одного такого эксперимента) 
# 2.Sample - образец измерения, в котором может быть несколько изучаемых областей (roi: region of interest). (= одному измерению области/ей, которые пользователь сам выбрал как отдельные по каким-либо параметрам) 
# 3.ROI - изучаемые области в образце. Для Rapiflex'а в одном файле .imzml может быть несколько ROI, которые могут иметь и разные настройки и прочее, но сохранены пользователем в одном файле .imzml.
# В ряде случаев Sample = ROI с одним roi "00"

# Более кратко: Slide - корневая папка со всеми измерениями -> Sample - сами измерения записанные в один файл slide, сохранённый в папке Slide -> ROI - отдельные области измерения.

### Base
def hdf5_Load(path_list, file_end=''):
    """
    Общее описание
    ----
    Базовый загрузчик hdf5
    
    :param path_list: list of str with paths to `hdf5` file
    :param file_end: Поиск файлов с определённым окончанием в названии помимо ".hdf5"
     
    :type path_list: list
    :type file_end: str

    :return: dictionary with hdf5 file objects
    :rtype: dict
    """
    logger("hdf5_Load",{**locals()})
    file_end=file_end+".hdf5"
    if isinstance(path_list, str):
        path_list=[path_list]
        
    hdf5path_list = find_paths(path_list,file_end=file_end)
    
    Slide_data={}
    for path in hdf5path_list:
        Slide_name=os.path.basename(path.replace(file_end,""))
        Slide_data[Slide_name] = File(path,"r")
    if not hdf5path_list:
        logger.warn(f"Data not readed due to missing hdf5 with spectra data (hdf5 with end \"{file_end}\" in the name is missing)")
    logger.ended()
    return Slide_data

def specdata_Load(path_list):
    """
    Функция открытия в режиме чтения `hdf5` данных обработанных спектров.
    """
    return hdf5_Load(path_list,file_end='_specdata')
def features_Load(path_list):
    """
    Функция открытия в режиме чтения `hdf5` данных пиклистов сгруппированных по mz.
    """
    return hdf5_Load(path_list,file_end='_features')
def rawdata_Load(path_list):
    """
    Функция открытия в режиме чтения `hdf5` данных сырых спектров.
    """
    return hdf5_Load(path_list,file_end='_rawdata')
def grouped_MSIdata_Load(path):
    """
    Функция открытия в режиме чтения `hdf5` данных пиклистов сгруппированных по mz по нескольким областям.
    :param path: direct path to hdf5 file ending by '_grouped_MSIdata.hdf5' 
    """
    gr_fdata = hdf5_Load(path,file_end='_grouped_MSIdata')
    group_list = list(gr_fdata.keys())
    return gr_fdata[group_list[0]]


def peakl2DF(batch_path, extr_columns=None, extract_coords = True, return_source_path = False, pivoting4val = None, peaks_dataset_name = "peaklists"):
    """
    Общее описание
    ----
    Функция преобразует данные пиклисты `hdf5` в словарь с датафреймами пиклистов образцов согласно выставленным параметрам.

    :param batch_path: лист путей или путь к папке/файлу с `hdf5`. 
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где `"spectra_ind"` и `"mz"` или `"Peak"` экстрагируются всегда. Default: `None` - экстракция всех столбцов
    `"Intensity"`, `"Area"`, `"SNR"`, `"PextL"`, `"PextR"`, `"FWHML"`, `"FWHMR"`, `"Noise"`, `"Mean noise"`
    :param extract_coords: `True` - extracting to dict coordinates Dataframe, `False` - coordinates doesn't extracting. Default: `True`
    :param pivoting4val: list of columns or None (default) - extracted data is pivoted by index: spectra_ind, columns: Peak with fill_value = 0, and values: list of columns from pivoting4val. If None - do nothing about pivoting
    :param return_source_path: If `True` - return full path to source. Optional. Used in some functions.
    
    :type batch_path: `str` or `list`
    :type extr_columns: `list`
    :type extract_coords: `bool`
    :type pivoting4val: `list`
    :type return_source_path: `bool`

    :return: dictionary with peaklist and coordinates dataframes. `dict` structure as in `hdf5`. Return `tuple` if return_source_path is `True` with additional source pathes

    :rtype: `dict` or `tuple`
    """
    
    logger("peakl2DF",{**locals()})

    if isinstance(batch_path, str):
        batch_path=[batch_path]

    ### hdf5 load
    Slide_data = specdata_Load(batch_path)
    if return_source_path:
        table, source_path = table2DF(Slide_data, peaks_dataset_name, extr_columns, extract_coords, return_source_path, pivoting4val)
        logger.ended()
        return table, source_path
    else:
        logger.ended()
        return table2DF(Slide_data, peaks_dataset_name, extr_columns, extract_coords, return_source_path, pivoting4val) 

def feat2DF(batch_path, extr_columns=None,extract_coords = True, return_source_path = False, pivoting4val = None):
    """
    Общее описание
    ----
    Функция преобразует данные фича-матрицы `hdf5` в словарь с датафреймами пиклистов образцов согласно выставленным параметрам.

    :param batch_path: лист путей или путь к папке/файлу с `hdf5`. 
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где экстрагируются всегда `"spectra_ind"` и `"mz"` или `"Peak"`. Default: `None` - экстракция всех столбцов
    `"Intensity"`,`"Area"`,`"SNR"`,`"PextL"`,`"PextR"`,`"FWHML"`,`"FWHMR"`,`"Noise"`,`"Mean noise"`
    :param extract_coords: `True` - extracting to dict coordinates Dataframe, `False` - coordinates doesn't extracting. Default: `True`
    :param pivoting4val: list of columns or None (default) - extracted data is pivoted by index: spectra_ind, columns: Peak with fill_value = 0, and values: list of columns from pivoting4val. If None - do nothing about pivoting
    :param return_source_path: If `True` - return full path to source. Optional. Used in some functions.
    
    :type batch_path: `str` or `list`
    :type extr_columns: `list`
    :type extract_coords: `bool`
    :type pivoting4val: `list`
    :type return_source_path: `bool`

    :return: dictionary with peaklist and coordinates dataframes. `dict` structure as in `hdf5`. Return `tuple` if return_source_path is `True` with additional source pathes

    :rtype: `dict` or `tuple`
    """
    logger("feat2DF",{**locals()})
    
    if isinstance(batch_path, str):
        batch_path=[batch_path]

    ### hdf5 load
    feat_type = "features"
    Slide_data = features_Load(batch_path)
    if return_source_path:
        table,sourcse_path = table2DF(Slide_data,feat_type,extr_columns,extract_coords, return_source_path, pivoting4val)
        logger.ended()
        return table,sourcse_path
    else:
        logger.ended()
        return table2DF(Slide_data,feat_type,extr_columns,extract_coords, return_source_path, pivoting4val) 
    
def table2DF(Slide_data, dataset_name, extr_columns=None,extract_coords = True, return_source_path = False, pivoting4val = None, close = True):
    """
    Convert HDF5 data tables into pandas DataFrames.

    Recursively processes HDF5 file(s) to extract peaklist/feature data and
    optional spatial coordinates into a structured dictionary of DataFrames.

    :param Slide_data: HDF5 file object(s) — either a single h5py.File or a dict of them.
    :param dataset_name: Name of the dataset inside each ROI group (e.g. ``"peaklists"``, ``"features"``).
    :param extr_columns: Columns to extract. ``"spectra_ind"`` and ``"mz"``/``"Peak"`` are always included.
        Default ``None`` extracts all available columns.
    :param extract_coords: If ``True``, extract ``xy`` and ``z`` coordinate datasets. Default ``True``.
    :param return_source_path: If ``True``, return a tuple ``(DataFeat, Source_path)`` with source file paths.
    :param pivoting4val: If a list of column names, pivot the result table with ``index=spectra_ind``,
        ``columns=Peak``, ``fill_value=0`` and the given list as values. Default ``None`` (no pivoting).
    :param close: If ``True`` and ``Slide_data`` is a single file, close it after extraction. Default ``True``.

    :type Slide_data: dict or h5py.File
    :type dataset_name: str
    :type extr_columns: list or None
    :type extract_coords: bool
    :type return_source_path: bool
    :type pivoting4val: list or None
    :type close: bool

    :return: Nested dict ``{slide: {sample: {roi: {dataset_name: DataFrame, "xy": DataFrame, ...}}}}``.
        If ``return_source_path`` is ``True``, returns ``(DataFeat, Source_path)``.
    :rtype: dict or tuple
    """
    logger("table2DF",{**locals()})
    if isinstance(extr_columns, str):
        extr_columns=[extr_columns]
    DataFeat ={}
    Source_path={}
    if isinstance(Slide_data, dict):
        for slides in list(Slide_data.keys()):
            Source_path[slides] = Slide_data[slides].filename
            DataFeat[slides]= table2DF(Slide_data[slides], dataset_name, extr_columns = extr_columns, extract_coords = extract_coords, return_source_path = False, pivoting4val = pivoting4val, close = False)
            if not DataFeat[slides]:
                DataFeat.pop(slides,None)
    else:
        Source_path = Slide_data.filename
        slides = Source_path.split("\\")[-2]
        for sample in Slide_data.keys():
            DataFeat[sample] = {}
            for roi in Slide_data[sample].keys():

                try:
                    headers = list(Slide_data[sample][roi][dataset_name].attrs['Column headers'])
                except KeyError as error:
                    logger.warn(f"An error occurred on {slides} {sample} {roi}: {error}")
                    try:
                        logger.warn(f'HDF5 file from source {Source_path} on {slides} {sample} {roi} has datasets: {Slide_data[sample][roi].keys()}. And atributes: {Slide_data[sample][roi].attrs.keys()}')
                    except:
                        pass
                    continue

                DataFeat[sample][roi]={}
                ## Setting columns to extract
                if "Peak" in headers:
                    mz_type = "Peak"
                else:
                    mz_type = "mz"
                if extr_columns is None:
                    extr_columns = headers
                    column_nums = range(len(extr_columns))
                else:
                    column_nums=[]

                    if "mz" in extr_columns:
                        del extr_columns[extr_columns.index("mz")]
                    if "Peak" in extr_columns:
                        del extr_columns[extr_columns.index("Peak")]
                    if "spectra_ind" not in extr_columns:
                        extr_columns.append("spectra_ind")
                    temp_extr_column = extr_columns.copy()
                    temp_extr_column.append(mz_type)

                    for head in headers:
                        if head in temp_extr_column:
                            column_nums.append(headers.index(head))
                            del temp_extr_column[temp_extr_column.index(head)]
                    if temp_extr_column:
                        logger.warn(f"Columns: {temp_extr_column} - are not extracted. Columns in loading dataset is {headers}")
                ## Setting columns to extract - Ended

                if Slide_data[sample][roi][dataset_name].shape[1] == len(headers):

                    DataFeat[sample][roi][dataset_name] = pd.DataFrame(Slide_data[sample][roi][dataset_name][:,column_nums], columns= (headers[column_num] for column_num in column_nums)).sort_values(['spectra_ind',mz_type])
                else:
                    DataFeat[sample][roi][dataset_name] = pd.DataFrame(Slide_data[sample][roi][dataset_name][column_nums,:].T, columns= (headers[column_num] for column_num in column_nums)).sort_values(['spectra_ind',mz_type])
                try:
                    DataFeat[sample][roi][dataset_name] = DataFeat[sample][roi][dataset_name].astype({"spectra_ind": int})
                except:
                    logger.warn("spectra_ind to int is unsuccessful")
                    pass
                if extract_coords:
                    try:
                        #print(Slide_data[slides][sample][roi]['xy'][:])
                        DataFeat[sample][roi]["xy"] = pd.DataFrame(Slide_data[sample][roi]['xy'][:],columns=["x","y"], index=pd.Index(range(Slide_data[sample][roi]['xy'].shape[0]),name="spectra_ind"))
                        
                        logger.info(f"{slides}, {sample} and roi {roi}. x and y coordinates were extracted")
                        DataFeat[sample][roi]["z"] = pd.Series(Slide_data[sample][roi]['z'][:],columns=['z'], index=pd.Index(range(Slide_data[sample][roi]['z'].shape[0]),name="spectra_ind"))
                        logger.info(f"{slides}, {sample} and roi {roi}. z coordinates were extracted")
                    except:
                        pass#print(f"{slides}, {sample} and roi {roi}. The extraction of other coordinates was unsuccessful")
                if pivoting4val:
                    DataFeat[sample][roi][dataset_name] = DataFeat[sample][roi][dataset_name].pivot_table(index="spectra_ind", columns="Peak",fill_value = 0, values =pivoting4val)
            if not DataFeat[sample]:
                DataFeat.pop(sample,None)
        ## adding to DataFrame metadata
        for sample, roi, metadata in _hdf5_get_metadata(Slide_data):
            if sample is not None:
                DataFeat[sample][roi][dataset_name].attrs.update(metadata)
                if 'mean_spectrum' in Slide_data[sample][roi]:
                    DataFeat[sample][roi][dataset_name].attrs['mean_spectrum'] = Slide_data[sample][roi]['mean_spectrum'][:]

    if close and not isinstance(Slide_data, dict):
        Slide_data.close()
    if not DataFeat:
        warnings.warn(f"Warning. Any dataset doesn't have {dataset_name} data")
        logger.ended()
        return
    if return_source_path:
        logger.ended()
        return DataFeat, Source_path
    logger.ended()
    return DataFeat

def grouped_feat2DF(path, extr_columns=None,extract_coords = True, pivoting4val = None):
    """
    Общее описание
    ----
    Функция для работы со сгруппированными между имаджами пиклистами (признаками). Преобразует данные из `hdf5` в датафрейм.

    :param batch_path: лист путей или путь к папке/файлу с `hdf5`. 
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где 0 и 1 - экстрагируются всегда (`"spectra_ind"` и `"mz"` или `"Peak"`). Default: `None` - экстракция всех столбцов
    2 - `"Intensity"`, 3 -`"Area"`, 4 - `"SNR"`, 5 - `"PextL"`, 6 - `"PextR"`, 7 - `"FWHML"`, 8 - `"FWHMR"`, 9-`"Noise"`, 10-`"Mean noise"`
    :param extract_coords: `True` - extracting to dict coordinates Dataframe, `False` - coordinates doesn't extracting. Default: `True`
    :param pivoting4val: list of columns or None (default) - extracted data is pivoted by index: spectra_ind, columns: Peak with fill_value = 0, and values: list of columns from pivoting4val. If None - do nothing about pivoting
    
    :type batch_path: `str` or `list`
    :type extr_columns: `list`
    :type extract_coords: `bool`
    :type pivoting4val: `list`

    :return: dictionary with peaklist and coordinates dataframes. `dict` structure as in `hdf5`. Return `tuple` if return_source_path is `True` with additional source pathes

    :rtype: `dict` or `tuple`
    """
    logger("grouped_feat2DF",{**locals()},path)

    if isinstance(path, str):
        path=[path]
    
    grouped_images_DF=pd.DataFrame()
    Coords = pd.DataFrame(columns=['x','y'])

    Slide_data = grouped_MSIdata_Load(path)

    feat_type = "features"
    logger.log('Data converting to pandas DataFrame started')
    headers = Slide_data.attrs['Column headers']
    ####Probably delete this code. mz_type must be only "Peak"
    if "Peak" in headers:
        mz_type = "Peak"
    else:
        mz_type = "mz"
    ####
    for slides in list(Slide_data.keys()):
        for sample in Slide_data[slides].keys():
            for roi in Slide_data[slides][sample].keys():
                if extr_columns is None:
                    column_list = range(len(headers))
                    
                else:
                    column_list = extr_columns

                if Slide_data[slides][sample][roi][feat_type].shape[1] == len(headers):

                    DataFeat=pd.DataFrame(Slide_data[slides][sample][roi][feat_type], columns= headers).sort_values(['spectra_ind',mz_type])[Slide_data.attrs['Column headers'][column_list]]
                else:
                    DataFeat=pd.DataFrame(Slide_data[slides][sample][roi][feat_type][column_list,:].T, columns= headers).sort_values(['spectra_ind',mz_type])#[Slide_data[slides][sample][roi]['peaklists'].attrs['Column headers'][column_list]]
                try:
                    DataFeat=DataFeat.astype({"spectra_ind": int})
                except:
                    print("spectra_ind to int is unsuccessful")
                    pass
                ##concatenate
                n =DataFeat.shape[0]
                DataFeat.set_index([pd.Index([slides]*n),pd.Index([sample]*n),pd.Index([roi]*n)],inplace = True)

                if not set(DataFeat.columns).issubset(set(grouped_images_DF.columns)):
                    for col in list(DataFeat.columns):
                        if col not in set(grouped_images_DF.columns):
                            grouped_images_DF[col]=[]
                grouped_images_DF = pd.concat([DataFeat,grouped_images_DF])
                logger.log(f'Extracted data {slides}, {sample} roi {roi}. And concatenated')
                if extract_coords:
                    n = Slide_data[slides][sample][roi]['xy'].shape[0]
                    temp_coords = pd.DataFrame(Slide_data[slides][sample][roi]['xy'][:],columns=["x","y"]).set_index([pd.Index([slides]*n),pd.Index([sample]*n),pd.Index([roi]*n),pd.Index(range(n))])
                    logger.log(f"{slides}, {sample} and roi {roi}. x and y coordinates were extracted")
                    Coords = pd.concat([temp_coords,Coords])
    grouped_images_DF.index.names = ['slide','sample','roi']
    grouped_images_DF=grouped_images_DF.astype({'spectra_ind':int})
    logger.log('Data converting to pandas DataFrame ended')
    if pivoting4val:
        logger.log(f'Data pivoting started for values:{pivoting4val}')
        grouped_images_DF=grouped_images_DF.pivot_table(index=[grouped_images_DF.index,'spectra_ind'], columns="Peak",fill_value = 0, values =pivoting4val)
        grouped_images_DF.index.names = ['MS_image','spectra_ind']
        logger.log(f'Data pivoting ended')
    Slide_data.close()
    

    logger.ended()
    if extract_coords:
        Coords.index.names = ['slide','sample','roi',"spectra_ind"]
        return grouped_images_DF, Coords
    return grouped_images_DF

def IMGfeats_concat(paths,extr_columns,extracts_coords=True,processed_feat = False):
    """    
    Общее описание
    ----
    Функция объединяет данные пиклистов в разных `hdf5` в датафрейм пиклистов образцов согласно выставленным параметрам.

    :param paths: dict = {path_1: [[sample_1,[roi_list_1]], [sample_2,[roi_list_2]], ....],path_2:[[sample_3,[roi_list_3]],[sample_4,[roi_list_4]],....]}, "path" - path to hdf5 file directory, "sample_n" - какой именно sample (string), если None - берёт всё, "roi_list_n" - список каких roi использовать, если отсутствует, то берёт всё (example: dict value: list[sample_n])
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где экстрагируются всегда `"spectra_ind"` и `"mz"` или `"Peak"`. Default: `None` - экстракция всех столбцов
    `"Intensity"`,`"Area"`,`"SNR"`,`"PextL"`,`"PextR"`,`"FWHML"`,`"FWHMR"`,`"Noise"`,`"Mean noise"`
    :param processed_feat: `True` - Dataframe from grouped peaklists, `False` - Dataframe from raw image peaklists. Default: `False`
    :param extract_coords: `True` - extracting to dict coordinates Dataframe, `False` - coordinates doesn't extracting. Default: `True`

    :type batch_path: `str` or `list`
    :type extr_columns: `list`
    :type processed_feat: `bool`
    :type extract_coords: `bool`

    :return: peaklist DataFrame, where slide, sample and roi are in index. Return `tuple` if `extract_coords` is `True` with additional Coords

    :rtype: `dict` or `tuple`
    """
    
    ### Data_loading
    logger("IMGfeats_concat",locals())
    grouped_images_DF=pd.DataFrame() 
    Coords = pd.DataFrame(columns=['x','y'], dtype = float)
    if isinstance(paths,list):
        path_list=paths
        samples = False
        rois =  None
    elif isinstance(paths,dict):
        path_list=paths.keys()
        samples = True
    elif isinstance(paths,str):
        path_list = [paths]
        samples = False
    if isinstance(extr_columns, str):
        extr_columns=[extr_columns]
    for path in path_list:
       
        ### hdf5 load
        if processed_feat:
            feat_type = 'features'
            Slide_data = features_Load([path])
        else:
            feat_type = "peaklists"
            Slide_data = specdata_Load([path])
        ### slide load
        if not Slide_data:
            logger.warn(f'For path: {path}, data doesn\'t loaded')
        for slide in list(Slide_data.keys()):
            
            ### samples to load
            if samples:
                s_iter = paths[path]
                if s_iter is None:
                    s_iter = Slide_data[slide].keys()
            else:
                s_iter = Slide_data[slide].keys()

            for sample in s_iter:
                if len(sample)>1 and isinstance(sample,tuple):
                    rois = sample[1]
                    
                    sample = sample[0]
                else:
                    rois = Slide_data[slide][sample].keys()

                for roi in rois:                    
                    headers = Slide_data[slide][sample][roi][feat_type].attrs['Column headers']
                    if "Peak" in headers:
                        mz_type = "Peak"
                    else:
                        mz_type = "mz"
                    ## Setting columns to extract
                    if extr_columns is None:
                        column_nums = range(len(headers))
                    else:
                        
                        column_nums=[]


                        if "mz" in extr_columns:
                            del extr_columns[extr_columns.index("mz")]
                        if "Peak" in extr_columns:
                            del extr_columns[extr_columns.index("Peak")]
                        if "spectra_ind" not in extr_columns:
                            extr_columns.append("spectra_ind")
                        temp_extr_column = extr_columns.copy()
                        temp_extr_column.append(mz_type)
                        if not isinstance(headers,list):
                            headers = list(headers)
                        for head in headers:
                            if head in temp_extr_column:
                                column_nums.append(headers.index(head))
                                del temp_extr_column[temp_extr_column.index(head)]
                        if temp_extr_column:
                            logger.warn(f"Columns: {temp_extr_column} - are not extracted. Columns in loading dataset is {headers}")
                    ## Setting columns to extract - Ended
                    if Slide_data[slide][sample][roi][feat_type].shape[1] == len(headers):

                        #DataFeat[slide][sample][roi]=pd.DataFrame(Slide_data[slide][sample][roi][feat_type], columns= headers).sort_values(['spectra_ind',mz_type])[Slide_data[slide][sample][roi][feat_type].attrs['Column headers'][column_list]]
                        DataFeat=pd.DataFrame(Slide_data[slide][sample][roi][feat_type][:,column_nums], columns= (headers[column_num] for column_num in column_nums)).sort_values(['spectra_ind',mz_type])
                    else:
                        DataFeat=pd.DataFrame(Slide_data[slide][sample][roi][feat_type][column_nums,:].T, columns= (headers[column_num] for column_num in column_nums)).sort_values(['spectra_ind',mz_type])
                    
                    try:
                        #DataFeat[slide][sample][roi]=DataFeat[slide][sample][roi].astype({"spectra_ind": int})
                        DataFeat=DataFeat.astype({"spectra_ind": int})
                    except:
                        print("spectra_ind to int is unsuccessful")
                        pass
                    #n =DataFeat[slide][sample][roi].shape[0]
                    n = DataFeat.shape[0]
                    #DataFeat[slide][sample][roi].set_index([pd.Index([slide]*n,name='slide'),pd.Index([sample]*n,name='sample'),pd.Index([roi]*n,name='roi')],inplace = True)
                    DataFeat.set_index([pd.Index([slide]*n),pd.Index([sample]*n),pd.Index([roi]*n)],inplace = True)
                    
                    if not set(DataFeat.columns).issubset(set(grouped_images_DF.columns)):
                        for col in list(DataFeat.columns):
                            if col not in set(grouped_images_DF.columns):
                                grouped_images_DF[col]=[]
                    #grouped_images_DF = pd.concat([DataFeat[slide][sample][roi],grouped_images_DF])
                    grouped_images_DF = pd.concat([DataFeat,grouped_images_DF])
                    if extracts_coords:
#                        Coords[slide][sample][roi] = Slide_data[slide][sample][roi]['xy'][:]
                        n = Slide_data[slide][sample][roi]['xy'].shape[0]
                        temp_coords = pd.DataFrame(Slide_data[slide][sample][roi]['xy'][:],columns=["x","y"]).set_index([pd.Index([slide]*n),pd.Index([sample]*n),pd.Index([roi]*n),pd.Index(range(n))])
                        Coords = pd.concat([temp_coords,Coords])
                    #DataFeat[slides][sample][roi][feat_type]["xy"] = Slide_data[slides][sample][roi]['xy'][:]

            Slide_data[slide].close()
    grouped_images_DF.index.names = ['slide','sample','roi']

    if extracts_coords:
        Coords.index.names = ['slide','sample','roi','spectra_ind']
        logger.ended()
        return grouped_images_DF.astype({'spectra_ind':int}), Coords
    logger.ended()
    return grouped_images_DF

def hdf5_close():
    """
    Функция для закрытия всех hdf5 файлов разом
    """
    gc.collect()
    for obj in gc.get_objects():   # Browse through ALL objects
        try:
            if isinstance(obj, File):   # Just HDF5 files
                try:
                    obj.close()
                except:
                    pass # Was already closed
        except:
            pass

### utils functions
def find_paths(path_list,file_end = '.imzML'):
    """
    Общее описание
    ----
    Поисковик файлов
    
    :param path_list: list of str with paths to files
    :param file_end: Поиск файлов с определённым окончанием.
     
    :type path_list: list
    :type file_end: str

    :return: list with full paths to files
    :rtype: list
    """
    logger("find_paths",{**locals()})
    if isinstance(path_list,str):
        path_list=[path_list]
    file_end = file_end.lower()
    files_path_list = []
    for path in path_list:
        if path.lower().endswith(file_end) and os.path.exists(path):
            files_path_list.append(path)
        else:
            for root, dirs, files in os.walk(path):
                for file in files: 
                    if file.lower().endswith(file_end):
                        files_path_list.append(os.path.join(root,file))
    if not files_path_list:
        logger.warn(f'Files matching pattern {file_end} not found in specified paths: {path_list}')
    logger.ended()
    return files_path_list

def create_file_path(hdf5_save_folder, slide_name = None, hdf5_end = None):
    """
    Construct a full HDF5 file path from folder, slide name and file ending.

    :param hdf5_save_folder: Directory where the HDF5 file will be saved.
    :param slide_name: Slide name used as the filename stem. If ``None``, uses the basename of ``hdf5_save_folder``.
    :param hdf5_end: File ending (with or without ``.hdf5``). If ``None``, defaults to ``".hdf5"``.

    :type hdf5_save_folder: str
    :type slide_name: str or None
    :type hdf5_end: str or None

    :return: Full path to the HDF5 file.
    :rtype: str
    """
    if slide_name is None:
        slide_name = os.path.basename(hdf5_save_folder)
    if hdf5_end is None:
        hdf5_end = ".hdf5"
    elif not hdf5_end.endswith(".hdf5"):
        hdf5_end = hdf5_end+".hdf5"
    return os.path.join(hdf5_save_folder,slide_name) + hdf5_end

def del_hdf5(hdf5_path):
    """
    Delete an HDF5 file from disk if it exists.

    :param hdf5_path: Path to the HDF5 file to delete.
    :type hdf5_path: str
    """
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        print(f"Deleted file {os.path.basename(hdf5_path)} in directory {os.path.dirname(hdf5_path)}")
    

def del_datasets_hdf5(hdf5_obj, samples_path = None, dataset_to_del = None): 
    """
    Delete datasets or entire samples from an open HDF5 file.

    :param hdf5_obj: Open h5py.File object.
    :param samples_path: Source path(s) to match for deletion. If ``None``, matches all samples.
    :param dataset_to_del: Dataset name(s) to delete within matched samples.
        If ``None``, the entire sample group is deleted.

    :type hdf5_obj: h5py.File
    :type samples_path: str or list or None
    :type dataset_to_del: str or list or None

    :return: ``True`` if any data was deleted (requiring repacking), ``False`` otherwise.
    :rtype: bool
    """
    Need_repacking = False
    hdf5_name = os.path.basename(hdf5_obj.filename)
    if dataset_to_del is not None and not isinstance(dataset_to_del, list):
        dataset_to_del = [dataset_to_del]
    if not isinstance(samples_path, list):
        samples_path = [samples_path]
    for sample in hdf5_obj.keys():
        for sample_path in samples_path:
            for roi in hdf5_obj[sample].keys():        
                if (hdf5_obj[sample][roi].attrs["source"] == sample_path) or sample_path is None:
                    if dataset_to_del is None:
                        del hdf5_obj[rf'/{sample}']
                        Need_repacking = True
                        print(f"{hdf5_name}. Deleted sample {sample}")
                        break
                    else:
                        for dataset in dataset_to_del:
                            if hdf5_obj.get(rf'{sample}/{roi}/{dataset}', None):
                                del hdf5_obj[rf'{sample}/{roi}/{dataset}']
                                print(f"{hdf5_name}. Deleted dataset {dataset} from sample {sample} roi {roi}")
                                Need_repacking = True
    return Need_repacking

def repack_hdf5(hdf5_obj):
    """
    Repack an HDF5 file to reclaim space after deletions.

    Creates a temporary copy, removes the original, and renames the copy.

    :param hdf5_obj: Open h5py.File object or path string to the HDF5 file.
    :type hdf5_obj: h5py.File or str
    """
    if isinstance(hdf5_obj, str):
        hdf5_path = hdf5_obj
        hdf5_obj = File(hdf5_obj,"r")
    else:
        hdf5_path = hdf5_obj.filename
    print(f"Repacking {hdf5_path}")
    def _repacking_func(hdf5_old, hdf5_rep):
        for name, obj in hdf5_old.items():
            if isinstance(obj, h5py.Dataset):
                hdf5_rep.create_dataset(name, data=obj[:], chunks = obj.chunks ,dtype = obj.dtype)
                    # Копируем атрибуты датасета
                for attr_name, attr_value in obj.attrs.items():
                    hdf5_rep[name].attrs[attr_name] = attr_value
            elif isinstance(obj, h5py.Group):
                _repacking_func(obj, hdf5_rep.create_group(name)[obj.name])
                # Копируем атрибуты группы
                for attr_name, attr_value in obj.attrs.items():
                    hdf5_rep[name].attrs[attr_name] = attr_value
    hdf5_rep_path = hdf5_path.replace(".hdf5","_rep.hdf5")
    with File(hdf5_path,"r") as hdf5_old, File(hdf5_rep_path,"w") as hdf5_rep:
        _repacking_func(hdf5_old,hdf5_rep)
    hdf5_obj.close()
    os.remove(hdf5_path)
    os.rename(hdf5_rep_path,hdf5_path)

def hdf5_metadata(file_path, data_obj_metadata_donor, chunk_size = None):
    """
    Общее описание
    ----
    Вспомогательная функция для создания двух hdf5 файлов "[Slidename]_specdata.hdf5" и "[Slidename]_features.hdf5" и записи в них координат и часть данных таких как путь к первоисточнику, континуальность данных и записью принадлежности индекса спектра к определённому roi в sample.

    :param file_path: path to folder for writing `hdf5`.
    :param slide: параметр задающий Slidename в названии файла `hdf5`
    :param data_obj_coord: словарь схожий по структуре записи с будущими hdf5 и непосредственно из которого берутся все данные для записи.
    :param chunk_size: количество строк, на которые разделяется матрица в hdf5 файле
    
    :type file_path: `str`
    :type slide: `str`
    :type data_obj_coord: `dict`
    :type chunk_size: `int`

    :return: `None`
    :rtype: `NoneType`
    """ 
    with File(file_path,"a") as data_obj:
        for sample in data_obj_metadata_donor.keys():
            for roi in data_obj_metadata_donor[sample].keys():
                groups_path = rf"/{sample}/{roi}"
                if rf"{groups_path}/xy" not in data_obj:
                    try:
                        if isinstance(chunk_size, dict):
                            data_obj.create_dataset(rf"{groups_path}/xy",data=data_obj_metadata_donor[sample][roi]["xy"], chunks = (chunk_size[data_obj_metadata_donor[sample][roi]['source']],2))
                        elif chunk_size:
                            data_obj.create_dataset(rf"{groups_path}/xy",data=data_obj_metadata_donor[sample][roi]["xy"], chunks = (chunk_size,2))
                        else:
                            data_obj.create_dataset(rf"{groups_path}/xy",data=data_obj_metadata_donor[sample][roi]["xy"], chunks = True)
                    except ValueError:
                        data_obj.create_dataset(rf"{groups_path}/xy",data=data_obj_metadata_donor[sample][roi]["xy"])
                        
                if rf"{groups_path}/z" not in data_obj and "z" in data_obj_metadata_donor[sample][roi].keys():     
                        data_obj.create_dataset(rf"{groups_path}/z",data=data_obj_metadata_donor[sample][roi]["z"])
                if isinstance(data_obj_metadata_donor, File):
                    for attr_name, attr_value in data_obj_metadata_donor[sample][roi].attrs.items():
                        data_obj[sample][roi].attrs[attr_name] = attr_value
                else:
                    for key in ['xy','z','peaklists', 'features','int','mz']:
                        if key in data_obj_metadata_donor[sample][roi]:
                            del data_obj_metadata_donor[sample][roi][key]
                    if 'mean_spectrum' in data_obj_metadata_donor[sample][roi]:
                        data_obj.create_dataset(rf"{groups_path}/mean_spectrum",data=data_obj_metadata_donor[sample][roi]["mean_spectrum"])
                        del data_obj_metadata_donor[sample][roi]['mean_spectrum']
                    
                    for attr_name, attr_value in data_obj_metadata_donor[sample][roi].items():
                        data_obj[sample][roi].attrs[attr_name] = attr_value
class logger:
    """
    logging messages in local package format:

    `logger.warn(text)` - write text as warning message to log

    `logger.log(text)` - write text message to log

    `logger.ended()` - write message of successful end of function to log
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name=[]
    getted_log=logging.getLogger()
    for handler in getted_log.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.WARN)
    def __init__(self,func_name,args,path = None):        
        if not path:
            try:
                os.mkdir('logs')
            except:
                pass
            path_to_file_handler = os.path.join("logs",str(func_name))+"_log.log"
        else:
            path_to_file_handler = os.path.join(path,str(func_name))+"_log.log"
        
        h_not_exist = True
        for handler in self.getted_log.handlers:
            if isinstance(handler, logging.FileHandler):
                h_not_exist = False
                handler.setLevel(logging.INFO)
            # if isinstance(handler, logging.StreamHandler):
            #     handler.setLevel(logging.WARN)
        if h_not_exist:
            fhandler = logging.FileHandler(filename=path_to_file_handler, mode="w")
            fhandler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            fhandler.setLevel(logging.INFO)
            self.getted_log.addHandler(fhandler)

            
        self.name.append(func_name)
        self.getted_log.info(f"====================================Function {func_name} arguments========================================")
        for arg in args.keys():
            self.getted_log.info(f"{arg} = {args[arg]}")
        self.getted_log.info(f"====================================Function {func_name} STARTED========================================")
    def warn(text):
        logger.getted_log.warn(f"{text}")
    def log(text):
        logger.getted_log.info(f"{text}")
    def ended():
        logger.getted_log.info(f"====================================Function {logger.name[-1]} ENDED==========================================")
        if len(logger.name)<2:
            del logger.name[-1]

            for handler in logger.getted_log.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.getted_log.removeHandler(handler)
        else:
            del logger.name[-1]
def source_search(hdf5_object):
    """
    Recursively search an HDF5 file/group for ``source`` attributes and collect metadata.

    :param hdf5_object: Path to an HDF5 file or an open h5py File/Group object.
    :type hdf5_object: str or h5py.File or h5py.Group

    :return: Dict mapping each unique source path to its metadata attributes.
    :rtype: dict
    """
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

def _hdf5_get_metadata(obj, datasets_list = None):
    """
    Generator that yields ``(sample, roi, metadata_dict)`` tuples from an HDF5 hierarchy.

    :param obj: HDF5 file path or open h5py File/Group object.
    :param datasets_list: Optional list of sample names or ``(sample, [rois])`` tuples to filter by.
    :type obj: str or h5py.File or h5py.Group
    :type datasets_list: list or None

    :yields: ``(sample_name, roi_name, metadata_dict)`` tuples.
    :rtype: generator
    """
    local_read = False
    if isinstance(obj, str):
        local_read = True
        obj = File(obj, 'r')
    metadata = {}
    for _, local_obj in obj.items():
        if isinstance(local_obj,(h5py.Group, h5py.File)):
            for sample, roi, metadata in _hdf5_get_metadata(local_obj,datasets_list):
                yield sample, roi, metadata
        else:
            samples = {}
            if isinstance(datasets_list, list):
                rois = []
                for dataset in datasets_list:
                    if isinstance(dataset, (list,tuple)):
                        if len(dataset) > 1:
                            rois = dataset[1]
                        else:
                            rois = None
                        samples[dataset[0]] =  rois
            _, sample, roi = obj.name.split("/")
            
            if (sample in samples) or (datasets_list is None):
                metadata = {}
                if (roi in samples.get(sample, [])) or (samples.get(sample, None) is None):
                    for attr_name, attr_value in obj.attrs.items():
                        metadata[attr_name] = attr_value
                    yield sample, roi, metadata
                    break
            
            yield None, None, {}
            break

    if local_read:
        obj.close()


class DataSource: 
    """
    WIP
    Класс основного управления обработкой данных от ОДНОГО файла. Использует класс DataManager для получения источника данных с унифицированным интерфейсом подгрузки.
    СОдержит в себе метаданные источника данных. Сохранные метаданные и конфиги данных.
    Сохраняет всю обработку рядом с источником данных. (может даже для удобства в одной папке с файлом источника данных)
    Подаёдтся непосредственное обращение к источнику данных.
    WIP Отработать подгрузку континуальных и неконтинуальных данных.
    """

    def __init__(self, path, respec = False, RamGb_limit_usage = 2):
        """
        Инициализация источника данных.

        :param path: путь к файлу данных
        :param respec: If ``True``, force re-creation of metadata. Default ``False``.
        :param RamGb_limit_usage: RAM limit in GB for batch processing. Default ``2``.
        :param config: Processing config dict for this source. If ``None``,
            defaults to ``{}`` (will be resolved later by DataSet).
        :type path: str
        :type respec: bool
        :type RamGb_limit_usage: int or float
        :type config: dict or None
        """

        self.Ramcap = RamGb_limit_usage * (1024**3)
        self.file_path = path
        sample_name = os.path.splitext(os.path.basename(path))[0]
        folder_name = os.path.basename(os.path.dirname(path))
        if folder_name != sample_name:
            sample_name = folder_name + "_" + sample_name
        self.sample_name = sample_name
        self.loader = DataManager().get_loader(path)(path, respec = respec) # Composition pattern
        for attr_name in dir(self.loader):
            if not attr_name.startswith('_'): # Игнорируем приватные методы
                attr = getattr(self.loader, attr_name)
                if callable(attr):
                    setattr(self, attr_name, attr)


        dirpath = os.path.split(path)[0]
        processed_dirname = 'processed_pelmesha'
        configs_name = f'{sample_name}_processing_recipe'
        if os.path.exists(os.path.join(dirpath, processed_dirname, configs_name + '.yaml')):
            self.configs_path = os.path.join(dirpath, processed_dirname, configs_name + '.yaml')
        else:
            self.configs_path = None

        spectra_hdf5_name = sample_name + '_processed_spectra.hdf5'
        if os.path.exists(os.path.join(dirpath, processed_dirname, spectra_hdf5_name)):
            self.processed_spectra_path = os.path.join(dirpath, processed_dirname, spectra_hdf5_name)
        else:
            self.processed_spectra_path = None

        peaklist_hdf5_name = sample_name + '_peaklists.hdf5'
        if os.path.exists(os.path.join(dirpath, processed_dirname, peaklist_hdf5_name)):
            self.peaklist_path = os.path.join(dirpath, processed_dirname, peaklist_hdf5_name)
        else:
            self.peaklist_path = None
    
        # Выгрузка метаданных
        self.meta_file_path = os.path.join(os.path.dirname(self.file_path),'raw_pelmesha',sample_name + '_ingredients.hdf5')
        if not os.path.exists(self.meta_file_path):
            raise FileNotFoundError("Metadata file not found")
        with File(self.meta_file_path,"r") as hdf5:
            self.metadata = pd.DataFrame([dict(hdf5['metadata'].attrs.items())], index = [self.sample_name])
            columns_list = set()
            metadata_dict = {}
            for roi in hdf5['metadata']:
                columns_list.update(hdf5[f'metadata/{roi}'].attrs.keys())
                metadata_dict[roi] = dict(hdf5[f'metadata/{roi}'].attrs.items())
            self.roi_metadata = pd.DataFrame.from_dict(metadata_dict, orient='index', columns=list(columns_list))
    
    def __getattr__(self, name):
        return getattr(self.loader, name)
    # ------------------------------------------------------------------ #
    #  Representations (__repr__ / _repr_html_)                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _file_url(path):
        """Convert a local path to a ``file:///`` URL (opens in OS file manager)."""
        from urllib.parse import quote
        return "file:///" + quote(path.replace("\\", "/"), safe="/:")

    def _repr_html_(self):
        """
        HTML summary of the DataSource for Jupyter Notebook.

        Shows file info, data type, and a table of ROIs with index ranges
        and m/z ranges.  The file path is a clickable link to the parent
        folder in the OS file manager.

        Returns
        -------
        str
            A valid HTML block.
        """
        dir_url = self._file_url(os.path.dirname(self.file_path))
        dcont_str = "Yes" if self.dcont else "No"
        dtype_str = self.metadata.get("dtype_raw", ["—"]).iloc[0]

        html = [
            '<div style="font-family: sans-serif; font-size: 13px;">',
            # --- header line: file name + folder link ----------------- #
            f'  <b>DataSource:</b> {self._escape_html(self.sample_name)}',
            f'  &nbsp;—&nbsp;'
            f'<a href="{dir_url}" target="_blank" '
            f'style="color: #1a73e8; text-decoration: none; font-size: 12px;">'
            f'📁 {self._escape_html(os.path.dirname(self.file_path))}</a>',
            # --- info table ------------------------------------------- #
            '  <table style="border-collapse: collapse; margin-top: 6px; '
            'font-size: 13px;">',
            '    <thead>',
            '      <tr>',
            '        <th style="border: 1px solid #999; padding: 4px 8px; '
            'background-color: #4a4a4a; color: #fff; text-align: left;">Property</th>',
            '        <th style="border: 1px solid #999; padding: 4px 8px; '
            'background-color: #4a4a4a; color: #fff; text-align: left;">Value</th>',
            '      </tr>',
            '    </thead>',
            '    <tbody>',
            f'      <tr style="background-color: #f9f9f9;">'
            f'<td style="border: 1px solid #ccc; padding: 3px 8px;">Continuous</td>'
            f'<td style="border: 1px solid #ccc; padding: 3px 8px;">{dcont_str}</td></tr>',
            f'      <tr style="background-color: #ffffff;">'
            f'<td style="border: 1px solid #ccc; padding: 3px 8px;">Raw dtype</td>'
            f'<td style="border: 1px solid #ccc; padding: 3px 8px; font-family: monospace;">'
            f'{self._escape_html(dtype_str)}</td></tr>',
            '    </tbody>',
            '  </table>',
        ]

        # --- ROI table ------------------------------------------------ #
        if len(self.roi_metadata) > 0:
            html.append(
                '  <table style="border-collapse: collapse; margin-top: 8px; '
                'font-size: 13px;">'
            )
            html.append('    <thead>')
            html.append('      <tr>')
            for col in ["ROI", "Index range", "m/z range"]:
                html.append(
                    f'        <th style="border: 1px solid #999; padding: 4px 8px; '
                    f'background-color: #4a4a4a; color: #fff; text-align: left;">'
                    f'{col}</th>'
                )
            html.append('      </tr>')
            html.append('    </thead>')
            html.append('    <tbody>')

            for roi_idx, (roi_name, roi_row) in enumerate(self.roi_metadata.iterrows()):
                bg = "#f9f9f9" if roi_idx % 2 == 0 else "#ffffff"
                idxroi = roi_row.get("idxroi")
                if isinstance(idxroi, np.ndarray):
                    if idxroi.ndim == 2 and idxroi.shape[1] == 2:
                        ranges = ", ".join(f"[{s}, {e})" for s, e in idxroi)
                    else:
                        ranges = str(idxroi)
                else:
                    ranges = str(idxroi)

                mz_range = roi_row.get("mz_range")
                if isinstance(mz_range, (tuple, list, np.ndarray)) and len(mz_range) == 2:
                    mz_str = f"{mz_range[0]:.4f} – {mz_range[1]:.4f}"
                else:
                    mz_str = str(mz_range)

                html.append(
                    f'      <tr style="background-color: {bg};">'
                    f'<td style="border: 1px solid #ccc; padding: 3px 8px; '
                    f'font-family: monospace;">{self._escape_html(str(roi_name))}</td>'
                    f'<td style="border: 1px solid #ccc; padding: 3px 8px; '
                    f'font-family: monospace;">{self._escape_html(ranges)}</td>'
                    f'<td style="border: 1px solid #ccc; padding: 3px 8px; '
                    f'font-family: monospace;">{self._escape_html(mz_str)}</td>'
                    f'</tr>'
                )

            html.append('    </tbody>')
            html.append('  </table>')

        html.append('</div>')
        return "\n".join(html)

    def __repr__(self):
        """
        Text summary of the DataSource for the console.

        Shows file name, data type, raw dtype, and a table of ROIs with
        index ranges and m/z ranges.

        Returns
        -------
        str
        """
        dcont_str = "Yes" if self.loader.dcont else "No"
        dtype_str = self.metadata.get("dtype_raw", ["—"]).iloc[0]

        lines = [
            f"DataSource: {self.sample_name}",
            f"  File:      {self.file_path}",
            f"  Continuous: {dcont_str}",
            f"  Raw dtype: {dtype_str}",
        ]

        if len(self.roi_metadata) > 0:
            # Build a mini text table for ROIs
            roi_rows = []
            for roi_name, roi_row in self.roi_metadata.iterrows():
                idxroi = roi_row.get("idxroi")
                if isinstance(idxroi, np.ndarray):
                    if idxroi.ndim == 2 and idxroi.shape[1] == 2:
                        ranges = ", ".join(f"[{s}, {e})" for s, e in idxroi)
                    else:
                        ranges = str(idxroi)
                else:
                    ranges = str(idxroi)

                mz_range = roi_row.get("mz_range")
                if isinstance(mz_range, (tuple, list, np.ndarray)) and len(mz_range) == 2:
                    mz_str = f"{mz_range[0]:.4f} – {mz_range[1]:.4f}"
                else:
                    mz_str = str(mz_range)

                roi_rows.append((str(roi_name), ranges, mz_str))

            # Dynamic column widths
            col_widths = [max(len(r[i]) for r in roi_rows) for i in range(3)]
            col_widths[0] = max(col_widths[0], len("ROI"))
            col_widths[1] = max(col_widths[1], len("Index range"))
            col_widths[2] = max(col_widths[2], len("m/z range"))

            hdr = (
                f"  {'ROI'.ljust(col_widths[0])} | "
                f"{'Index range'.ljust(col_widths[1])} | "
                f"{'m/z range'.ljust(col_widths[2])}"
            )
            sep = "  " + "-" * (sum(col_widths) + 2 * len(col_widths) - 1)
            lines.append("")
            lines.append(hdr)
            lines.append(sep)
            for r in roi_rows:
                lines.append(
                    f"  {r[0].ljust(col_widths[0])} | "
                    f"{r[1].ljust(col_widths[1])} | "
                    f"{r[2].ljust(col_widths[2])}"
                )

        return "\n".join(lines)

    @staticmethod
    def _escape_html(text):
        """Escape HTML special characters."""
        table = {
            "&": "&",
            '"': '"',
            "'": "'",
            ">": ">",
            "<": "<",
        }
        return "".join(table.get(c, c) for c in str(text))

    def _normalize_indices(self, idxs):
        if idxs is None:
            idxs = np.concatenate(self.roi_metadata['idxroi'].to_numpy())
        elif isinstance(idxs, (Indexator, SliceIndexator)):
            idxs = idxs.view(np.ndarray)
        elif isinstance(idxs, str):
            idxs = self.roi_metadata.loc[idxs, 'idxroi'].to_numpy()
        elif isinstance(idxs,slice):
            idxs = np.array((idxs.start,idxs.stop))
        elif isinstance(idxs,int):
            idxs = np.array([idxs,idxs+1])
        elif isinstance(idxs,(list, tuple)):
            idxs = np.array(idxs)

        if idxs.ndim == 1:
            idxs = idxs[np.newaxis, :]
        return idxs

    def _get_local_roi_idx(self, idxs, roi_name = None):
        """
        Convert global spectrum indices to local indices within a ROI,
        accounting for gaps (discontinuous segments) in the ROI.

        For a ROI with segments ``[[0, 21158], [30000, 30100]]``,
        global index ``30000`` maps to local index ``21158``
        (the cumulative offset of the first segment).

        Parameters
        ----------
        idxs : int or np.ndarray
            Global index or array of index segments (shape ``(n, 2)``).
            Accepts :class:`Indexator`, :class:`SliceIndexator`, ``int``,
            ``list``, ``tuple``, or a raw :class:`np.ndarray`.
        roi_name : str, optional
            ROI name. If ``None``, the ROI is auto-detected from the
            index range of *idxs*.

        Returns
        -------
        int or np.ndarray
            Local index (if *idxs* was an ``int``) or array of local index
            segments with the same shape as the input.

        Raises
        ------
        ValueError
            If *roi_name* is ``None`` and the index does not belong to any ROI.
        KeyError
            If *roi_name* is not found in ``roi_metadata``.
        """
        # --- Resolve ROI segments -------------------------------------- #
        if roi_name is None:
            roi_name = self._get_roi(idxs)
        roi_segments = self.roi_metadata.loc[roi_name, 'idxroi']
        # roi_segments is an Indexator (ndarray subclass) with shape (M, 2)
        roi_segments = np.asarray(roi_segments, dtype=np.int64)
        if roi_segments.ndim == 1:
            roi_segments = roi_segments[np.newaxis, :]

        # --- Handle single int ----------------------------------------- #
        if isinstance(idxs, (int, np.integer)):
            # Find which ROI segment this index belongs to
            for seg_i, (start, end) in enumerate(roi_segments):
                if start <= idxs < end:
                    # Cumulative local start of this segment
                    local_start = np.sum(
                        np.diff(roi_segments[:seg_i], axis=1)
                    )
                    return int(idxs - (start - local_start))
            raise ValueError(
                f"Global index {idxs} does not belong to ROI '{roi_name}' "
                f"with segments {roi_segments.tolist()}"
            )

        # --- Normalise array input ------------------------------------- #
        idxs = self._normalize_indices(idxs)  # -> (N, 2)

        # --- Compute cumulative sizes and offsets per ROI segment ----- #
        sizes = np.diff(roi_segments, axis=1).ravel()                # (M,)
        cum_sizes = np.zeros(len(roi_segments), dtype=np.int64)
        cum_sizes[1:] = np.cumsum(sizes[:-1])                        # (M,) — local start of each segment
        # offset = global_start - local_start
        offsets = roi_segments[:, 0] - cum_sizes                     # (M,)

        # --- Map each batch segment to its ROI segment ---------------- #
        starts = idxs[:, 0, None]                                    # (N, 1)
        mask = (starts >= roi_segments[:, 0]) & \
               (starts <  roi_segments[:, 1])                        # (N, M)
        seg_idx = np.argmax(mask, axis=1)                            # (N,)

        # --- Subtract the corresponding offset ------------------------ #
        local_segments = idxs - offsets[seg_idx, None]

        return local_segments

    def get_coords(self, idxs = None, extract = None):
        idxs = self._normalize_indices(idxs)

        with File(self.meta_file_path,"r") as hdf5:
            coords = []
            if extract is None:
                extract = hdf5['coords'].keys()
            for idx in SliceIndexator(idxs):
                coords_slice = np.vstack((range(idx.start, idx.stop), *(hdf5[f'coords/{coord}'][idx] for coord in extract))).T
                coords.append(coords_slice)
            coords = pd.DataFrame(np.vstack(coords), columns = ['spectra_ind', *extract])
        coords['spectra_ind'] = coords['spectra_ind'].astype(np.int64)
        
        return coords.set_index('spectra_ind')

    def _get_roi(self, idx): 
        """
        WIP
        Метод возвращает первый roi, относящийся индексу или точь-в-точь тому полного сегмента/отрезка (пока не умеет определять нахождение в промежутке). 
        Функция сильно упрощённая для решения простой задачи - определить принадлежность к ПЕРВОМУ roi

        :param idx: Spectrum index or array of index segments.
        :type idx: int or np.ndarray or Indexator

        :return: ROI name (string) matching the given index.
        :rtype: str

        :raises ValueError: If no ROI matches the given index.
        """ # TODO Доработать для использования в объединённых датасетах (с мультиинексом, где ещё и sample)
        # in_bool = np.zeros(len(self.roi_metadata), dtype = bool) # TODO: Оставить до момента решения судьбы функции на вопрос её усложнения и возвращения нескольких roi
        
        for roi_num, indexes in enumerate(self.roi_metadata['idxroi'].values):
            if isinstance(idx, np.ndarray):
                for rng in idx:
                    range_bool = (indexes == rng).sum(axis = 1) == 2
                    if range_bool.any():
                        # in_bool[roi_num] = True
                        return self.roi_metadata.index[roi_num]
                for rng in idx:
                    if (rng[0]>=indexes[0,0]) & (rng[1]<=indexes[0,1]):
                        return self.roi_metadata.index[roi_num]
                # if range_bool.any(): 
                #     in_bool[roi_num] = True
                #     continue
            if isinstance(idx, int):
                if indexes.ndim ==1:
                    indexes = indexes[np.newaxis, :]
                if any(start <= idx < end for start, end in indexes):
                    return self.roi_metadata.index[roi_num]

        raise ValueError(f"ROI with index/indexes {idx} not found in {self.roi_metadata['idxroi'].values}")

    def get_mean_spectrum(self, idxs = None, mz_range = None):
        """
        Compute the mean (average) mass spectrum over the specified indices.

        Handles both continuous (TOF) and discontinuous (Orbitrap) data types,
        automatically detecting the data type and applying appropriate interpolation.

        :param idxs: Spectrum indices or ROI name. If ``None`` and only one ROI exists, uses that ROI.
        :param mz_range: Optional ``(mz_min, mz_max)`` tuple to restrict the m/z range.
        :type idxs: str or np.ndarray or None
        :type mz_range: tuple or None

        :return: Tuple ``(mz_scale, mean_intensity)`` where both are 1-D numpy arrays.
        :rtype: tuple
        """
        if isinstance(idxs, str):
            idxs = self.roi_metadata.loc[idxs, 'idxroi'] # TODO: Проверить работоспособность
        elif idxs is None:
            if len(self.roi_metadata) == 1:
                idxs = self.roi_metadata['idxroi'].iloc[0]
            else:
                raise ValueError(
                    f"Multiple regions (ROIs) found: {self.roi_metadata.index.tolist()}. In this case 'idxs' must be a ROI name (str) "
                    f"or an np.ndarray of continuous segments with shape (n, 2)"
                )
        if mz_range is not None:
            mz_min, mz_max = mz_range
        if self.dcont:
            d = 1
        else:
            d = 2

        if self.dcont:
            mz = self.mz_scale_cont

            if mz_range is not None:
                start_idx = np.searchsorted(mz, mz_min, side='left')
                stop_idx = np.searchsorted(mz, mz_max, side='right')
                mz_range_slice = slice(start_idx, stop_idx)
            else:
                mz_range_slice = slice(None)
            mz = mz[mz_range_slice]
            stats_sum = np.zeros(len(mz))
            stats_count = np.zeros(len(mz))
            indexes = Indexator(idxs)

            for intens in self.get_intensities_stream(indexes):
                stats_sum += intens[mz_range_slice]
            stats_count = indexes.count()
            # plt.plot(mz, stats_sum/stats_count)
            # plt.show()
            return mz, stats_sum/stats_count
        else:
            idxs_batches = self.split_idxs(idxs = idxs, cpu_count = 1, d = d)
            mz_range_slice = slice(None)
            pilot_batch = idxs_batches.pop()
            if np.diff(pilot_batch) < 5: # Проверка на то, что pilot_batch содержит мало масс спектров, на основе которых невозможно будет оценить надёжно орбитрепные это данные или нет. 5 спектров минимум - эмпирическое
                pilot_batch = np.array([idxs_batches.pop()[0], pilot_batch[1]])

            batch_mz = []
            batch_intens = []
            for mz, intens in self.get_batch(Indexator(pilot_batch)):
                if mz_range is not None:
                    start_idx = np.searchsorted(mz, mz_min, side='left')
                    stop_idx = np.searchsorted(mz, mz_max, side='right')
                    mz_range_slice = slice(start_idx, stop_idx)
                    mz = mz[mz_range_slice]
                    intens = intens[mz_range_slice]
                batch_mz.append(mz)
                batch_intens.append(intens)

            values, counts = np.unique(np.concatenate(batch_mz, axis=0), return_counts=True)
            if len(values[counts > 1])/len(values) < 0.33: # Если общих точек дискретизации m/z, которые совпадают хотя бы раз со всех масс спектров пробного батча, менее 33%, то считаем, что полученные данные с орбитрепа.
                roi = self._get_roi(pilot_batch)
                if mz_range is None:
                    mz_min, mz_max = self.roi_metadata['mz_range'][roi]
                discret_coeffs = self.roi_metadata.loc[roi, 'discret_coeffs']
                mz_discretion_model = np.poly1d(discret_coeffs)
                mz_scale = [mz_min]
                mz_end = mz_min
                while mz_end <= mz_max:
                    mz_end += mz_discretion_model(mz_end)
                    mz_scale.append(mz_end)
                mz_scale = np.array(mz_scale)
                stats_sum = np.zeros(len(mz_scale))
                stats_count = np.zeros(len(mz_scale))
                
                for mz_loc, intens_loc in zip(batch_mz, batch_intens):
                    if len(mz_loc) == 0:
                        continue
                    stats_sum += np.interp(mz_scale, mz_loc, intens_loc, left=0, right=0)
                    stats_count[np.unique(np.searchsorted(mz_scale, mz_loc, side='left'))] += 1
                del batch_mz, batch_intens

                for idxs_batch in idxs_batches:
                    for mz_loc, intens_loc in self.get_batch(Indexator(idxs_batch)):
                        if mz_range is not None:
                            start_idx = np.searchsorted(mz_loc, mz_min, side='left')
                            stop_idx = np.searchsorted(mz_loc, mz_max, side='right')
                            mz_range_slice = slice(start_idx, stop_idx)
                            mz_loc = mz_loc[mz_range_slice]
                            if len(mz_loc) == 0:
                                continue
                            intens_loc = intens_loc[mz_range_slice]
                        stats_sum += np.interp(mz_scale, mz_loc, intens_loc, left=0, right=0)
                        stats_count[np.unique(np.searchsorted(mz_scale, mz_loc, side='left'))] += 1
                nonzero_count_bool = stats_count != 0
                mean_intensity = np.zeros(len(mz_scale))
                mean_intensity[nonzero_count_bool] = stats_sum[nonzero_count_bool]/stats_count[nonzero_count_bool]
                return mz_scale, mean_intensity

            else:
                stats = pd.DataFrame(np.vstack([np.concatenate(batch_mz, axis=0), np.concatenate(batch_intens, axis=0)]).T, columns=['mz','intensities']).groupby('mz')['intensities'].agg(['sum', 'count'])
                batch_mz = []
                batch_intens = []
                for idxs_batch in idxs_batches:
                    for mz, intens in self.get_batch(Indexator(idxs_batch)):
                        if mz_range is not None:
                            start_idx = np.searchsorted(mz, mz_min, side='left')
                            stop_idx = np.searchsorted(mz, mz_max, side='right')
                            mz_range_slice = slice(start_idx, stop_idx)
                            mz = mz[mz_range_slice]
                            intens = intens[mz_range_slice]
                        batch_mz.append(mz)
                        batch_intens.append(intens)
                    stats = stats.add(pd.DataFrame(np.vstack([np.concatenate(batch_mz, axis=0), np.concatenate(batch_intens, axis=0)]).T, columns=['mz','intensities']).groupby('mz')['intensities'].agg(['sum', 'count']), fill_value=0)
                return stats.index, stats['sum']/stats['count']
            
    def split_idxs(self, idxs = None, d = None, cpu_count = 1, Ramcap_GB = None):
        """
        Split spectrum indices into batches based on RAM cap configs.

        :param idxs: Index segments to split. If ``None``, uses all available spectra.
        :param d: Data dimensionality factor (1 for continuous, 2 for discontinuous), обозначает количество используемых в батчах векторов. 
        Для континуальных данных d = 1, так как достаточно обрабатывать один вектор 'y' (интенсивность), а mz у всех общее и под него выделить память - единожды, 
        но для неконтинуальных данных необходимо выделить память для mz и y, так как для каждого спектра свой mz. Default ``None`` - функция выберет автоматически.
        :param cpu_count: Number of CPU cores to account for in RAM budgeting. Default ``1``.

        :type idxs: np.ndarray or None
        :type d: int
        :type cpu_count: int

        :return: List of index segment arrays, each fitting within the RAM budget.
        :rtype: list
        """
        spectrum_sizes = self.get_spectrum_sizes(idxs)
        idxs_batches = []
        remainder = []
        if d is None:
            if self.dcont:
                d = 1
            else:
                d = 2
        if Ramcap_GB is None:
            Ramcap = self.Ramcap 
        else:
            Ramcap = Ramcap_GB * (1024 ** 3)
        Ram_usage_per_batch = Ramcap/ (cpu_count * d)
        
        idxs = self._normalize_indices(idxs)

        length_per_batch = Ram_usage_per_batch // np.dtype(self.metadata['dtype_raw'].iloc[0]).itemsize
        if self.dcont:
            chunk_size = length_per_batch // spectrum_sizes[0]
            remainder_count = 0
            
            for idx in idxs:
                
                segment_size = np.diff(idx)
                if remainder:
                    if remainder_count+segment_size >= chunk_size:
                        idx_rem_stop = idx[0] + chunk_size - remainder_count
                        idxs_batches.append(np.vstack(remainder.append([idx[0], idx_rem_stop])))
                        idx = np.array([idx_rem_stop, idx[1]])
                        segment_size = remainder_count+segment_size - chunk_size
                        assert segment_size == np.diff(idx)
                        remainder_count = 0
                        remainder = []
                    else:
                        remainder.append(idx)
                        remainder_count = remainder_count + segment_size
                if segment_size >= chunk_size:
                    for idx_batch in pairwise(np.arange(idx[0], idx[1], chunk_size)):
                        idxs_batches.append(np.array(idx_batch))
                    if idx[1] != idx_batch[1]:
                        remainder.append([idx_batch[1], idx[1]])
                        remainder_count = remainder_count + idx[1] - idx_batch[1]
                else:
                    remainder.append(idx)
                    remainder_count = remainder_count + segment_size
            if remainder:
                idxs_batches.append(np.vstack(remainder))
                remainder = []
                remainder_count = 0
        else:
            length_usage = 0
            idxs = Indexator(idxs)
            idx_start = idxs[0][0]
            idx_last = idx_start - 1
            for n, idx in enumerate(idxs):
                if idx - idx_last > 1: # Учёт сегментного разрыва в индексации
                    remainder.append(np.array([idx_start, idx_last + 1]))
                    idx_start = idx
                    
                if length_usage + spectrum_sizes[n] >= length_per_batch:
                    if remainder:
                        idxs_batches.append(np.vstack(remainder))
                        remainder = []
                    else:   
                        idxs_batches.append(np.array([idx_start, idx + 1]))
                    idx_start = idx
                    length_usage = 0
                else:
                    length_usage += spectrum_sizes[n]
                idx_last = idx

            if length_usage != 0:
                if remainder:
                    idxs_batches.append(np.vstack(remainder))
                    remainder = []
                else:
                    idxs_batches.append(np.array([idx_start, idx + 1]))

        return idxs_batches
    @property
    def config_path(self):
        path = os.path.join(os.path.dirname(self.file_path), 'processed_pelmesha', 'processing_recipe.yaml')
        return path if os.path.exists(path) else None
    @property
    def dcont(self):
        """get dcont from loader"""
        return self.loader.dcont
    @property
    def close(self):
        """
        Close the data source.
        """
        self.loader.close()


class BaseLoader(ABC): #TODO @задачка: Базовый абстрактный класс
    """
    Базовый класс для всех подгрузчиков данных с абстрактными и общими методами.
    Ни один конкретный подгрузчик не запустится, если в нем не будут определены ряд абстрактных методов, индивидуальные для них.
    """
    def create_metafile(self, draw = True, respec = False, chunk_Mbsize = 10):
        """
        Create or update the metadata HDF5 file (``<sample_name>_ingredients.hdf5``) for the data source.

        Stores metadata attributes, ROI metadata, and coordinate data (xy, z or another) in a structured
        HDF5 file under the ``raw_pelmesha`` subdirectory.

        :param draw: If ``True``, display diagnostic plots during metadata extraction. Default ``True``.
        :param respec: If ``True``, force re-creation of the metadata file even if it exists. Default ``False``.
        :param chunk_Mbsize: Chunk size in MB for HDF5 dataset storage. Default ``10``.

        :type draw: bool
        :type respec: bool
        :type chunk_Mbsize: int
        """ 
        file_path = self.file_path                                             
        base_folder_path = os.path.dirname(file_path)
        meta_file_folder = os.path.join(base_folder_path,'raw_pelmesha')
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        folder_name = os.path.basename(os.path.dirname(file_path))
        if folder_name != sample_name:
            sample_name = folder_name + "_" + sample_name

        meta_file_name = sample_name +'_ingredients.hdf5'
        meta_file_path = os.path.join(base_folder_path,'raw_pelmesha', meta_file_name)
        meta_file_exist = os.path.exists(meta_file_path)
        if not os.path.exists(meta_file_folder):
            os.makedirs(meta_file_folder)
        if (not meta_file_exist) or respec:
            chunk_Mbsize = chunk_Mbsize * (1024**2)
            metadata, roi_metadata, coords = self.get_metadata(draw) # TODO @задачка: Создать в новом классе выгрузки данных из cdf метод get_metadata, чтобы сработал этот метод. 
                                                                        #             Отмечу: coords в cdf - это RT, а roi - это по планам каналы 0-3 (где разные масс анализаторы и моды) 

            del_hdf5(meta_file_path) # del hdf5 if it exists
            
            with h5py.File(meta_file_path, 'w') as f:
                f.create_group('metadata')
                for key, value in metadata.items():
                    f['metadata'].attrs[key] = value

                for roi in roi_metadata.keys():
                    f['metadata'].create_group(roi)
                    for key, value in roi_metadata[roi].items():
                        f[f'metadata/{roi}'].attrs[key] = value
                
                f.create_group('coords')
                for coord_key, data in coords.items():
                    val_size = data.shape
                    if len(val_size) == 1:
                        f.create_dataset(f"coords/{coord_key}", data=data, dtype=data.dtype, chunks = True)
                    else:
                        auto_row = int(chunk_Mbsize/(data.itemsize * val_size[1])) 
                        auto_row = val_size[0] if auto_row > val_size[0] else auto_row
                        f.create_dataset(f'coords/{coord_key}', data=data, dtype=data.dtype, chunks = (auto_row, val_size[1]))

    def get_mz_range(self, idxs): 
        """
        Determine the minimum and maximum m/z values across the specified spectra.

        :param idxs: Index segments to scan.
        :type idxs: np.ndarray

        :return: Tuple ``(mz_min, mz_max)``.
        :rtype: tuple
        """
        if self.dcont:
            mz = self.mz_scale_cont
            return (mz[0], mz[-1])
        
        min_mz = np.inf
        max_mz = -np.inf

        for mz in self.get_mz_stream(idxs):
            min_mz = min(min_mz, mz[0])
            max_mz = max(max_mz, mz[-1])
        return (min_mz, max_mz)
    
    def get_mz_discretion_coeffs(self, idxs, degree = 3, mz_range = None, draw = True):
        """
        Fit polynomial coefficients describing the m/z discretisation step size.

        Uses a least-squares regression on the median-filtered m/z differences.

        :param idxs: Index segments to use for fitting.
        :param degree: Polynomial degree for the fit. Default ``3``.
        :param mz_range: Optional ``(mz_min, mz_max)`` to restrict the fitting range.
        :param draw: If ``True``, display a diagnostic plot. Default ``True``.

        :type idxs: np.ndarray
        :type degree: int
        :type mz_range: tuple or None
        :type draw: bool

        :return: Polynomial coefficients (highest degree first).
        :rtype: np.ndarray
        """
        nsize_matrix = degree + 1
        sum_XTX = np.zeros((nsize_matrix, nsize_matrix))
        sum_XTy = np.zeros(nsize_matrix)
        if self.dcont: #Достаточно одного спектра
            start_idx = idxs[0][0]
            idxs = Indexator([start_idx, start_idx+1])
        if mz_range:
            mz_min, mz_max = mz_range
        else:
            mz_min, mz_max = self.get_mz_range(idxs)
        for mz in self.get_mz_stream(idxs):
            mz_vander = np.vander(mz[:-1], nsize_matrix)
            sum_XTX += mz_vander.T @ mz_vander
            sum_XTy += mz_vander.T @ medfilt(np.diff(mz),5)
        discret_coeffs = np.linalg.solve(sum_XTX, sum_XTy)

        #перепроверка на равномерность дискретизации шкалы
        mz_discret = np.poly1d(discret_coeffs)(np.linspace(mz_min, mz_max, 100000))
        discret_mean = np.mean(mz_discret)
        discret_std = np.std(mz_discret,ddof=1)
        if discret_std/discret_mean < 0.025: #равномерная дискретизация
            discret_coeffs = np.linalg.solve(sum_XTX[degree:,:1], sum_XTy[:1])

        if draw:
            plt.figure(figsize = (25,4))
            plt.plot(mz[:-1], medfilt(np.diff(mz),5))
            plt.plot(mz[:-1], np.poly1d(discret_coeffs)(mz[:-1]))
            plt.legend(["median filtered m/z discretization example", "m/z discretization regression"])
            plt.xlabel('m/z')
            plt.ylabel('m/z discretion')
        return discret_coeffs

    @abstractmethod
    def get_spectrum(self, idx):
        """Обязательный метод для загрузки одного масс спектра"""
        pass

    @abstractmethod
    def get_metadata(self):
        """Обязательный метод для извлечения метаданных"""
        pass 
    
    @abstractmethod
    def get_mz(self, idx):
        """Обязательный метод для извлечения mz шкалы масс спектра"""
        pass

    @abstractmethod
    def get_intensity(self, idx):
        """Обязательный метод для извлечения интенсивности масс спектра"""
        pass

    @abstractmethod
    def get_batch(self, idxs):
        """Обязательный метод для загрузки пачки масс спектров либо в ленивом формате, если данные неконтинуальные, либо матрицей, если данные континуальные"""
        pass
    
    @abstractmethod
    def get_mz_stream(self, idxs = None):
        """Обязательный метод для извлечения нескольких mz шкал масс спектра ленивым методом"""
        pass

    @abstractmethod
    def get_intensities_stream(self, idxs = None):
        """Обязательный метод для извлечения нескольких интенсивностей масс спектра ленивым методом"""
        pass

    @abstractmethod
    def get_spectrum_sizes(self, idxs = None):
        """Обязательный метод для извлечения размеров масс спектров"""
        pass

###################### IMZML
class loader_imzml(BaseLoader): #TODO @задачка: Класс выгрузки как пример
    """
    Loader for imzML mass spectrometry imaging data.

    Automatically detects whether the data is continuous (TOF) or discontinuous (Orbitrap)
    and selects the appropriate batch-loading strategy.
    """
    def __init__(self, file_path, respec = False):
        self.file_path = file_path
        source = ImzMLParser(file_path)
        # Назначение функции подгрузки в зависимости от континуальности данных 
        mzoffsets = source.mzOffsets
        if mzoffsets[0] == mzoffsets[1]:
            self.dcont = True
            self.mz_scale_cont = source.getspectrum(0)[0]
            self.get_batch = self._get_batch_cont
        else:
            self.dcont = False
            self.get_batch = self._get_batch_discont
        self.create_metafile(respec = respec)
    @cached_property
    def source(self):
        return ImzMLParser(self.file_path)
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('source', None) 
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_batch_cont(self, idxs):
        """
        Load a batch of continuous spectra as a single matrix.

        :param idxs: Index segments to load.
        :type idxs: Indexator

        :yields: Tuple ``(mz_scale, intensity_matrix)``.
        """
        yield self.mz_scale_cont, np.vstack(tuple(self.get_intensities_stream(idxs)))
    
    def _get_batch_discont(self, idxs):
        """
        Lazily yield individual discontinuous spectra.

        :param idxs: Index segments to load.
        :type idxs: Indexator

        :yields: Tuple ``(mz_array, intensity_array)`` for each spectrum.
        """
        for idx in idxs:
            yield self.source.getspectrum(idx)

    def _poslog_parser(self, poslog_path,specnum):
        """
        Parse a Bruker ``_poslog.txt`` file to extract ROI assignments and coordinates.

        :param poslog_path: Path to the ``_poslog.txt`` file.
        :param specnum: Total number of spectra expected.

        :type poslog_path: str
        :type specnum: int

        :return: Tuple ``(roi_list, roi_idx, poslog_specdata)`` where ``roi_idx`` maps ROI names
            to :class:`Indexator` segments and ``poslog_specdata`` is a list of ``(roi, x, y, z)`` tuples.
        :rtype: tuple
        """
        idx=0
        roi_idx = {} 
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
                    roi_idx[current_roi] = Indexator([start_idx, idx]) #TODO @задачка: Indexator - особый класс, упрощающий индексацию и хранение индексов. Просто пишу краткое пояснение к нему
                current_roi = roi_num                                   #  Теперь нужно только знать начальный и конечный индекс. Класс может считать кол-во индексов в сложных случаях (когда много разрывов). 
                start_idx = idx                                         #  Класс используется часто для удобного итерирования. У него есть "родственник" SliceIndexator: Тоже самое, но итерирует слайсами  
                roi_list.append(roi_num)
            poslog_specdata[idx]=(x, y, z)
            idx += 1
        
        # Final ROI update
        if current_roi:
            roi_idx[current_roi] = Indexator((start_idx, idx))
        # return roi_list, roi_idx, poslog_specdata # 160626 перепись легаси функции, теперь координаты выгружаются как общие метаданные в виде единого массива. Суть, выгружать конкретные - позже, 
        # так как есть целая таблица с индексацией
        return roi_list, roi_idx, np.vstack(poslog_specdata)
    
    def get_mz_stream(self, idxs = None):
        if idxs is None:
            idxs = range(len(self.source.mzLengths))
        if self.dcont:
            for idx in idxs:
                yield self.mz_scale_cont
        else:
            for idx in idxs:
                yield self.source.getspectrum(idx)[0]
    def get_intensities_stream(self, idxs):
        for idx in idxs:
            yield self.source.getspectrum(idx)[1]
    
    def get_spectrum(self, idx):
        return self.source.getspectrum(idx)
    
    def get_mz(self, idx = None):
        if self.dcont:
            return self.mz_scale_cont
        else:
            return self.source.getspectrum(idx)[0]
        
    def get_intensity(self, idx):
        return self.source.getspectrum(idx)[1]
    
    def get_batch(self, idxs):
        """Заглушка от @abstractmethod, get_batch назначится динамически после __init__"""
        pass
    def get_metadata(self, draw = True): 
        metadata = {}
        roi_metadata = {}
        specdata = {}
        file_path = self.file_path
        sample_name = os.path.splitext(os.path.basename(file_path))[0] # Имя файла без расширения
        folder_path = os.path.dirname(file_path)

        base_path = os.path.join(folder_path, sample_name) # Базовый стринг пути к файлам без расширения
        poslog_path = os.path.join(folder_path, sample_name) + "_poslog.txt"
        if os.path.exists(base_path+"_info.txt"):
            with open(base_path+"_info.txt") as f:
                data_info = f.readlines()
                specnum = int(data_info[2].split(' ')[-1]) # Информация по кол-ву спектров в sample
        else:
            dpoints = self.source.mzLengths
            specnum = len(dpoints)
        if os.path.exists(poslog_path):
            roi_list, roi_idx, poslog_specdata = self._poslog_parser(poslog_path, specnum)
            specdata["x"] = poslog_specdata[:,0]
            specdata['y'] = poslog_specdata[:,1]
            specdata["z"] = poslog_specdata[:,2]
        else: # If there is no poslog file in the folder, take coordinates from imzml
            #Get base info and coordinates
            roi = "00" # Only one roi
            roi_list = [roi]
            roi_idx = {}
            roi_idx[roi] = Indexator((0,specnum))
            specdata = {}
            try:
                coords = np.vstack(list(self._get_physical_coordinates(range(specnum)))) 
                
            except:
                coords =  np.array(self.source.coordinates)[:,[0,1]]
            specdata["x"] = coords[:,0]
            specdata['y'] = coords[:,1]
            specdata["z"] = np.array([0]*specnum)
        for roi in roi_list:
            roi_metadata[roi] = {}
            roi_metadata[roi]["idxroi"] = roi_idx[roi]
        metadata["continuous"] = self.dcont
        metadata["dtype_raw"] = self.get_spectrum(0)[1].dtype.name

        for roi in roi_list:
            # находим коэффициенты для интерполяции дискретизации (для экономии хранения в метаданных, так как mz_scale может выйти на некоторых приборах в несколько ГБ)
            roi_metadata[roi]["mz_range"] = self.get_mz_range(roi_idx[roi])
            roi_metadata[roi]["discret_coeffs"] = self.get_mz_discretion_coeffs(roi_idx[roi], degree = 3, mz_range =  roi_metadata[roi]["mz_range"], draw = draw)
            if draw:
                plt.title(f'm/z discretization in sample {os.path.basename(file_path)} and roi {roi}')
                plt.show()
                
        # return metadata, roi_metadata, roi_data
        return metadata, roi_metadata, specdata

    def get_spectrum_sizes(self, idxs = None):
        if idxs is None:
            return self.source.mzLengths
        else:
            if isinstance(idxs, (np.ndarray, SliceIndexator)):
                if isinstance(idxs, np.ndarray):
                    idxs = SliceIndexator(idxs)
                mz_lengths = []
                for idxs_slice in idxs:
                    mz_lengths.extend(self.source.mzLengths[idxs_slice])
                return mz_lengths
            elif isinstance(idxs, int):
                return self.source.mzLengths[idxs]
            else:
                raise ValueError("Invalid index type")
    def _get_physical_coordinates(self, idxs):
        for idx in idxs:
            yield self.source.get_physical_coordinates(idx)
class loader_hdf5(BaseLoader): #TODO: Написать
        """
        Загрузка данных из источника.
        В зависимости от формата файла, данные загружаются разными способами.
        """
        def __init__(self, source, dtypeconv):
            self.source = source
            self.dtypeconv = dtypeconv
            self.mz_scale = source.getspectrum(0)[0]
            self.sample_metadata = {}

class loader_mzxml(BaseLoader): # TODO дописать.
    pass
class loader_cdf(BaseLoader): # TODO дописать. #TODO @задачка: Собственно сам класс, который надо написать
    pass

class DataManager():
    """
    WIP
    для работы с различными источниками масс-спектрометрических данных (IMZML, HDF5, MZXML). Подгружает необходимый класс для загрузки данных и
    Обеспечивает унифицированный интерфейс для получения данных m/z шкалы и интенсивностей спектра и метаданных."""
    def __init__(self):
        self._loaders = {
            'imzml': loader_imzml,
            'hdf5': loader_hdf5,
            'mzxml': loader_mzxml,
            'cdf': loader_cdf        
        }
    def get_loader(self, file_path):
        """
        Return the appropriate loader class for the given file path based on its extension.

        :param file_path: Path to the data file.
        :type file_path: str

        :return: Loader class corresponding to the file extension.
        :rtype: class

        :raises ValueError: If the file extension is not supported.
        """
        file_ext = os.path.splitext(file_path)[1][1:]
        loader = self._loaders.get(file_ext.lower(), None)
        if not loader:
            raise ValueError(f'Format {file_ext} is not supported')
        return loader

############### Utility functions

def get_mz_discretion_coeffs_legacy(mz, min_discret = None, draw = True):
    """
    Legacy function to estimate m/z discretisation coefficients from a single m/z array.

    :param mz: 1-D array of m/z values.
    :param min_discret: Optional minimum discretisation threshold for filtering.
    :param draw: If ``True``, display a diagnostic plot. Default ``True``.

    :type mz: np.ndarray
    :type min_discret: float or None
    :type draw: bool

    :return: Polynomial coefficients describing the m/z step size.
    :rtype: np.ndarray
    """
    dots_distance = np.diff(mz)
    
    if min_discret:
        float_error_bool = dots_distance >= min_discret - math.sqrt(np.finfo(float).eps)
        if not float_error_bool[0]:
            first_mz = True
        else:
            first_mz = False
        float_error_bool = np.append(first_mz,float_error_bool)
        mz = mz[float_error_bool]
        dots_distance = np.diff(mz)
    distance_diff = np.diff(dots_distance)
    std_diff = np.std(dots_distance, ddof=1) / np.sqrt(len(dots_distance))
    dots_distance_bool = np.abs(distance_diff) <= std_diff
    start_diff = (dots_distance[0] - dots_distance[1:][dots_distance_bool][0])
    if abs(start_diff) <= std_diff:
        dots_distance_bool = np.append(True, dots_distance_bool)
    else:
        dots_distance_bool = np.append(False, dots_distance_bool)
    
    mz_discret = np.array(medfilt(dots_distance[dots_distance_bool],5))
    mz = mz[:-1][dots_distance_bool]

    discret_mean = np.mean(mz_discret)
    discret_std = np.std(mz_discret,ddof=1)
    if discret_std/discret_mean < 0.025:
        discret_coeffs = np.polyfit(mz, mz_discret, deg = 0)
    else:
        discret_coeffs = np.polyfit(mz, mz_discret, deg = 3)
        if draw:
            plt.figure(figsize = (25,4))
            plt.plot(mz, mz_discret)
            plt.plot(mz, np.poly1d(discret_coeffs)(mz))
            plt.legend(["m/z discretization", "m/z discretization regression"])
            plt.xlabel('m/z')
            plt.ylabel('m/z discretion')
    return discret_coeffs
def _batch_mz_discret_props_legacy(source, idxs):
    """
    Legacy function to collect unique m/z values and minimum discretisation step from a batch of spectra.

    :param source: Data loader with a ``get_mz_stream`` method.
    :param idxs: Index segment ``(start, end)`` to scan.

    :type source: BaseLoader
    :type idxs: tuple

    :return: Tuple ``(unique_mz_array, min_discret_step)``.
    :rtype: tuple
    """
    mzs = []
    min_discret_batch = np.inf
    for mz in source.get_mz_stream(range(*idxs)):
        mzs.append(mz)
        min_discret_batch = min(min_discret_batch, np.diff(mz).min())
    mz_batch = np.unique(np.hstack(mzs))
    return mz_batch, min_discret_batch

