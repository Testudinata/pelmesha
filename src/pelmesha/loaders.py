import os
import re
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
from itertools import product
from functools import cached_property
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
    if slide_name is None:
        slide_name = os.path.basename(hdf5_save_folder)
    if hdf5_end is None:
        hdf5_end = ".hdf5"
    elif not hdf5_end.endswith(".hdf5"):
        hdf5_end = hdf5_end+".hdf5"
    return os.path.join(hdf5_save_folder,slide_name) + hdf5_end

def del_hdf5(hdf5_path):
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        print(f"Deleted file {os.path.basename(hdf5_path)} in directory {os.path.dirname(hdf5_path)}")
    

def del_datasets_hdf5(hdf5_obj, samples_path = None, dataset_to_del = None): 
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

class Dataset(dict):
    """
    WIP
    1. Объединяет датасеты для группировки фич в одну таблицу, с сохранением всех метаданных и ссылок на них. 
    2. При этом освобождает RAM от индивидуальных подгрузок.
    3. Ищет источники по списку путей, если найденный файл не имеет обработанного рядом результата - просит конфиг для обработки. Если есть обработанный, то сравнивает конфиги, если он передан в аргумент. 
    И если они не совпадают производит новую обработку по новому конфигу.
    4. Должен исключать конфликты в названиях sample. WIP ПРидумать как.
    """
    def __init__(self, path_list, dtypeconv):
        self.samples = {}
        if not isinstance(path_list, list):
            path_list = [path_list]
        for path in path_list:
            ## Getting names
            sample_folder_path = os.path.dirname(path)
            sample_name = os.path.splitext(os.path.basename(path))[0]
            folder_name = os.path.basename(sample_folder_path)
            # base_path = os.path.join(sample_folder_path, sample_name)
            if folder_name == sample_name:
                sample = sample_name
            else:
                sample = folder_name + "_" + sample_name
            
            self.samples[sample] = DataSource(path, dtypeconv)
    def __getitem__(self, sample):
        return self.samples[sample]
    def __getattr__(self, sample):
        return self.samples[sample]
    def __iter__(self):
        return iter(self.samples.values())
    def close(self, samples = None):
        """
        Close the data source.
        """
        if samples is not None:
            if isinstance(samples, list):
                for s in samples:
                    self.samples[s].close()
            else:
                self.samples[samples].close()
        else:
            for s in self.samples:
                self.samples[s].close()
# TODO: Основная причина написания класса - это сделать чётко контролируемую и гибкую систему, 
# с помощью которой можно будет подгружать усреднённый спектр при надобности 
# (ввиду того, что он может быть слишком тяжёлым - лучше создавать его непосредственно перед использованием).
class DataSource(dict): #TODO: Develop a source class that we can work with. It doesn’t matter which data source we use — IMZML or HDF5 or maybe MZXML; the approach will be the same.
    # TODO: Необходимо разработать обращение к единичному источнику данных типа imzml и hdf5, который является агрегацией данных из нескольких источников с разбивкой. 
    # Возможно, для создания общности - попробовать сделать как в hdf5 - создать класс DataSource/_loader_imzml для imzml, который может оперировать с несколькими источниками сразу и работал бы схоже с hdf5.
    # В дальнейшем, возможно, это будет затравкой для глобального рефакторинга.
 
    """
    WIP
    Класс основного управления обработкой данных от ОДНОГО файла. Использует класс DataManager для получения источника данных с унифицированным интерфейсом подгрузки.
    СОдержит в себе метаданные источника данных. Сохранные метаданные и конфиги данных.
    Сохраняет всю обработку рядом с источником данных. (может даже для удобства в одной папке с файлом источника данных)
    Подаёдтся непосредственное обращение к источнику данных.
    WIP Отработать подгрузку континуальных и неконтинуальных данных.
    """

    def __init__(self, path):
        """
        Инициализация источника данных.
        :param path: путь к файлу данных
        """
        self.manager = DataManager()
        self.source = self.manager.get_loader(path)(path)
        del self.manager # удаляем за ненадобностью более
        self.metadata, self.roi_metadata = self.source.get_metadata()
    def close(self):
        """
        Close the data source.
        """
        self.source.close()
###################### IMZML
class loader_imzml(): # TODO написать класс для загрузки данных из imzML и также их метадату
    def __init__(self, file_path):
        self.file_path = file_path
        source = ImzMLParser(file_path)

        # Создание функций для батчинговой подгрузки
        def get_batch_cont(self, idxs): 
            yield self.mz_scale_cont, np.fromiter(self._get_batch_intensities(idxs))
        
        def get_batch_discont(self, idxs):
            for idx in idxs:
                yield self.source.getspectrum(idx)

        # Назначение функции подгрузки в зависимости от континуальности данных 
        mzoffsets = source.mzOffsets
        if mzoffsets[0] == mzoffsets[1]:
            self.dcont = True
            self.mz_scale_cont = source.getspectrum(0)[0]
            self.get_batch = get_batch_cont
        else:
            self.dcont = False
            self.get_batch = get_batch_discont
    @cached_property
    def source(self):
        return ImzMLParser(self.file_path)
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('source', None) 
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
    def _get_batch_intensities(self, idxs):
        for idx in idxs:
            yield self.source.getspectrum(idx)[1]
    def get_spectrum(self, idx):
        return self.source.getspectrum(idx)
    def get_mz(self, idx):
        if self.dcont:
            return self.mz_scale_cont
        else:
            return self.source.getspectrum(idx)[0]
    def _get_mz_stream(self, idxs = None):
        if idxs is None:
            idxs = range(len(self.source.mzLengths))
        for idx in idxs:
            yield self.source.getspectrum(idx)[0]

    def get_metadata(self, Ram_Gb_usage = 1, free_cpu = 1, draw = True): # TODO разобрать 
        Ram_Gb_usage = Ram_Gb_usage * (1024 ** 3)
        self.metadata = {}
        self.roi_metadata = {}
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
            for roi in roi_list:
                self.roi_metadata[roi]={}
                self.roi_metadata[roi]["xy"] = np.empty((roi_idx[roi][1],2))
                self.roi_metadata[roi]["z"] = np.empty((roi_idx[roi][1],1))
            for idx, (roi,x,y,z) in enumerate(poslog_specdata):            
                self.roi_metadata[roi]["xy"][idx-roi_idx[roi][0],:] = [x, y]
                self.roi_metadata[roi]["z"][idx-roi_idx[roi][0]]  = z
        else: ### If there is no poslog file in the folder, take coordinates from imzml
            #### Stage 3 from imzml. Get base info and coordinates
            #### Initialization
            roi = "00" # Only one roi
            roi_list = [roi]
            roi_idx = {}
            self.roi_metadata[roi]={}
            roi_idx[roi] = (0,specnum)
            try:
                self.roi_metadata[roi]['xy'] = np.fromiter(self.get_physical_coordinates(range(specnum)))
            except:
                self.roi_metadata[roi]['xy'] = np.array(self.source.coordinates)[:,[0,1]]
            self.roi_metadata[roi]["z"] = np.array([0]*specnum) # Заглушка z- координаты нигде не узнать
        
        for roi in roi_list:
            indexes = roi_idx[roi]
            self.roi_metadata[roi]["idxroi"] = indexes


        self.metadata["continuous"] = self.dcont
        self.metadata["dtype_raw"] = self.source.get_spectrum(0)[1].dtype
        cpu_usage_count = cpu_count() - free_cpu
        Element_size_per_cpu = Ram_Gb_usage / (self.metadata["dtype_raw"].itemsize * cpu_usage_count)

        for roi in roi_list:
            # Определяем дискретизацию шкалы mz
            if self.dcont:
                mz_scale = self.mz_scale_cont
                min_discret = None
                min_mz = min(mz_scale)
                max_mz = max(mz_scale)
            else: #для неконтинуальных данных пытаемся восстановить полную дискретную mz шкалу перебирая уникальные значениия
                #new
                mz_scale = np.empty(0)
                min_discret = np.inf
                batching = []
                start = 0
                current_el_size = 0
                for idx in range(*roi_idx[roi]):
                    if current_el_size > Element_size_per_cpu:
                        batching.append((start, idx))
                        start = idx
                        current_el_size = dpoints[idx]
                    else:
                        current_el_size += dpoints[idx]
                batching.append((start,idx))
                with Pool(cpu_usage_count) as p:
                    for mz_batch, min_discret_batch in p.imap_unordered(_batch_mz_discret_props, product([self], batching)):
                        mz_scale = np.sort(np.unique(np.hstack((mz_scale, mz_batch))))
                        min_discret = min(min_discret_batch, min_discret)
                
                min_mz = mz_scale[0]
                max_mz = mz_scale[-1]
                # #old
                # mzs = []
                # mz_scale = np.empty(0)
                # min_discret = np.inf
                # dpoints = self.source.mzLengths
                # current_el_size = 0
                # for idx, mz in enumerate(self._get_mz_stream(range(*roi_idx[roi]))):
                #     mzs.append(mz)
                #     min_discret = min([min_discret, np.diff(mz).min()])
                #     current_el_size += dpoints[idx]
                #     if current_el_size > Element_size:
                #         mz_scale = np.sort(np.unique(np.hstack(mzs)))
                #         mzs = [mz_scale]
                #         current_el_size = mz_scale.shape[0]
                #         print(idx)
                # if len(mzs) > 1:
                #     mz_scale = np.sort(np.unique(np.hstack(mzs)))
                # min_mz = mz_scale[0]
                # max_mz = mz_scale[-1]
            # находим коэффициенты для интерполяции дискретизации (для экономии хранения в метаданных, так как mz_scale может выйти на некоторых приборах в несколько ГБ)
            discret_coeffs = get_mz_discretion_coeffs(mz_scale, min_discret, draw)
            if len(discret_coeffs) > 1:
                plt.title(f'm/z discretization in sample {os.path.basename(file_path)} and roi {roi}')
                plt.show()
            self.roi_metadata[roi]["mz_range"] = (min_mz, max_mz)
            self.roi_metadata[roi]["discret_coeffs"] = discret_coeffs
        return self.metadata, self.roi_metadata
    def _poslog_parser(self, poslog_path,specnum): # TODO сделать рефакторинг, так как это легаси код до создания фабрики загрузки данных
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
    def get_physical_coordinates(self, idxs):
        for idx in idxs:
            yield self.source.get_physical_coordinates(idx)
class loader_hdf5(): #TODO: Написать
        """
        Загрузка данных из источника.
        В зависимости от формата файла, данные загружаются разными способами.
        """
        def __init__(self, source, dtypeconv):
            self.source = source
            self.dtypeconv = dtypeconv
            self.mz_scale = source.getspectrum(0)[0]
            self.sample_metadata = {}



class loader_mzxml(): # TODO дописать.
    def __init__(self, path, dtypeconv):
        self.source = source
        self.dtypeconv = dtypeconv
class loader_cdf(): # TODO дописать.
    def __init__(self, source, dtypeconv):
        self.source = source
        self.dtypeconv = dtypeconv

class DataManager():
    """
    WIP
    для работы с различными источниками масс-спектрометрических данных (IMZML, HDF5, MZXML). Подгружает необходимый класс для загрузки данных и
    Обеспечивает унифицированный интерфейс для получения данных m/z шкалы и интенсивностей спектра."""
    def __init__(self):
        self._loaders = {
            'imzml': loader_imzml,
            'hdf5': loader_hdf5,
            'mzxml': loader_mzxml,
            'cdf': loader_cdf        
        }
    def get_loader(self, file_path):
        file_ext = os.path.splitext(file_path)[1][1:]
        loader = self._loaders.get(file_ext.lower(), None)
        if not loader:
            raise ValueError(f'Format {file_ext} is not supported')
        return loader

############### Utility functions
def get_mz_discretion_coeffs(mz, min_discret = None, draw = True):
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
    
    mz_discret = np.array(medfilt(dots_distance[dots_distance_bool],7))
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
def _batch_mz_discret_props(source, idxs):
    mzs = []
    min_discret_batch = np.inf
    for mz in source._get_mz_stream(range(*idxs)):
        mzs.append(mz)
        min_discret_batch = min(min_discret_batch, np.diff(mz).min())
    mz_batch = np.unique(np.hstack(mzs))
    return mz_batch, min_discret_batch
def get_mean_spectrum(file_path, roi_idx = None, Ram_Gb = 2): #TODO: need refactoring. Especially with integration to DataSource
    if isinstance(file_path, ImzMLParser):
        source = file_path
        file_path = source.filename
    else:
        source = ImzMLParser(file_path)
    Ram_Gb = Ram_Gb * (1024 ** 3)
    mean_spectrum ={}
    specnum = len(source.mzLengths)
    if roi_idx is None:
        sample_folder_path = os.path.dirname(file_path)
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        base_path = os.path.join(sample_folder_path, sample_name)
        poslog_path = os.path.join(base_path+'_poslog.txt')
        if os.path.exists(poslog_path): ### Extraction from _poslog and _info text files
            roi_list, roi_idx, poslog_specdata = _poslog_parser(poslog_path, specnum)
        else:
            roi = "00" # Only one roi
            roi_list = [roi]
            roi_idx = {}
            roi_idx[roi] = (0,specnum)
    else:
        if isinstance(roi_idx, (list,tuple)):
            roi_list = ['00']
            roi_idx = {roi_list[0]:roi_idx}
        elif isinstance(roi_idx, dict):
            roi_list = list(roi_idx.keys())
        else:
            raise ValueError('roi_idx must be a list, tuple or dict')
        
    for roi in roi_list:
        stats = pd.DataFrame(columns=['sum','count'])
        stats.index.name = 'mz'
        start_idx, end_idx = roi_idx[roi]
        batch = []
        Ram_usage = stats.memory_usage(deep=True, index=True).sum()
        for idx in range(start_idx, end_idx):
            spectrum = pd.DataFrame(np.vstack(source.getspectrum(idx)).T, columns=['mz', 'intensities'])
            Ram_usage += spectrum.memory_usage(deep=True, index=True).sum()
            batch.append(spectrum)
            if Ram_usage > Ram_Gb:
                stats = stats.add(
                                pd.concat(batch, axis=0, ignore_index=True)
                                .groupby('mz')['intensities']
                                .agg(sum='sum', count='count'),
                                fill_value = 0
                                )
                batch = []
                Ram_by_stats = stats.memory_usage(deep=True, index=True).sum()
                Ram_usage = Ram_by_stats if Ram_by_stats < Ram_Gb/2 else 0
        if batch:
            stats = stats.add(
                                pd.concat(batch, axis=0, ignore_index=True)
                                .groupby('mz')['intensities']
                                .agg(sum='sum', count='count'),
                                fill_value = 0
                                )
            del batch, Ram_usage, spectrum
        mean_spectrum[roi] = (np.array(list(stats.index)), np.array(list(stats['sum']/stats['count'])))
        del stats
    if isinstance(roi_idx, (list,tuple)):
        return mean_spectrum[roi]
    else:
        return mean_spectrum