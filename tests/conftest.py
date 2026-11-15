import copy
import os
import warnings
import pytest
import numpy as np
import datetime as dt
import pelmesha.kneading
from pelmesha import DataSet
from pelmesha.filling import DataSource
from pelmesha.dough import Indexator
import pelmesha.utensils as utensils 
from functools import wraps
import inspect
from datetime import datetime

import matplotlib
@pytest.fixture(scope="session", autouse=True)
def silent_matplotlib():
    matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class SuspiciousSpectraCountWarning(UserWarning):
    pass


date_now = datetime.now().strftime('%Y-%m-%d')
def draw_fig(func):
    sig = inspect.signature(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        func(*args,**kwargs)
        self = bound.arguments['self']
        ids = wrapper.current_request.node.callspec.id
        save2 = os.path.split(self.datasource.file_path)
        save2 = os.path.join(os.path.dirname(save2[0]), "test_figs", date_now, f"{save2[-1].split(".")[0]}_{ids}.png")

        os.makedirs(os.path.dirname(save2), exist_ok= True)
        plt.savefig(save2)
        plt.close('all')
        return
    return wrapper

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Test_data")

DATA_PATHS = [
    pytest.param(os.path.join(TEST_DATA_DIR, "imzml_cont"), id="imzml cont"),
    pytest.param(os.path.join(TEST_DATA_DIR, "imzml_discont"), id="imzml discont"),
    pytest.param(os.path.join(TEST_DATA_DIR, "cdf"), id="cdf"),
]
signal_distortions = [(99, 100.2),
                    (102.09, 103.86),
                    (107.79, 109.4),
                    (111.65, 113.5),
                    (122.2, 124.5),
                    (127, 128.2),
                    (133.4, 134.0), 
                    (134.33,136.12),
                    (140,142),
                    (154.7, 157.7),
                    (159, 164), 
                    (168.5, 170.8),
                    (171.5,172.9),
                    (188.75, 189.15),
                    (201.00, 203.44),
                    (203.6,206),
                    (206.4, 210.8),
                    (210.85,211.60), 
                    (220.10, 221.02), 
                    (228.36, 231.53), 
                    (241, 243.0),
                    (275.5, 278.4), 
                    (285.4, 287.175),
                    (295.5,301),
                    (321.6, 322.0),
                    (338,339.025),
                    (339.3,340.15),
                    (340.25,341),
                    (370.6, 370.95),
                    (375.0, 375.6),
                    (396.5, 401.7), 
                    (472.29, 472.89),
                    (496,496.25),
                    (496.38,497.3), 
                    (508.34, 512.61),
                    (606.4,607.45),
                    (607.5,608), 
                    (620.00, 627.5), 
                    (642.8, 647.03), 
                    (674.0, 688),
                    (806.6,807),
                    (809,809.5),
                    (809.7,810),
                    (881.9,882.5),
                    (882.6,883), 
                    (906.92, 908.83)]
PEAKPICKING_PARAMS = [
    pytest.param({"Baseline": None,
                  "SNR_threshold": None,
                  "smooth_algo": None,
                  "return_areas": False}, id = "ALL_OFF"),
    pytest.param({}, id = "default"),
    pytest.param({'resample_num_points': 300000,
                  'mz_crop_range': (100,1000),
                  'zero_points_to_peaks_ext': True,
                  "mz_segments_to_zero": signal_distortions,
                  'smooth_algo': 'SG',
                  'smooth_window':9,
                  'smooth_cycles':3,
                  'align_peaks': None, # Для каждого тестового источника данных - должны быть назначены свои
                  'align_shift_range': (-0.5,0.5),
                  'fwhm_min':0.01,
                  'fwhm_max':1.1,
                  'fwhm_merge_factor':0.85,
                  'peaklocation':0.95,
                  'return_areas':True,
                  'heightfilter': None,
                  'rel_heightfilter': 0.01,
                  'SNR_threshold': 5,
                  'lam': 500000}, id = "FULL_ON")
]


KDE_PARAM_GRID = [
    # базовые случаи
    pytest.param({},                                                  id='default'),
    pytest.param(dict(KD_bandwidth='fwhm', KDE_algo='fft'),           id='fwhm-fft'),
    pytest.param(dict(KD_bandwidth='mz_discret', KDE_algo='tree'),      id='mzdisc-tree'),
    pytest.param(dict(KD_bandwidth=0.02,   KDE_algo='tree'),          id='fixed-tree'),
    # вариация ширины
    pytest.param(dict(KD_bandwidth='fwhm', KDE_algo='tree', bwc=2.0), id='fwhm-bwc2'),
    pytest.param(dict(KD_bandwidth='fwhm', KDE_algo='tree', account_mzscale=False),
                 id='fwhm-clamp-off')
]
# Декоратор подсчёта кол-ва спектров в батчах для split_idxs 
def counting_in_out(func):
    sig = inspect.signature(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        idxs_batches = func(*args, **kwargs)
        idxs = bound.arguments['idxs']
        self = bound.arguments['self']

        idxs = self._normalize_indices(idxs)
        for idx in idxs:
            assert idx.size == 2, f'Индексы должны быть массивами размера 2, а не {idx.size}, вероятно проблема в _normalize_indices'
        assert np.vstack(idxs_batches).shape[1] == 2, f'Индексы в батчах должны быть массивами размера 2, а не {np.vstack(idxs_batches).shape[1]}'
            # for idx in idx_batch:
            #     assert idx.size == 2, f"Индексы в батчах должны быть массивами размера 2, а не {idx_batch.size}"
        assert Indexator(np.vstack(idxs_batches)).count == Indexator(idxs).count, f' Пропали спектры! На выходе {Indexator(np.vstack(idxs_batches)).count} != на входе {Indexator(idxs).count}'
        return idxs_batches
    return wrapper

DataSource.split_idxs = counting_in_out(DataSource.split_idxs)

# ПРоверка условия неперекрытия KDE сегментов
def kde_segments_not_overlap(func):

    sig = inspect.signature(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        peaklist = bound.arguments['pd_table']

        KD_data = func(*args, **kwargs)
        eps = np.finfo(np.float64).eps

        for (mz_a, bw_a), (mz_b, bw_b) in zip(KD_data[:-1], KD_data[1:]):
            seg_a_max = mz_a[-1] + 6.0 * bw_a[-1]   # правый край сетки сегмента A
            seg_b_min = mz_b[0] - 6.0 * bw_b[0]     # левый край сетки сегмента B
            scale = max(abs(seg_a_max), abs(seg_b_min), 1.0)
            assert seg_a_max <= seg_b_min + eps * scale, \
                f"Сетки сегментов пересекаются: segA_max={seg_a_max:.16g} > segB_min={seg_b_min:.16g} + {eps*scale:.2g}"
            
            assert len(mz_a) > 1 and len(mz_b) > 1, f"Сетки сегментов должны быть длиннее 1 {mz_a}, {mz_b}"

        assert peaklist['mz'].count() == sum( len(t[0]) for t in KD_data), f'Есть потери, капитан! Количество пиков в пиклисте {peaklist['mz'].count()} != кол-ву пиков в сегментированных данных {sum( len(t[0]) for t in KD_data)}'
        return KD_data
    return wrapper

pelmesha.kneading.split_pdtable_by_peaks_gap = kde_segments_not_overlap(utensils.split_pdtable_by_peaks_gap)

# fixture
def pytest_addoption(parser):
    """Добавляем свой флаг: pytest --seed 12345"""
    parser.addoption(
        "--seed", type=int, default=None,
        help="Фиксированный сид (иначе — UTC-дата запуска)",
    )


@pytest.fixture(scope="session")
def date_seed(request):
    """Значение из флага --seed или None, если флага не было."""
    return request.config.getoption("--seed")


@pytest.fixture(autouse=True, scope="session")
def _seed(date_seed):
    if date_seed is None:
        today = dt.datetime.now(dt.timezone.utc).date()
        seed = today.year * 10000 + today.month * 100 + today.day
    else:
        seed = date_seed
    np.random.seed(seed)
    print(f"\n[seed] {seed}")
    yield

@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()
@pytest.fixture(scope="session", autouse=True)
def apply_test_hooks(monkeypatch_session):
    """икстура обёртки методов проверками на время тестов"""
    monkeypatch_session.setattr(DataSource, "split_idxs", counting_in_out(DataSource.split_idxs))
    monkeypatch_session.setattr(utensils, "split_pdtable_by_peaks_gap", kde_segments_not_overlap(utensils.split_pdtable_by_peaks_gap))
    yield  


@pytest.fixture(scope="session", params = DATA_PATHS)
def path(request):
    return request.param

@pytest.fixture(scope="session", params=DATA_PATHS)
def ds(request):
    return DataSet(request.param)

def _generate_spectra(ds):
    for source_name, source in ds.sources.items():
        for roi in source.roi_metadata.index:
            idxs = source.roi_metadata.loc[roi,'idxroi']
            rand_idx = np.random.choice(list(Indexator(idxs)))
            yield source_name, roi, rand_idx, *source._datasource.get_spectrum(rand_idx) 

@pytest.fixture(params=PEAKPICKING_PARAMS)
def processing_params(request):
    return copy.deepcopy(request.param)


@pytest.fixture(scope="session", params = KDE_PARAM_GRID)
def kde_params(request):
    return request.param

def peaklists(ds):
    for source_name, source in ds.sources.items():
        for roi in source.roi_metadata.index:
            yield source_name, roi, ds[source_name].peaklist(roi)

def ds_kde(path, kde_params):
    return DataSet(path, kde_params)


def pytest_collection_modifyitems(session, config, items):
    """
    Гарантирует, что FULL_ON-варианты TestProcessing_full из tests/test_kde.py
    выполняются ПОСЛЕДНИМИ внутри своего модуля.

    Каждый запуск test_processing перезаписывает пиклисты на диске, поэтому
    именно FULL_ON-прогон, отработавший последним, оставляет на диске пиклисты
    с FULL_ON-обработкой. KDE/feature-matrix тесты переиспользуют эти пиклисты
    без повторной обработки данных. Хук объявлен в conftest, потому что в
    тестовом модуле хуки pytest не вызываются.
    """
    def _is_test_kde(item):
        module = getattr(item, "module", None)
        if module is None:
            return False
        module_file = getattr(module, "__file__", None) or ""
        return os.path.splitext(os.path.basename(module_file))[0] == "test_kde"

    positions = [i for i, item in enumerate(items) if _is_test_kde(item)]
    if not positions:
        return
    lo, hi = min(positions), max(positions) + 1
    inside = items[lo:hi]
    # Порядок внутри модуля: весь test_processing (не-FULL_ON) ->
    # весь test_processing (FULL_ON последним) -> остальные (KDE/feature-matrix).
    proc = [it for it in inside if "test_processing" in it.name]
    rest = [it for it in inside if "test_processing" not in it.name]
    items[lo:hi] = ([it for it in proc if "FULL_ON" not in it.name]
                    + [it for it in proc if "FULL_ON" in it.name]
                    + rest)
