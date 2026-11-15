from pelmesha import DataSet
import os
import copy
import inspect
import numpy as np
import pandas as pd
import pytest
import yaml
from tests.conftest import DATA_PATHS, PEAKPICKING_PARAMS, draw_fig, SuspiciousSpectraCountWarning
from pelmesha.serving import Drawer
from pelmesha.cookbook import KDEConfigs
from pelmesha.dough import Indexator
from pelmesha.kneading import FWHM_TO_SIGMA_FACTOR, _build_headers, peakpicker
from h5py import File
import pelmesha.kneading as kneading
import pelmesha.serving as serving
import warnings

# Все папки с урезанными реальными данными (пути заданы в conftest).
ALL_DATA_PATHS = [p.values[0] for p in DATA_PATHS]


def assert_kde_compute(func):
    def wrapper(*args, **kwargs):
        X, Y =func(*args, **kwargs)
        assert X.ndim == Y.ndim == 1
        assert X.size == Y.size and X.size > 0
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))
        # FFT-алгоритм даёт шум порядка -1e-14 из-за ошибок округления,
        # поэтому допускаем крошечные отрицательные значения плотности
        assert np.all(Y >= -1e-12)           # плотность неотрицательна
        assert np.all(np.diff(X) > 0)        # сетка строго монотонна после argsort
        return X, Y
    return wrapper

# _compute_KDE импортирован в serving по имени, поэтому для того чтобы sanity-
# проверки покрывали тестируемый код, патчим оба пространства имён.
kneading._compute_KDE = assert_kde_compute(kneading._compute_KDE)
serving._compute_KDE = assert_kde_compute(serving._compute_KDE)

def kde_recipe_path(source, sample_name):
    """Стандартный путь к сохранённому KDE-рецепту источника."""
    return os.path.join(os.path.dirname(source.file_path),
                        "processed_pelmesha", f"{sample_name}_kde_recipe.yaml")

# Порядок выполнения FULL_ON-вариантов TestProcessing_full гарантируется хуком
# pytest_collection_modifyitems в conftest.py (хуки не работают, если объявлены
# в самом тестовом модуле).

class TestProcessing_full:
    @pytest.fixture(autouse=True)
    def setup_draw_mock(self, monkeypatch, request):
        patched_draw = draw_fig(Drawer.draw_processed_data)
        patched_draw.current_request = request
        monkeypatch.setattr(Drawer, "draw_processed_data", patched_draw)

    def test_processing(self,
                        path,
                        processing_params,
                        request):
        if 'imzml_cont-FULL_ON' in request.node.callspec.id:
            processing_params['align_peaks'] = [518.3217, 780.5514, 804.5514, 828.5514]
            processing_params['align_shift_range'] = (-0.5,0.5) 
            processing_params['align_width'] = [0.5*FWHM_TO_SIGMA_FACTOR]
        elif 'imzml_discont-FULL_ON' in request.node.callspec.id:
            processing_params['align_peaks'] = [518.3217, 780.5514, 804.5514, 828.5514] 
            processing_params['align_shift_range'] = (-0.25,0.25) 
            processing_params['align_width'] = [0.025*FWHM_TO_SIGMA_FACTOR]
            processing_params['Baseline'] = None
        
        ds = DataSet(path, **processing_params)
        if 'cdf-FULL_ON' in request.node.callspec.id:
            for source_name, source in ds.sources.items():
                for roi in source.roi_metadata.index:
                    if roi == "0":
                        source.roi_configs[roi]["align_peaks"] = [518.3217, 780.5514, 804.5514, 828.5514]
                        source.roi_configs[roi]['align_shift_range'] = (-0.5,0.5) 
                        source.roi_configs[roi]['align_width'] = [0.5*FWHM_TO_SIGMA_FACTOR]
                    if roi == "1":
                        source.roi_configs[roi]["align_peaks"] =  [215.03, 255.233, 303.233, 327.233, 514.28388, 762.5079, 792.5549, 885.5499]
                        source.roi_configs[roi]['align_shift_range'] = (-0.5,0.5) 
                        source.roi_configs[roi]['align_width'] = [0.5*FWHM_TO_SIGMA_FACTOR]
                    if roi == "2":
                        source.roi_configs[roi]["align_peaks"] = [518.3217, 780.5514, 804.5514, 828.5514]
                        source.roi_configs[roi]['align_shift_range'] = (-0.25,0.25) 
                        source.roi_configs[roi]['align_width'] = [0.025*FWHM_TO_SIGMA_FACTOR]
                        source.roi_configs[roi]['Baseline'] = None
                    if roi == "3":
                        source.roi_configs[roi]['align_shift_range'] = (-0.25,0.25)
                        source.roi_configs[roi]["align_peaks"] = [215.03, 255.233, 303.233, 327.233, 514.28388, 762.5079, 792.5549, 885.5499]
                        source.roi_configs[roi]['align_width'] = [0.025*FWHM_TO_SIGMA_FACTOR]
                        source.roi_configs[roi]['Baseline'] = None
        ds.peakpick(draw = True,
                    draw_mz_range = (750,800))
        sig = inspect.signature(peakpicker)
        headers = _build_headers(processing_params.get('return_areas', 
                                                       sig.parameters['return_areas'].default),
                                processing_params.get('SNR_threshold', 
                                                        sig.parameters['SNR_threshold'].default))
        n_headers = len(headers)
        for source_name, source in ds.sources.items():
            assert os.path.exists(source.peaklists_path), f'Peaklist for {source_name} does not exist after peakpick'

            with File(source.peaklists_path, 'r') as h5file:
                for roi in h5file.keys():
                    num_spectra = Indexator(source.roi_metadata.loc[roi, 'idxroi']).count
                    attr_headers = h5file[roi].attrs['Column headers']
                    loc_spectra_ind = np.where(attr_headers == 'spectra_ind')[0]
                    unique_count = len(np.unique(h5file[roi][:, loc_spectra_ind]))
                    assert np.all(attr_headers == headers), f'HDF5 peaklist for {source_name} has wrong headers'
                    assert h5file[roi].shape[1] == n_headers, f'HDF5 peaklist for {source_name} is not have {n_headers} columns {h5file[roi].shape[1]}'
                    assert unique_count <= num_spectra, (
                        f"HDF5 peaklist for {source_name} has more unique numbers on "
                        f"spectra_ind column ({unique_count}) than number of spectra ({num_spectra})"
                    )
                    if unique_count < num_spectra:
                        warnings.warn(
                            f"HDF5 peaklist for {source_name}: number of unique spectra ({unique_count}) "
                            f"is LESS than expected ({num_spectra}). This is within the allowed norm, "
                            f"but it might be caused by an error in data generation.",
                            category=SuspiciousSpectraCountWarning
                        )
                    
                    peaklist = source.peaklists(roi)
                    assert peaklist is not None, f'Peaklist for {source_name} is empty'
                    assert isinstance(peaklist, pd.DataFrame), f'Peaklist for {source_name} is not a DataFrame'
                    assert list(peaklist.columns) == headers, f'Columns {peaklist.columns} do not match {headers} for {source_name} {roi}'
                    
                    # Спектр без единого пика законно отсутствует в пиклисте,
                    # поэтому требуем только непревышение числа спектров ROI.
                    assert peaklist['spectra_ind'].unique().size <= num_spectra, \
                        f"Extracted peaklist for {source_name} contains more spectra than expected ({num_spectra})"


###############################################################################
#  estimate_peak_density_kde                                                  #
###############################################################################

@pytest.fixture(scope="session", params=DATA_PATHS)
def ds_peak(request):
    """
    DataSet для одной папки тестовых данных.

    Пиклисты FULL_ON уже подготовлены последним FULL_ON-прогоном
    TestProcessing_full (см. pytest_collection_modifyitems) и просто читаются
    с диска без повторной обработки. Если модуль запущен изолированно и
    пиклистов нет - выполняется резервный пикпикинг.
    """
    ds = DataSet(request.param)
    if not all(os.path.exists(source.peaklists_path) for source in ds.sources.values()):
        ds.peakpick(draw=False)
    yield ds
    ds.close()


class TestKDEEstimate:
    """
    Sanity-проверки DataSet.estimate_peak_density_kde на урезанных реальных
    данных: результаты записаны в файлы, сетки/плотности согласованы между
    собой и с исходными пиклистами, KDE-рецепт сохранён рядом с данными.
    """

    @pytest.fixture(autouse=True)
    def kde_estimated(self, ds_peak, kde_params):
        """Запускает estimate_peak_density_kde один раз на пару (папка, KDE-конфиг)."""
        cfg = KDEConfigs(**kde_params)
        self.ds = ds_peak
        for source in ds_peak.sources.values():
            for roi in list(source.roi_kde_configs):
                source.roi_kde_configs[roi] = copy.deepcopy(cfg)
        ds_peak.estimate_peak_density_kde(draw=False)
        yield

    def test_peaks_density_file_written(self):
        for name, source in self.ds.sources.items():
            density_path = source._datasource.peaks_density_path
            assert os.path.exists(density_path), \
                f'Peaks density file for {name} does not exist after estimate_peak_density_kde'

            with File(density_path, "r") as f:
                for roi in source.rois:
                    assert roi in f, f'ROI "{roi}" is missing in the peaks density file of {name}'
                    assert "mz" in f[roi] and "peaks_density" in f[roi], \
                        f'Peaks density group of {name} roi {roi} must contain "mz" and "peaks_density"'

                    mz = f[roi]["mz"][:]
                    density = f[roi]["peaks_density"][:]
                    assert mz.ndim == density.ndim == 1, \
                        f'KDE grids for {name} roi {roi} must be 1-dimensional'
                    assert mz.size == density.size, \
                        f'm/z and density grids have different sizes for {name} roi {roi}'

                    if mz.size:
                        assert np.all(np.isfinite(mz)) and np.all(np.isfinite(density)), \
                            f'KDE result for {name} roi {roi} contains non-finite values'
                        # допуск на шум FFT-алгоритма (~ -1e-14)
                        assert np.all(density >= -1e-12), \
                            f'Negative density values for {name} roi {roi}'
                        assert np.all(np.diff(mz) > 0), \
                            f'KDE m/z grid for {name} roi {roi} is not strictly increasing'

    def test_kde_grid_overlaps_peaklist_mz(self):
        """Сетка KDE должна перекрываться с диапазоном m/z пиклиста, по которому она построена."""
        for name, source in self.ds.sources.items():
            for roi in source.rois:
                with File(source._datasource.peaks_density_path, "r") as f:
                    mz = f[roi]["mz"][:]
                peaklist_mz = source._datasource.peaklists(roi)["mz"].to_numpy()

                if mz.size and peaklist_mz.size:
                    assert mz.min() <= peaklist_mz.max() and mz.max() >= peaklist_mz.min(), \
                        f'KDE grid of {name} roi {roi} does not overlap its peaklist m/z range'

    def test_kde_recipe_dumped(self, kde_params):
        """estimate_peak_density_kde сохраняет актуальный KDE-рецепт каждого сэмпла."""
        for name, source in self.ds.sources.items():
            recipe_path = kde_recipe_path(source, name)
            assert os.path.exists(recipe_path), f'KDE recipe for {name} was not saved'

            with open(recipe_path, encoding="utf-8") as f:
                recipe = yaml.safe_load(f)
            assert isinstance(recipe, dict) and recipe, f'KDE recipe of {name} is empty'
            for roi in source.rois:
                assert roi in recipe, f'KDE recipe of {name} misses ROI "{roi}"'
                for key, value in kde_params.items():
                    assert recipe[roi].get(key) == value, \
                        f'KDE recipe of {name} roi {roi} does not reflect the "{key}" parameter'


###############################################################################
#  feature_matrix                                                             #
###############################################################################

@pytest.fixture(scope="session")
def ds_fm():
    """
    DataSet сразу по всем папкам с тестовыми данными.

    Пиклисты FULL_ON подготовлены последним FULL_ON-прогоном TestProcessing_full
    и читаются с диска без повторной обработки (с резервным пикпикингом при
    изолированном запуске). Затем однократно оценивается плотность пиков (KDE).
    """
    ds = DataSet(ALL_DATA_PATHS)
    if not all(os.path.exists(source.peaklists_path) for source in ds.sources.values()):
        ds.peakpick(draw=False)
    ds.estimate_peak_density_kde(draw=False)
    yield ds
    ds.close()


class TestFeatureMatrix:
    """
    Sanity-проверки DataSet.feature_matrix на всех урезанных реальных данных:
    структура матрицы, сохранение результатов в файлы и базовое поведение
    фильтрации/склейки дубликатов.
    """

    @pytest.fixture(autouse=True)
    def fm_base(self, ds_fm):
        self.ds = ds_fm
        # Без частотной фильтрации и без склейки дубликатов матрица обязана
        # содержать ровно все строки исходных пиклистов.
        self.fm = ds_fm.feature_matrix(draw=False, show_stats=False,
                                       countf=None, countf_rel=None,
                                       duplicates_drop=False)
        yield

    def _total_peaklist_rows(self):
        return sum(source._datasource.peaklists(roi).shape[0]
                   for source in self.ds.sources.values()
                   for roi in source.rois)

    def test_feature_matrix_structure(self):
        fm = self.fm
        assert isinstance(fm, pd.DataFrame) and not fm.empty, \
            'feature_matrix must return a non-empty DataFrame'
        assert isinstance(fm.index, pd.MultiIndex), \
            'feature_matrix index must be a MultiIndex (sample, roi, spectra_ind)'
        assert list(fm.index.names) == ["sample", "roi", "spectra_ind"], \
            f'Wrong index names: {fm.index.names}'

        presented = set(fm.index.get_level_values("sample"))
        assert presented, 'No samples got into the feature matrix'
        assert presented <= set(self.ds.sources.keys()), \
            f'Unexpected samples in the matrix: {presented - set(self.ds.sources.keys())}'

        for col in ("mz", "Intensity", "FWHM"):
            assert col in fm.columns, f'Column "{col}" is missing in the feature matrix'

        assert np.all(np.isfinite(fm["mz"])) and np.all(fm["mz"] > 0), \
            'm/z column contains non-finite or non-positive values'

        # Ни одна строка пиклистов не потеряна и не задублирована.
        assert len(fm) == self._total_peaklist_rows(), \
            f'feature_matrix has {len(fm)} rows while source peaklists have {self._total_peaklist_rows()}'

    def test_duplicates_drop_reduces_rows(self):
        """Склейка дубликатов не должна увеличивать число строк."""
        fm_merged = self.ds.feature_matrix(draw=False, show_stats=False,
                                           countf=None, countf_rel=None,
                                           duplicates_drop=True)
        assert len(fm_merged) <= len(self.fm)

    def test_frequency_filtration(self):
        """Частотная фильтрация оставляет подмножество пиков из неотфильтрованной матрицы."""
        fm_kept = self.ds.feature_matrix(draw=False, show_stats=False,
                                         countf_rel=0.1, duplicates_drop=False)
        assert len(fm_kept) <= len(self.fm), \
            'Filtration must not add rows'
        assert set(fm_kept["mz"]) <= set(self.fm["mz"]), \
            'Filtration produced m/z values that are absent in the unfiltered matrix'

    def test_pivot_and_save(self, tmp_path):
        """Пивот по Intensity + сохранение: parquet матрицы, координат, binning rule и конфига агрегации."""
        save_path = tmp_path / "feature_matrix.parquet"
        fm = self.ds.feature_matrix(draw=False, show_stats=False,
                                    countf=None, countf_rel=None,
                                    pivot_values="Intensity",
                                    save_path=str(save_path),
                                    save_binning_rule=True)

        # Матрица сохранена и совпадает с возвращённой.
        assert save_path.exists(), 'Feature matrix parquet was not saved'
        saved = pd.read_parquet(save_path)
        assert saved.shape == fm.shape, 'Saved feature matrix differs from the returned one'

        # После пивота значения m/z стали колонками, пропусков нет (fill_values=0.0).
        assert fm.columns.name == "mz", \
            f'After pivoting the columns must be named by m/z, got name={fm.columns.name!r}'
        assert not fm.isna().any().any(), 'Pivoted feature matrix contains missing values'

        # Координаты сохранены рядом и покрываются строками матрицы.
        coords_path = tmp_path / "feature_matrix_coordinates.parquet"
        assert coords_path.exists(), 'Coordinates parquet was not saved'
        coords = pd.read_parquet(coords_path)
        assert len(coords) > 0, 'Saved coordinates are empty'
        assert set(map(tuple, coords.index)) <= set(map(tuple, fm.index)), \
            'Saved coordinates contain spectra that are absent in the feature matrix'

        # Binning rule: колонки согласованы с колонками итоговой матрицы.
        br_path = tmp_path / "feature_matrix_binning_rule.parquet"
        assert br_path.exists(), 'Binning rule parquet was not saved'
        binning_rule = pd.read_parquet(br_path)
        assert set(binning_rule.columns) == \
            {"peak_mz", "bin_left_edge", "bin_right_edge", "pdf_y"}, \
            f'Wrong binning rule columns: {list(binning_rule.columns)}'
        assert set(binning_rule["peak_mz"]) == set(fm.columns), \
            'Binning rule peak m/z values do not match the feature matrix columns'
        assert np.all(binning_rule["bin_left_edge"] < binning_rule["peak_mz"]) and \
            np.all(binning_rule["peak_mz"] < binning_rule["bin_right_edge"]), \
            'Binning rule edges do not bracket the peak m/z'

        # Конфиг агрегации сохранён и перечисляет все сэмплы.
        cfg_path = tmp_path / "feature_matrix_aggregation_configs.yaml"
        assert cfg_path.exists(), 'Aggregation configs yaml was not saved'
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert set(cfg["samples"].keys()) == set(self.ds.sources.keys()), \
            'Aggregation configs do not list all samples of the dataset'

    def test_missing_peaks_density_raises(self):
        """feature_matrix должен падать с понятной ошибкой, если peaks-density файлов нет."""
        ds = DataSet(r"tests\Test_data\imzml_discont")
        hidden = []
        try:
            for source in ds.sources.values():
                density_path = source._datasource.peaks_density_path
                if os.path.exists(density_path):
                    os.rename(density_path, density_path + "_hidden_by_test")
                    hidden.append(density_path)
            with pytest.raises(FileNotFoundError):
                ds.feature_matrix(draw=False, show_stats=False)
        finally:
            for path in hidden:
                os.rename(path + "_hidden_by_test", path)
        ds.close()
