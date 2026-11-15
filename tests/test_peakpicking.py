import copy
import os
import pytest
import numpy as np
import yaml
from pelmesha import DataSet
from pelmesha.serving import Drawer
from pelmesha.cookbook import Configs, KDEConfigs, PipelineConfigurator, PreparedDataSource
from pelmesha.dough import Indexator
from pelmesha.kneading import msalign, modify_raw_spectrum, resample_mz_scale, peakpicker, _build_headers, add_zero_points_to_peaks, smoothing, reduce_signal_to_zero
from pelmesha.utensils import split_pdtable_by_peaks_gap
from pybaselines import Baseline
import inspect
from tests.constants import SIGNAL_DISTORTIONS
from tests.conftest import _generate_spectra, draw_fig
from numpy.polynomial import Polynomial

pytestmark = pytest.mark.peakpicking

SMOOTH_ALGOS = ["GA", "MA", "SG"]
SMOOTH_ALGOS_NONNEGATIVE = {"MA"}   # всё, что строится из неотрицательных весов

def _match_idx(mz_out, mz_in, tol):
    """Индекс ближайшей точки mz_in для каждой точки mz_out (-1, если дальше tol)."""
    pos = np.searchsorted(mz_in, mz_out)
    idx = np.full(mz_out.size, -1, dtype=int)
    for side in (pos - 1, pos):
        valid = (side >= 0) & (side < mz_in.size)
        cand = np.zeros(mz_out.size, dtype=bool)
        cand[valid] = np.abs(mz_out[valid] - mz_in[side[valid]]) <= tol
        idx[cand] = side[cand]
    return idx

def _expected_zero_points(mz, coeffs):
    """Ожидаемые нулевые точки, которые add_zero_points_to_peaks добавит к mz.

    Воспроизводит классификацию промежутков: большие промежутки дают две
    точки (слева и справа), малые — одну, а границы диапазона расширяются
    на один шаг дискретизации (только если есть хотя бы один промежуток).
    """
    model = Polynomial(coeffs)
    d = model(mz[:-1])
    diff = np.diff(mz)
    big = diff > 2.5 * d
    small = (diff > 1.25 * d) ^ big

    expected = []
    for i in np.flatnonzero(big):
        expected.append(mz[i] + d[i])
        expected.append(mz[i + 1] - d[i])
    for i in np.flatnonzero(small):
        expected.append(mz[i] + d[i])
    if expected:
        expected.append(mz[0] - model(mz[0]))
        expected.append(mz[-1] + model(mz[-2]))
    return np.array(expected)

def test_building_metadata(path):
    ds = DataSet(path, rebuild_metadata=True, draw_mz_discretization= False)
    # TODO После обновления интерполяции шкалы дискретизации, добавить сюда проверку того, что при любома m/z в диапазоне нет ни нулевых значений, ни ниже нуля дискретизаций

def test_displaying_dataset(ds):
    """Verify that the internal structure of the DataSet is displayed correctly:
    the ``repr``/``_repr_html_`` of the DataSet itself and of its configs
    (per-ROI processing and KDE configs) are generated without errors and
    reflect the actually stored data.
    """
    assert ds.sources, "The `ds` fixture must provide at least one registered source"

    # ============================ DataSet itself ============================
    ds_repr = repr(ds)
    assert isinstance(ds_repr, str) and len(ds_repr) > 0
    assert str(ds) == ds_repr

    # The summary line must report the real number of files and ROIs
    total_rois = sum(len(source.rois) for source in ds.sources.values())
    assert f"DataSet: {len(ds.sources)} files, {total_rois} ROIs" in ds_repr

    # Parse the text table: [summary, blank, header, separator, data rows...]
    lines = ds_repr.splitlines()
    assert len(lines) >= 5
    headers = [h.strip() for h in lines[2].split(" | ")]
    assert headers == [
        "Sample name", "Mass spectra number", "Continious", "Peaklists",
        "Processed mass spectra", "Peaks density", "Directory", "Previous configs",
    ]
    col = {h: i for i, h in enumerate(headers)}

    # One table row per registered source, with correct internals
    rows = lines[4:]
    assert len(rows) == len(ds.sources)
    for row, (sample_name, source) in zip(rows, ds.sources.items()):
        cells = [c.strip() for c in row.split(" | ")]
        assert cells[col["Sample name"]] == sample_name
        expected_n = Indexator(np.vstack(source.roi_metadata['idxroi'].to_numpy())).count
        assert cells[col["Mass spectra number"]] == str(expected_n)
        assert cells[col["Continious"]] == ("Yes" if source._datasource.dcont else "No")
        assert cells[col["Directory"]] == os.path.dirname(source.file_path)

    # HTML (Jupyter) representation must contain the same content
    html = ds._repr_html_()
    assert isinstance(html, str) and html.startswith("<div")
    assert f"DataSet: {len(ds.sources)} files" in html
    assert all(tag in html for tag in ("<table", "<thead>", "<tbody>", "</table>"))
    for sample_name in ds.sources:
        assert sample_name in html

    # ============================== Configs ==============================
    for sample_name, source in ds.sources.items():
        # DataSource-wrapper-level repr
        src_repr = repr(source)
        assert src_repr.startswith("PreparedDataSource(") and src_repr.endswith(")")
        assert sample_name in src_repr
        assert f"rois: {source.rois}" in src_repr

        # Per-ROI configs must exist for every ROI of the linked datasource
        assert set(source.rois) == set(source.roi_metadata.index)
        assert set(source.roi_configs) == set(source.rois)
        assert set(source.roi_kde_configs) == set(source.rois)

        for roi in source.rois:
            # ---- processing (PipelineConfigurator) config ----
            config = source[roi]
            assert "functions" in config.configs and "methods" in config.configs

            cfg_repr = repr(config)
            assert isinstance(cfg_repr, str) and cfg_repr.startswith("Configs(")
            assert cfg_repr.endswith(")")
            assert "functions:" in cfg_repr or "methods:" in cfg_repr

            cfg_html = config._repr_html_()
            assert isinstance(cfg_html, str) and "Configs" in cfg_html
            if config.configs.get("functions") or config.configs.get("methods"):
                assert "<table" in cfg_html
                # Every configured step must actually be shown in the HTML
                for func_name in config.configs.get("functions", {}):
                    assert func_name in cfg_html
                for cls_name, method_buckets in config.configs.get("methods", {}).items():
                    for method_name in method_buckets:
                        assert method_name in cfg_html

            # ---- KDE config ----
            kde = source.roi_kde_configs[roi]
            kde_repr = repr(kde)
            assert isinstance(kde_repr, str) and kde_repr.startswith("KDEConfigs(")

# --------------------------------------------------------------------------- #
#  Config change coverage: direct, PreparedDataSource, DataSet                 #
# --------------------------------------------------------------------------- #

def test_config_changes_direct(tmp_path):
    """All config change methods applied directly to PipelineConfigurator /
    KDEConfigs instances (no data source required)."""
    # ===================== PipelineConfigurator =====================
    cfg = PipelineConfigurator()

    # --- update(): flat mode (parameter name found across groups) ---
    cfg.update({"smooth_window": 0.15})
    assert cfg["smoothing"]["smooth_window"] == 0.15

    # --- update(): partial nested path (class/function name as key) ---
    cfg.update({"msalign": {"align_shift_range": [-1.0, 1.0]}})
    assert cfg["msalign"]["align_shift_range"] == [-1.0, 1.0]

    # --- update(): structured mode (callable as key) ---
    cfg.update({msalign: {"align_ratio": 0.5}})
    assert cfg["msalign"]["align_ratio"] == 0.5

    # --- update(): full config dict (deep merge, change-only) ---
    cfg.update({"methods": {"Baseline": {"asls": {"params": {"lam": 1e5}}}}})
    assert cfg["Baseline.asls"]["lam"] == 1e5

    # --- update(): keyword arguments ---
    cfg.update(smooth_cycles=3)
    assert cfg["smoothing"]["smooth_cycles"] == 3

    # --- update(): from a YAML file ---
    yaml_path = tmp_path / "overrides.yaml"
    yaml_path.write_text("smooth_window: 0.25\n")
    cfg.update(str(yaml_path))
    assert cfg["smoothing"]["smooth_window"] == 0.25

    # --- __setitem__: dotted raw path ---
    cfg["methods.Baseline.asls.params.lam"] = 2e5
    assert cfg["Baseline.asls"]["lam"] == 2e5

    # --- __setitem__: tuple raw path ---
    cfg[("methods", "Baseline", "asls", "params", "lam")] = 3e5
    assert cfg["Baseline.asls"]["lam"] == 3e5

    # --- delete(): structured path ---
    cfg.delete("msalign.align_ratio")
    assert "align_ratio" not in cfg["msalign"]

    # --- delete(): flat mode (unique parameter) ---
    cfg.delete("smooth_window")
    assert "smooth_window" not in cfg["smoothing"]

    # --- set_method(): replace a method with overrides ---
    cfg.set_method(Baseline, "asls", lam=1e6)
    assert cfg["Baseline.asls"]["lam"] == 1e6

    # --- set_function(): replace a function with overrides ---
    cfg.set_function("msalign", align_ratio=0.7)
    assert cfg["msalign"]["align_ratio"] == 0.7

    # --- replace_config(): full replacement ---
    cfg.replace_config({"functions": {"smoothing": {"smooth_window": 0.5}}})
    assert cfg.configs == {"functions": {"smoothing": {"smooth_window": 0.5}}}

    # --- delete(): entire class / function ---
    cfg2 = PipelineConfigurator()
    cfg2.delete("Baseline")
    assert "Baseline" not in cfg2.configs.get("methods", {})
    cfg2.delete("msalign")
    assert "msalign" not in cfg2.configs.get("functions", {})

    # ========================= KDEConfigs =========================
    kde = KDEConfigs()
    kde.update(bwc=2.0, split_peaks_min=50)
    assert kde.bwc == 2.0
    assert kde.split_peaks_min == 50

    kde["KD_bandwidth"] = "silverman"
    assert kde.KD_bandwidth == "silverman"

    kde_copy = kde.with_overrides(bwc=3.0)
    assert kde.bwc == 2.0          # original untouched
    assert kde_copy.bwc == 3.0     # copy carries the override


def test_method_swap_and_disable_class(tmp_path):
    """Class-named method-swap kwargs, disabling a whole class with None,
    None-safety guards, repr/HTML rendering of disabled classes, and YAML
    round-trip of null entries."""
    # --- constructor class-named kwarg: swap the method ---
    cfg = PipelineConfigurator(Baseline='snip')
    baseline_methods = cfg.configs['methods']['Baseline']
    assert isinstance(baseline_methods, dict) and 'snip' in baseline_methods
    assert 'asls' not in baseline_methods           # old method replaced
    assert isinstance(cfg['Baseline.snip'], dict)   # defaults re-extracted

    # --- update() dict form: swap the method again ---
    cfg.update({'Baseline': 'asls'})
    assert 'asls' in cfg.configs['methods']['Baseline']
    assert 'snip' not in cfg.configs['methods']['Baseline']

    # --- item assignment: swap the method (simplest form) ---
    cfg['Baseline'] = 'snip'
    assert 'snip' in cfg.configs['methods']['Baseline']
    assert 'asls' not in cfg.configs['methods']['Baseline']

    # --- item assignment: disable the whole class ---
    cfg['Baseline'] = None
    assert cfg.configs['methods']['Baseline'] is None

    # --- item assignment: re-enable ---
    cfg['Baseline'] = 'asls'
    assert 'asls' in cfg.configs['methods']['Baseline']

    # --- update() dict form: disable the whole class ---
    cfg.update({'Baseline': None})
    assert cfg.configs['methods']['Baseline'] is None

    # --- None-safety of __getitem__ / get ---
    with pytest.raises(KeyError):
        cfg['Baseline.snip']
    with pytest.raises(KeyError):
        cfg[('Baseline', 'snip')]
    assert cfg.get('Baseline') is None
    assert cfg.get(('Baseline', 'snip')) is None

    # --- partial-path update against a disabled class warns and keeps None ---
    with pytest.warns(UserWarning, match="disabled"):
        cfg.update({'Baseline': {'asls': {'lam': 1e5}}})
    assert cfg.configs['methods']['Baseline'] is None

    # --- delete inside a disabled class warns and keeps None ---
    with pytest.warns(UserWarning, match="disabled"):
        cfg.delete('Baseline.snip')
    assert cfg.configs['methods']['Baseline'] is None

    # --- get_step_configs silently skips the disabled class ---
    pre_cfg = cfg.get_step_configs('preprocess')
    assert 'Baseline' not in pre_cfg.configs.get('methods', {})

    # --- repr / HTML rendering of the disabled class ---
    rep = repr(cfg)
    assert 'Baseline: None (disabled' in rep
    assert 'snip' in rep        # available-methods list in the repr text

    html = cfg._repr_html_()
    assert 'None (disabled' in html
    assert 'Available methods' in html
    assert 'snip' in html

    # --- disable_class() on an unknown class warns ---
    with pytest.warns(UserWarning, match="not found"):
        cfg.disable_class('NoSuchClass')

    # --- re-enable via update() ---
    cfg.update({'Baseline': 'snip'})
    assert 'snip' in cfg.configs['methods']['Baseline']
    assert isinstance(cfg['Baseline.snip'], dict)

    # --- YAML round-trip preserves the null (disabled) entry ---
    cfg2 = PipelineConfigurator()
    cfg2.update({'Baseline': None})
    cfg2.dump(str(tmp_path))  # writes tmp_path/processing_recipe.yaml (+ .py)
    cfg_rt = PipelineConfigurator(str(tmp_path / 'processing_recipe.yaml'))
    assert cfg_rt.configs['methods']['Baseline'] is None

    # --- structure validation accepts None as a disabled class ---
    Configs._validate_configs_structure({'methods': {'Baseline': None}})


def test_config_changes_prepared_datasource(ds, tmp_path):
    """ROI-aware config changes through PreparedDataSource: update /
    update_kde / delete applied to a single ROI or to all ROIs."""
    sample_name, source = next(iter(ds.sources.items()))
    assert source.rois, "The source must have at least one ROI"
    roi = source.rois[0]

    # Snapshot the pristine configs so the shared session-scoped `ds`
    # fixture is left untouched after the test.
    roi_configs_snapshot = copy.deepcopy(source.roi_configs)
    kde_configs_snapshot = copy.deepcopy(source.roi_kde_configs)
    try:
        # --- update(): single ROI only ---
        original_smooth = {r: source[r]["smoothing"]["smooth_window"] for r in source.rois}
        source.update({"smooth_window": 0.15}, rois=[roi])
        assert source[roi]["smoothing"]["smooth_window"] == 0.15
        for other in source.rois:
            if other != roi:
                assert source[other]["smoothing"]["smooth_window"] == original_smooth[other]

        # --- update(): all ROIs ---
        source.update({"smooth_window": 0.2})
        for r in source.rois:
            assert source[r]["smoothing"]["smooth_window"] == 0.2

        # --- update(): keyword arguments for a single ROI ---
        original_cycles = {r: source[r]["smoothing"]["smooth_cycles"] for r in source.rois}
        source.update(smooth_cycles=4, rois=[roi])
        assert source[roi]["smoothing"]["smooth_cycles"] == 4
        for other in source.rois:
            if other != roi:
                assert source[other]["smoothing"]["smooth_cycles"] == original_cycles[other]

        # --- update_kde(): single ROI only ---
        original_bwc = {r: source.roi_kde_configs[r].bwc for r in source.rois}
        source.update_kde(bwc=2.5, rois=[roi])
        assert source.roi_kde_configs[roi].bwc == 2.5
        for other in source.rois:
            if other != roi:
                assert source.roi_kde_configs[other].bwc == original_bwc[other]

        # --- update_kde(): all ROIs ---
        source.update_kde(split_peaks_min=50)
        for r in source.rois:
            assert source.roi_kde_configs[r].split_peaks_min == 50

        # --- update_kde(): from a YAML file (regression: the loaded
        #     config used to be discarded because load_yaml returns a
        #     new instance) ---
        kde_override_path = tmp_path / "kde_override.yaml"
        kde_override_path.write_text("bwc: 4.0\nsplit_peaks_min: 60\n")
        source.update_kde(str(kde_override_path), rois=[roi])
        assert source.roi_kde_configs[roi].bwc == 4.0
        assert source.roi_kde_configs[roi].split_peaks_min == 60
        for other in source.rois:
            if other != roi:
                assert source.roi_kde_configs[other].bwc == original_bwc[other]

        # --- delete(): single ROI only ---
        source.delete("smooth_window", rois=[roi])
        assert "smooth_window" not in source[roi]["smoothing"]
        for other in source.rois:
            if other != roi:
                assert "smooth_window" in source[other]["smoothing"]

        # --- direct per-ROI access: source[roi] is a PipelineConfigurator ---
        source[roi].update({"msalign": {"align_ratio": 0.5}})
        assert source[roi]["msalign"]["align_ratio"] == 0.5

        # --- setdefault(): returns the existing config for a known ROI ---
        assert source.setdefault(roi) is source[roi]

        # --- __getitem__(): unknown ROI raises KeyError ---
        with pytest.raises(KeyError):
            source["__no_such_roi__"]
    finally:
        source.roi_configs = roi_configs_snapshot
        source.roi_kde_configs = kde_configs_snapshot


def test_config_changes_dataset(ds, tmp_path):
    """DataSet-level config handling: change detection via
    check_config_changes, config application via add_sources, and
    save/load round-trips via dump / dump_kde_configs."""
    sample_name, source = next(iter(ds.sources.items()))

    # ============ check_config_changes: statuses ============
    config_yaml = os.path.join(
        os.path.dirname(source.file_path), "raw_pelmesha", "config.yaml"
    )
    if os.path.exists(config_yaml):
        os.remove(config_yaml)
    try:
        # No saved config -> "new"
        changes = ds.check_config_changes()
        assert set(changes) == set(ds.sources)
        assert changes[sample_name]["status"] == "new"
        assert changes[sample_name]["old"] is None

        # Saved config equal to current (empty) config -> "unchanged"
        with open(config_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({}, f)
        changes = ds.check_config_changes()
        assert changes[sample_name]["status"] == "unchanged"

        # Saved config different from current config -> "changed"
        saved = {"functions": {"smoothing": {"smooth_window": 0.5}}}
        with open(config_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(saved, f)
        changes = ds.check_config_changes()
        assert changes[sample_name]["status"] == "changed"
        assert changes[sample_name]["old"] == saved
    finally:
        if os.path.exists(config_yaml):
            os.remove(config_yaml)

    # ============ add_sources: apply a config dict to a new source ============
    fresh_ds = DataSet(None)
    config_dict = {"functions": {"smoothing": {"smooth_window": 0.33}}}
    fresh_ds.add_sources({source.file_path: config_dict})
    added = next(iter(fresh_ds.sources.values()))
    assert added[added.rois[0]]["smoothing"]["smooth_window"] == 0.33

    # ============ save/load round-trip (explicit paths, no pollution) ============
    recipe_path = str(tmp_path / "recipe.yaml")
    source.dump(recipe_path)
    assert os.path.exists(recipe_path)
    loaded = PreparedDataSource(configs_source=recipe_path)
    assert set(loaded.rois) == set(source.rois)
    for r in source.rois:
        assert loaded[r].configs == source[r].configs

    kde_path = str(tmp_path / "kde.yaml")
    source.dump_kde_configs(kde_path)
    assert os.path.exists(kde_path)
    with open(kde_path, encoding="utf-8") as f:
        kde_data = yaml.safe_load(f)
    assert set(kde_data) == set(source.rois)
    for r in source.rois:
        assert kde_data[r]["bwc"] == source.roi_kde_configs[r].bwc

class TestAudit_Processing:
    @pytest.fixture(autouse=True)
    def setup_draw_mock(self, monkeypatch,request):
        patched_draw = draw_fig(Drawer.audit_processing)
        patched_draw.current_request = request
        monkeypatch.setattr(Drawer, "audit_processing", patched_draw)
    
    def test_draw(self, ds):
        ds.audit_processing(draw_mz_range=(750,800))


def _in_segments_mask(mz, segments):
    mask = np.zeros(mz.size, dtype=bool)
    for s, e in segments:
        mask |= (mz > s) & (mz < e)
    return mask

class TestProcessingSteps:
    #synthetic tests
    const_coeffs = [1.0]
    def test_inserts_exactly_where_expected(self):
        # шаг 1.0 везде; промежутки: 1.5 (small), 1.5 (small), 3.0 (big)
        mz = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 5.5, 7.0, 10.0])
        ints = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

        out_mz, out_ints = add_zero_points_to_peaks(mz, ints, self.const_coeffs)

        expected_mz = np.array([-1.0, 0, 1, 2, 3, 4.0, 4.5, 5.5, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0])
        expected_ints = np.array([0.0, 10, 20, 30, 40, 0, 50, 60, 0, 70, 0, 0, 80, 0])

        np.testing.assert_allclose(out_mz, expected_mz)
        np.testing.assert_allclose(out_ints, expected_ints)
    def test_no_gaps_returns_input_unchanged(self):
        mz = np.arange(10.0)
        ints = np.arange(10.0)
        out_mz, out_ints = add_zero_points_to_peaks(mz, ints, self.const_coeffs)
        np.testing.assert_allclose(out_mz, mz)
        np.testing.assert_allclose(out_ints, ints)   # границ -1.0 и 10.0 тут НЕТ
        

    @pytest.mark.parametrize(
        "resample_mz_step, resample_num_points",
        [(None, None), (None, 200000), (0.0001, None)],
        ids=["no-args", "num-points", "step"],
    )
    def test_resample_mz_scale(self,
                               ds,
                               resample_mz_step,
                               resample_num_points):
        for source_name, source in ds.sources.items():
            for roi in source.rois:
                mz_min, mz_max = source.roi_metadata.loc[roi, 'mz_range']
                resampled_mz = resample_mz_scale(mz_min, mz_max, resample_mz_step, resample_num_points)
                if resample_mz_step is not None or resample_num_points is not None:
                    assert resampled_mz is not None
                    assert len(resampled_mz) == resample_num_points or len(resampled_mz) == int(np.round((mz_max - mz_min)/resample_mz_step)) + 1
                else:
                    assert resampled_mz is None

    @pytest.mark.parametrize("mz_crop_range", [None, (250,900)])
    @pytest.mark.parametrize("zero_points_to_peaks_ext", [False, True])
    @pytest.mark.parametrize("mz_segments_to_zero", [None, SIGNAL_DISTORTIONS])
    def test_modify_raw_spectrum(self,
                                 ds,
                                 mz_crop_range,
                                 zero_points_to_peaks_ext,
                                 mz_segments_to_zero):
        """Проверка modify_raw_spectrum: обрезка, обнуление сегментов и
        добавление нулевых точек сверяются с независимо рассчитанным
        ожидаемым результатом.
        """
        tol = 1e-6
        for source_name, roi, spectra_ind, mz, intens in _generate_spectra(ds):
            coeffs = ds[source_name].roi_metadata.loc[roi, 'discret_coeffs']
            # get_spectrum может вернуть read-only массивы — берём копии,
            # чтобы не мутировать данные источника
            mz = np.array(mz, copy=True)
            intens = np.array(intens, copy=True)

            mod_mz, mod_intens = modify_raw_spectrum(mz, intens, coeffs,
                                                     mz_crop_range,
                                                     zero_points_to_peaks_ext,
                                                     mz_segments_to_zero)

            # 1. Обрезка: база — в точности исходные точки, попавшие в границы
            if mz_crop_range is not None:
                low, high = mz_crop_range
                base_mask = (mz >= low) & (mz <= high)
            else:
                base_mask = np.ones(mz.size, dtype=bool)
            base_mz, base_ints = mz[base_mask], intens[base_mask]

            # 2. Обнуление сегментов: внутри сегментов должны быть нули
            in_seg = (_in_segments_mask(base_mz, mz_segments_to_zero)
                      if mz_segments_to_zero is not None
                      else np.zeros(base_mz.size, dtype=bool))
            ref_ints = np.where(in_seg, 0.0, base_ints)

            # 3. Нулевые точки: шкала результата — базовые точки плюс
            #    ожидаемые новые (в промежутках и на границах шкалы)
            expected_pts = (_expected_zero_points(base_mz, coeffs)
                            if zero_points_to_peaks_ext else np.array([]))
            expected_mz = (np.sort(np.concatenate([base_mz, expected_pts]))
                           if expected_pts.size else base_mz)

            np.testing.assert_array_equal(mod_mz, expected_mz)
            assert mod_mz.size == base_mz.size + expected_pts.size

            # Интенсивности: новые точки — нули, базовые — с учётом обнуления
            orig_pos = _match_idx(mod_mz, base_mz, tol)
            is_new = orig_pos < 0
            np.testing.assert_array_equal(mod_intens[is_new],
                                          np.zeros(is_new.sum()))
            np.testing.assert_array_equal(mod_intens[~is_new],
                                          ref_ints[orig_pos[~is_new]])

            # Каждая ожидаемая новая точка присутствует в результате
            if expected_pts.size:
                covered = _match_idx(expected_pts, mod_mz, tol)
                np.testing.assert_array_equal(covered >= 0,
                                              np.ones(expected_pts.size, bool))

            # Идемпотентность обнуления сегментов
            if mz_segments_to_zero is not None:
                _, again = reduce_signal_to_zero(mod_mz, mod_intens.copy(),
                                                 mz_segments_to_zero)
                np.testing.assert_array_equal(again, mod_intens)

    @pytest.mark.parametrize("smooth_algo",SMOOTH_ALGOS)
    def test_output_length(self, smooth_algo):
        y = np.random.default_rng(0).uniform(size=137)
        assert len(smoothing(y, smooth_algo, 7, 2)) == 137

    @pytest.mark.parametrize("smooth_algo",SMOOTH_ALGOS)
    def test_input_not_mutated(self, smooth_algo):
        y = np.random.default_rng(0).uniform(size=50)
        y_before = y.copy()
        smoothing(y, smooth_algo, 7, 3)
        np.testing.assert_array_equal(y, y_before)

    @pytest.mark.parametrize("smooth_algo", SMOOTH_ALGOS)
    def test_constant_signal_invariant(self, smooth_algo):
        y = np.full(50, 3.14)
        out = smoothing(y, smooth_algo, 9, 5)
        np.testing.assert_allclose(out, 3.14)
    
    @pytest.mark.parametrize("smooth_algo", SMOOTH_ALGOS)
    def test_even_window_behaves_like_next_odd(self, smooth_algo):
        y = np.random.default_rng(1).uniform(size=100)
        assert np.array_equal(smoothing(y, smooth_algo, 8), smoothing(y, smooth_algo, 9))

    @pytest.mark.parametrize("smooth_algo", SMOOTH_ALGOS)
    def test_peak_is_reduced_but_present(self, smooth_algo):
        y = np.zeros(50)
        y[25] = 100.0
        out = smoothing(y, smooth_algo, 7)
        assert out[25] < 100.0          # пик сгладился
        assert out[25] > 0.0            # но не пропал
        assert abs(out - y).max() < 100.0   # артефакты много меньше самой вершины
        if smooth_algo in SMOOTH_ALGOS_NONNEGATIVE:
            assert np.all(out >= 0.0)   # только для них честно
        else:
            # SG: допустим undershoot, но ограниченный — не должен улетать
            assert np.all(out >= -0.3 * y.max())

    def test_more_cycles_smooths_more(self):
        rng = np.random.default_rng(2)
        y = rng.normal(0, 1, 200)
        r1 = np.var(smoothing(y, 'MA', 7, 1))
        r3 = np.var(smoothing(y, 'MA', 7, 3))
        assert r3 < r1          # дисперсия падает с числом проходов
    def test_empty_array(self):
        assert len(smoothing(np.array([]), 'MA')) == 0

    @pytest.mark.parametrize("smooth_algo", SMOOTH_ALGOS)
    def test_short_array_returns_copy(self, smooth_algo):
        y = np.array([1.0, 2.0, 3.0])          # короче окна 7
        out = smoothing(y, smooth_algo, 7)
        np.testing.assert_array_equal(out, y)
        assert out is not y                    # именно копия, а не тот же объект


    def test_window_2_behaves_like_3(self):
        y = np.random.default_rng(3).uniform(size=50)
        np.testing.assert_array_equal(smoothing(y, 'MA', 2),
                                    smoothing(y, 'MA', 3))

    def test_window_1_noop(self):
        y = np.random.default_rng(3).uniform(size=50)
        np.testing.assert_array_equal(smoothing(y, 'MA', 1), y)

    def test_unknown_algo_returns_input(self, capsys):
        y = np.random.default_rng(4).uniform(size=30)
        out = smoothing(y, 'Chokobo')
        np.testing.assert_array_equal(out, y)
        captured = capsys.readouterr()
        assert "not recognized" in captured.out

    @pytest.mark.parametrize(
    "align_peaks, align_method",
    [
        (None, "pchip"),                # без пиков — метод не важен, берём дефолт
        ([128, 768, 769], "pchip"),
        ([128, 768, 769], "slinear"),
        ([128, 768, 769], "cubic")
    ],
    ids=["no-peaks", "peaks-pchip", "peaks-linear", "peaks-cubic"],
)

    def test_alignment_step(self,
                            ds,
                            align_peaks,
                            align_method):
        changed_any = False
        for source_name, roi, rand_idx, mz, intens in _generate_spectra(ds):
            aligned = msalign(mz, intens, align_peaks, align_method)
            assert isinstance(aligned, np.ndarray)
            assert aligned.shape == intens.shape, f"{source_name}/{roi}/{rand_idx}"
            assert np.all(np.isfinite(aligned)), f"{source_name}/{roi}/{rand_idx}: NaN/inf"
            assert aligned.max() <= intens.max() * 1.2
            if align_peaks is not None:
                assert not np.array_equal(aligned, intens)
                changed_any |= not np.allclose(aligned, intens, atol=1e-12)
        if align_peaks is not None:
            assert changed_any, "alignment did nothing on any spectrum"

    VALID_COMBOS = (
    # SNR_threshold, noise_mz_width, noise_est_iter
    pytest.param(None, 9, 3, id="snr-default"),
    pytest.param(5, 5, 1, id="snr5-w5-i1"),
    pytest.param(5, 5, 3, id="snr5-w5-i3"),
    pytest.param(5, 9, 1, id="snr5-w9-i1"),
    pytest.param(5, 9, 3, id="snr5-w9-i3"),
    pytest.param(5, 30, 1, id="snr5-w30-i1"),
    pytest.param(5, 30, 3, id="snr5-w30-i3"),
    pytest.param(9, 5, 1, id="snr9-w5-i1"),
    pytest.param(9, 5, 3, id="snr9-w5-i3"),
    pytest.param(9, 9, 1, id="snr9-w9-i1"),
    pytest.param(9, 9, 3, id="snr9-w9-i3"),
    pytest.param(9, 30, 1, id="snr9-w30-i1"),
    pytest.param(9, 30, 3, id="snr9-w30-i3"),
    pytest.param(30, 5, 1, id="snr30-w5-i1"),
    pytest.param(30, 5, 3, id="snr30-w5-i3"),
    pytest.param(30, 9, 1, id="snr30-w9-i1"),
    pytest.param(30, 9, 3, id="snr30-w9-i3"),
    pytest.param(30, 30, 1, id="snr30-w30-i1"),
    pytest.param(30, 30, 3, id="snr30-w30-i3"),
    )


    @pytest.mark.parametrize("SNR_threshold, noise_mz_width, noise_est_iter", VALID_COMBOS)
    def test_peakpicker_SNR(self,
                            ds,
                            SNR_threshold,
                            noise_mz_width,
                            noise_est_iter):
        peaklists = {}
        headers = _build_headers(False, SNR_threshold)
        mz_idx = headers.index("mz")
        intens_idx = headers.index("Intensity")
        pextl_idx = headers.index("PextL")
        pextr_idx = headers.index("PextR")
        # колонки SNR/Noise/Mean noise существуют только при SNR_threshold != None
        if SNR_threshold is not None:
            SNR_idx = headers.index("SNR")
            mnoise_idx = headers.index("Mean noise")
            noise_idx = headers.index("Noise")
        for numspec, (source_name, roi, spectra_ind, mz, intens) in enumerate(_generate_spectra(ds)):
            
            peaklist = peakpicker(mz,
                                  intens,
                                  spectra_ind,
                                  SNR_threshold=SNR_threshold,
                                  noise_mz_width=noise_mz_width, 
                                  noise_est_iter=noise_est_iter, 
                                  headers=headers)
            peaklists[numspec] = peaklist
            if SNR_threshold is not None:
                assert np.all(peaklist[:,SNR_idx] >= SNR_threshold), f'Пики не отфильтрованны корректно по SNR. File {source_name}, roi {roi}, idx {spectra_ind}'
                assert np.all((peaklist[:,intens_idx]-peaklist[:,mnoise_idx])/(peaklist[:,noise_idx]) - peaklist[:,SNR_idx] == 0), "Назначенные SNR пиков отличаются от рассчётных (intensity - mean noise)/std_noise f'Пики не отфильтрованны корректно по SNR. File {source_name}, roi {roi}, idx {spectra_ind}'"
            assert np.all(peaklist[:, mz_idx] >= mz.min()) and np.all(peaklist[:, mz_idx] <= mz.max())
            assert np.all(peaklist[:, intens_idx] <= intens.max() * 1.05)
            assert np.all(peaklist[:, pextl_idx] < peaklist[:, pextr_idx])        
            assert np.all(peaklist[:, pextl_idx] >= mz.min() - 1e-6)
        peaklists = np.vstack(tuple(peaklists.values()))
        assert len(headers) == peaklists.shape[1], "Кол-во колонок не соответствует кол-ву ожидаемых столбцов"

    @pytest.mark.parametrize('fwhm_min',[None, 0.001, 2])
    @pytest.mark.parametrize('fwhm_max',[None, 0.001, 2])
    def test_fwhm_filter(self, ds, fwhm_min, fwhm_max):
        headers = _build_headers(False, None)
        fwhm_idx = headers.index("FWHM")
        for numspec, (source_name, roi, spectra_ind, mz, intens) in enumerate(_generate_spectra(ds)):
            res = peakpicker(mz,
                             intens,
                             spectra_ind,
                             SNR_threshold = None,
                             fwhm_min=fwhm_min,
                             fwhm_max=fwhm_max,
                             headers=headers)
            
            if fwhm_min is not None and fwhm_max is not None:
                assert np.all((res[:, fwhm_idx] >= fwhm_min) & (res[:, fwhm_idx] <= fwhm_max))
            else:
                if fwhm_min is not None:
                    assert np.all(res[:, fwhm_idx] >= fwhm_min)
                if fwhm_max is not None:
                    assert np.all(res[:, fwhm_idx] <= fwhm_max)
    @pytest.mark.parametrize('heightfilter',[None,10000])
    @pytest.mark.parametrize('rel_heightfilter',[None,0.01])
    def test_heightfilters(self,
                           ds,
                           heightfilter,
                           rel_heightfilter):
        headers = _build_headers(False, None)
        intens_idx = headers.index("Intensity")
        for numspec, (source_name, roi, spectra_ind, mz, intens) in enumerate(_generate_spectra(ds)):
            rel_heightfilter_loc = -np.inf
            res = peakpicker(mz,
                             intens,
                             spectra_ind,
                             SNR_threshold = None,
                             heightfilter=heightfilter,
                             rel_heightfilter = rel_heightfilter,
                             headers=headers)
            
            if rel_heightfilter is not None:
                rel_heightfilter_loc = intens.max()*rel_heightfilter
            # эффективный порог = максимум из heightfilter и rel_heightfilter;
            # если оба None — фильтрации нет (порог -inf)
            heightfilter_loc = heightfilter if heightfilter is not None else rel_heightfilter_loc
            if heightfilter_loc < rel_heightfilter_loc:
                heightfilter_loc = rel_heightfilter_loc
            assert np.all(res[:, intens_idx] >= heightfilter_loc)


# ============================================================================
#  peakpicker — проверка параметров по отдельности.
#  Запускается на случайных спектрах каждого типа данных (imzml_cont /
#  imzml_discont / cdf — это даёт фикстура `ds` + генератор _generate_spectra).
#  Почему headers строятся через _build_headers: выход пиклиста собирается
#  строго по переданному `headers` (`props[h] for h in headers`). Колонки
#  Area/SNR/Noise/Mean noise существуют в props только при return_areas=True
#  и SNR_threshold != None, поэтому несогласованный headers даст KeyError —
#  это контракт API, и тесты обязаны его соблюдать.
# ============================================================================

def _assert_peaklist_well_formed(peaklist, headers, mz, intens, where=""):
    """Общие проверки таблицы пиклиста: форма, отсутствие NaN, границы m/z,
    согласованность PextL/PextR и интенсивностей. Пустой пиклист (0 пиков) —
    допустимый результат и возвращается сразу после проверки формы."""
    idx = {h: i for i, h in enumerate(headers)}
    assert isinstance(peaklist, np.ndarray), f"{where}: peaklist не ndarray"
    assert peaklist.ndim == 2 and peaklist.shape[1] == len(headers), \
        f"{where}: форма {peaklist.shape}, ожидалось (n, {len(headers)})"
    if peaklist.shape[0] == 0:
        return  # пустой пиклист — валидный результат peakpicker
    assert np.all(np.isfinite(peaklist)), f"{where}: есть NaN/inf в пиклисте"
    assert np.all(peaklist[:, idx["mz"]] >= mz.min() - 1e-6), f"{where}: mz левее спектра"
    assert np.all(peaklist[:, idx["mz"]] <= mz.max() + 1e-6), f"{where}: mz правее спектра"
    assert np.all(peaklist[:, idx["PextL"]] < peaklist[:, idx["PextR"]]), f"{where}: PextL >= PextR"
    assert np.all(peaklist[:, idx["mz"]] >= peaklist[:, idx["PextL"]]), f"{where}: центр левее PextL"
    assert np.all(peaklist[:, idx["mz"]] <= peaklist[:, idx["PextR"]]), f"{where}: центр правее PextR"
    assert np.all(peaklist[:, idx["Intensity"]] >= 0), f"{where}: отрицательная интенсивность"
    assert np.all(peaklist[:, idx["Intensity"]] <= intens.max() * 1.05 + 1e-9), \
        f"{where}: интенсивность выше исходного спектра"
    if "Area" in idx:
        assert np.all(peaklist[:, idx["Area"]] >= -1e-9), f"{where}: отрицательная площадь пика"

class TestPeakpicker:
    @pytest.mark.parametrize("return_areas", [True, False], ids=["areas-on", "areas-off"])
    def test_peakpicker_return_areas(self, ds, return_areas):
        """Проверка return_areas. Почему: параметр управляет колонкой Area —
        при True она обязана быть (и быть посчитанной, т.е. неотрицательной),
        при False колонки нет вовсе. Дополнительно проверяется контракт: набор
        колонок в точности равен _build_headers(return_areas, None)."""
        for num, (source_name, roi, spectra_ind, mz, intens) in enumerate(_generate_spectra(ds)):
            where = f"{source_name}/{roi}/#{spectra_ind}"
            headers = _build_headers(return_areas=return_areas, snr_threshold=None)
            peaklist = peakpicker(mz, intens, spectra_ind,
                                SNR_threshold=None,
                                return_areas=return_areas,
                                headers=headers)
            _assert_peaklist_well_formed(peaklist, headers, mz, intens, where)
            assert ("Area" in headers) == return_areas, where


    @pytest.mark.parametrize("fwhm_merge_factor", [None, 0.85, 1.5],
                            ids=["merge-none", "merge-0.85", "merge-1.5"])
    def test_peakpicker_fwhm_merge_factor(self, ds, fwhm_merge_factor):
        """Проверка fwhm_merge_factor. Почему: мерж объединяет пики ближе, чем
        FWHM*factor, и потому НЕ может создавать новые пики — с фактором пиков
        должно быть не больше, чем без него (сравниваем два прогона одного
        спектра). Плюс `total > 0` в конце: защита от «зелёного» теста, в котором
        пиков не нашлось вообще (на урезанных данных это реальный риск)."""
        headers = _build_headers(return_areas=False, snr_threshold=None)
        total = 0
        for num, (source_name, roi, spectra_ind, mz, intens) in enumerate(_generate_spectra(ds)):
            where = f"{source_name}/{roi}/#{spectra_ind}"
            res = peakpicker(mz, intens, spectra_ind,
                            SNR_threshold=None,
                            fwhm_merge_factor=fwhm_merge_factor,
                            headers=headers)
            _assert_peaklist_well_formed(res, headers, mz, intens, where)
            if fwhm_merge_factor is not None:
                base = peakpicker(mz, intens, spectra_ind,
                                SNR_threshold=None, fwhm_merge_factor=None,
                                headers=headers)
                assert res.shape[0] <= base.shape[0], \
                    f"{where}: мерж {fwhm_merge_factor} увеличил число пиков {base.shape[0]} -> {res.shape[0]}"
            total += res.shape[0]
        assert total > 0, "Ни на одном случайном спектре не найдено пиков — проверьте тестовые данные"


    @pytest.mark.parametrize("peaklocation", [1, 0.75], ids=["pl-1.0", "pl-0.75"])
    def test_peakpicker_peaklocation(self, ds, peaklocation):
        """Проверка peaklocation (доля высоты пика, на которой считается центр).
        Почему именно такой инвариант: точное положение центра при 0.75 на
        реальных данных предсказать нельзя, но центр ОБЯЗАН оставаться внутри
        базовых точек пика (PextL, PextR) — это и проверяем через хелпер."""
        headers = _build_headers(return_areas=False, snr_threshold=None)
        for num, (source_name, roi, spectra_ind, mz, intens) in enumerate(_generate_spectra(ds)):
            where = f"{source_name}/{roi}/#{spectra_ind}"
            res = peakpicker(mz, intens, spectra_ind,
                            SNR_threshold=None,
                            peaklocation=peaklocation,
                            headers=headers)
            _assert_peaklist_well_formed(res, headers, mz, intens, where)

    # Три набора параметров:
    #   1) все параметры выключены (None / False);
    #   2) всё на дефолтах peakpicker — вызываем с пустым словарём kwargs;
    #   3) все параметры включены с разумными значениями.
    # Главный критерий сценариев — ничего не упало и таблицы пиклистов корректны.

    PEAKPICKER_SCENARIOS = [
        pytest.param(
            dict(SNR_threshold=None, fwhm_min=None, fwhm_max=None, fwhm_merge_factor=None,
                heightfilter=None, rel_heightfilter=None,
                peaklocation=1, return_areas=False),
            id="all-off",
        ),
        pytest.param(
            {},  # вообще ничего не передаём — все дефолтные параметры peakpicker
            id="all-default",
        ),
        pytest.param(
            dict(SNR_threshold=5.0, noise_mz_width=9.0, noise_est_iter=3,
                fwhm_min=0.001, fwhm_max = 1.0, fwhm_merge_factor=0.85,
                heightfilter=1000.0, rel_heightfilter=0.01,
                peaklocation=0.75, return_areas=True),
            id="all-on",
        ),
    ]


    @pytest.mark.parametrize("scenario_kwargs", PEAKPICKER_SCENARIOS)
    def test_peakpicker_scenarios_all_data(self,ds, scenario_kwargs):
        headers = _build_headers(return_areas=scenario_kwargs.get("return_areas", True),
                                snr_threshold=scenario_kwargs.get("SNR_threshold", 3.5))
        for num, (source_name, roi, spectra_ind, mz, intens) in enumerate(_generate_spectra(ds)):
            where = f"{source_name}/{roi}/#{spectra_ind}"
            res = peakpicker(mz, intens, spectra_ind, headers=headers, **scenario_kwargs)
            _assert_peaklist_well_formed(res, headers, mz, intens, where)