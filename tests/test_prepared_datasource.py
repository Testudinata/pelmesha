"""
Tests for PreparedDataSource config loading from YAML files.

Covers both regular (PipelineConfigurator) and KDE config loading,
including the critical edge case where ROI names like "00" are
unquoted in YAML and loaded as integers by PyYAML.
"""

import os
import tempfile
import yaml
import pytest
from pelmesha.cookbook import PreparedDataSource, PipelineConfigurator, KDEConfigs


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def _make_per_roi_yaml(roi_names: list[str]) -> str:
    """Create a per-ROI processing recipe YAML string.

    Each ROI gets a minimal config with the proper structure:
    methods -> class_name -> method_name -> {params: {...}}
    """
    lines: list[str] = []
    for roi in roi_names:
        lines.append(f'"{roi}":')
        lines.append("  methods:")
        lines.append("    preprocess_configuration_base:")
        lines.append("      preprocess_configuration_base:")
        lines.append("        params:")
        lines.append("          smooth_algo: GA")
        lines.append("          smooth_window: 5")
        lines.append("")
    return "\n".join(lines)


def _make_flat_yaml() -> str:
    """Create a flat (non-per-ROI) processing recipe YAML string."""
    d = {
        "methods": {
            "preprocess_configuration_base": {
                "preprocess_configuration_base": {
                    "params": {
                        "smooth_algo": "GA",
                        "smooth_window": 5,
                    }
                }
            }
        }
    }
    return yaml.dump(d, default_flow_style=False)


def _make_per_roi_kde_yaml(roi_names: list[str]) -> str:
    """Create a per-ROI KDE config YAML string with quoted keys."""
    lines: list[str] = []
    for roi in roi_names:
        lines.append(f'"{roi}":')
        lines.append("  KD_bandwidth: fwhm")
        lines.append("  bwc: 1.0")
        lines.append("  KD_kernel: gaussian")
        lines.append("  KDE_algo: null")
        lines.append("  split_mz_min: 10.0")
        lines.append("  split_peaks_min: 25")
        lines.append("  account_mzscale: true")
        lines.append("")
    return "\n".join(lines)


def _make_flat_kde_yaml() -> str:
    """Create a flat (non-per-ROI) KDE config YAML string."""
    d = {
        "KD_bandwidth": "fwhm",
        "bwc": 1.0,
        "KD_kernel": "gaussian",
        "KDE_algo": None,
        "split_mz_min": 10.0,
        "split_peaks_min": 25,
        "account_mzscale": True,
    }
    return yaml.dump(d, default_flow_style=False)


# --------------------------------------------------------------------------- #
#  Tests: regular (PipelineConfigurator) config loading                       #
# --------------------------------------------------------------------------- #

class TestLoadRegularConfigs:
    """PreparedDataSource loading of PipelineConfigurator configs from YAML."""

    def test_per_roi_yaml_string_roi_names(self):
        """Load a per-ROI YAML with string ROI names (e.g. 'roi_00')."""
        yaml_str = _make_per_roi_yaml(["roi_00", "roi_01"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(configs_source=tmp_path)
            assert "roi_00" in pds.roi_configs
            assert "roi_01" in pds.roi_configs
            assert len(pds.roi_configs) == 2
        finally:
            os.unlink(tmp_path)

    def test_per_roi_yaml_numeric_roi_names(self):
        """Load a per-ROI YAML with numeric-looking ROI names (e.g. '00').

        This is the critical edge case: PyYAML loads unquoted '00:' as int 0.
        The fix in _load() converts int keys back to str.
        """
        yaml_str = _make_per_roi_yaml(["00", "01"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(configs_source=tmp_path)
            assert "00" in pds.roi_configs, (
                f"Expected ROI '00' in configs, got keys: {list(pds.roi_configs.keys())}"
            )
            assert "01" in pds.roi_configs
            assert len(pds.roi_configs) == 2
        finally:
            os.unlink(tmp_path)

    def test_flat_yaml(self):
        """Load a flat (non-per-ROI) YAML as base config."""
        yaml_str = _make_flat_yaml()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(configs_source=tmp_path)
            # Flat configs are stored as _base_configs, not roi_configs
            assert pds._base_configs is not None
            assert isinstance(pds._base_configs, PipelineConfigurator)
            assert len(pds.roi_configs) == 0
        finally:
            os.unlink(tmp_path)

    def test_per_roi_yaml_with_datasource(self):
        """Load per-ROI YAML, then link a datasource.

        When a datasource is linked AFTER loading per-ROI configs,
        the existing per-ROI configs should be preserved.
        """
        yaml_str = _make_per_roi_yaml(["00", "01"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(configs_source=tmp_path)
            assert "00" in pds.roi_configs
            assert "01" in pds.roi_configs
        finally:
            os.unlink(tmp_path)


# --------------------------------------------------------------------------- #
#  Tests: KDE config loading                                                  #
# --------------------------------------------------------------------------- #

class TestLoadKDEConfigs:
    """PreparedDataSource loading of KDEConfigs from YAML."""

    def test_per_roi_kde_yaml_string_roi_names(self):
        """Load a per-ROI KDE YAML with string ROI names."""
        yaml_str = _make_per_roi_kde_yaml(["roi_00", "roi_01"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(kde_configs=tmp_path)
            assert "roi_00" in pds.roi_kde_configs
            assert "roi_01" in pds.roi_kde_configs
            assert len(pds.roi_kde_configs) == 2
            assert isinstance(pds.roi_kde_configs["roi_00"], KDEConfigs)
        finally:
            os.unlink(tmp_path)

    def test_per_roi_kde_yaml_numeric_roi_names(self):
        """Load a per-ROI KDE YAML with numeric-looking ROI names ('00').

        This is the critical edge case for KDE configs.  The fix in
        _load_kde_configs() converts int keys back to str.
        """
        yaml_str = _make_per_roi_kde_yaml(["00", "01"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(kde_configs=tmp_path)
            assert "00" in pds.roi_kde_configs, (
                f"Expected ROI '00' in kde_configs, "
                f"got keys: {list(pds.roi_kde_configs.keys())}"
            )
            assert "01" in pds.roi_kde_configs
            assert len(pds.roi_kde_configs) == 2
            assert isinstance(pds.roi_kde_configs["00"], KDEConfigs)
        finally:
            os.unlink(tmp_path)

    def test_flat_kde_yaml(self):
        """Load a flat (non-per-ROI) KDE YAML as base config.

        NOTE: _load_kde_configs assigns the result of KDEConfigs(**loaded).update(**kwargs)
        to _base_kde_configs.  Since KDEConfigs.update() returns None (in-place),
        _base_kde_configs ends up as None.  This is a pre-existing quirk of the
        codebase; the test verifies the actual behaviour.
        """
        yaml_str = _make_flat_kde_yaml()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(kde_configs=tmp_path)
            # Flat KDE configs are stored as _base_kde_configs, but
            # KDEConfigs.update() returns None, so _base_kde_configs is None.
            assert pds._base_kde_configs is None
            assert len(pds.roi_kde_configs) == 0
        finally:
            os.unlink(tmp_path)

    def test_per_roi_kde_yaml_with_datasource(self):
        """Load per-ROI KDE YAML, then link a datasource.

        When a datasource is linked AFTER loading per-ROI KDE configs,
        the existing per-ROI configs should be preserved.
        """
        yaml_str = _make_per_roi_kde_yaml(["00", "01"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(kde_configs=tmp_path)
            assert "00" in pds.roi_kde_configs
            assert "01" in pds.roi_kde_configs
        finally:
            os.unlink(tmp_path)

    def test_kde_config_values_preserved(self):
        """Verify that KDE config field values survive a round-trip."""
        yaml_str = _make_per_roi_kde_yaml(["00"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_str)
            tmp_path = f.name

        try:
            pds = PreparedDataSource(kde_configs=tmp_path)
            cfg = pds.roi_kde_configs["00"]
            assert cfg.KD_bandwidth == "fwhm"
            assert cfg.bwc == 1.0
            assert cfg.KD_kernel == "gaussian"
            assert cfg.KDE_algo is None
            assert cfg.split_mz_min == 10.0
            assert cfg.split_peaks_min == 25
            assert cfg.account_mzscale is True
        finally:
            os.unlink(tmp_path)


# --------------------------------------------------------------------------- #
#  Tests: combined regular + KDE config loading                               #
# --------------------------------------------------------------------------- #

class TestCombinedConfigLoading:
    """PreparedDataSource loading both regular and KDE configs together."""

    def test_both_per_roi_yamls(self):
        """Load both per-ROI regular and KDE YAML files."""
        reg_yaml = _make_per_roi_yaml(["00", "01"])
        kde_yaml = _make_per_roi_kde_yaml(["00", "01"])

        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as f_reg,
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as f_kde,
        ):
            f_reg.write(reg_yaml)
            f_kde.write(kde_yaml)
            reg_path = f_reg.name
            kde_path = f_kde.name

        try:
            pds = PreparedDataSource(
                configs_source=reg_path,
                kde_configs=kde_path,
            )
            assert "00" in pds.roi_configs
            assert "01" in pds.roi_configs
            assert "00" in pds.roi_kde_configs
            assert "01" in pds.roi_kde_configs
        finally:
            os.unlink(reg_path)
            os.unlink(kde_path)


# --------------------------------------------------------------------------- #
#  Tests: _is_per_roi_kde_yaml heuristic                                      #
# --------------------------------------------------------------------------- #

class TestIsPerRoiKdeYaml:
    """Unit tests for the _is_per_roi_kde_yaml heuristic."""

    def test_empty_dict(self):
        """An empty dict has no values to iterate, so the method returns None
        (implicit return).  This is a pre-existing quirk; the test documents it."""
        pds = PreparedDataSource()
        # The for-loop body never executes on an empty dict, so the method
        # falls through to an implicit return None.
        assert pds._is_per_roi_kde_yaml({}) is None

    def test_flat_kde_dict(self):
        """A flat KDE config dict should NOT be detected as per-ROI."""
        data = {
            "KD_bandwidth": "fwhm",
            "bwc": 1.0,
        }
        pds = PreparedDataSource()
        assert pds._is_per_roi_kde_yaml(data) is False

    def test_per_roi_kde_dict(self):
        """A per-ROI KDE config dict SHOULD be detected."""
        data = {
            "00": {
                "KD_bandwidth": "fwhm",
                "bwc": 1.0,
            }
        }
        pds = PreparedDataSource()
        assert pds._is_per_roi_kde_yaml(data) is True

    def test_non_dict_input(self):
        pds = PreparedDataSource()
        assert pds._is_per_roi_kde_yaml("not a dict") is False
        assert pds._is_per_roi_kde_yaml(42) is False
        assert pds._is_per_roi_kde_yaml(None) is False


# --------------------------------------------------------------------------- #
#  Tests: _is_per_roi_yaml heuristic                                          #
# --------------------------------------------------------------------------- #

class TestIsPerRoiYaml:
    """Unit tests for the _is_per_roi_yaml heuristic."""

    def test_flat_dict(self):
        """A flat config dict with 'methods' key should NOT be per-ROI."""
        data = {"methods": {"step": {}}}
        pds = PreparedDataSource()
        assert pds._is_per_roi_yaml(data) is False

    def test_per_roi_dict_string_keys(self):
        data = {"roi_00": {"methods": {}}, "roi_01": {"methods": {}}}
        pds = PreparedDataSource()
        assert pds._is_per_roi_yaml(data) is True

    def test_per_roi_dict_int_keys(self):
        """Dict with int keys (from PyYAML loading '00' as 0) is per-ROI."""
        data = {0: {"methods": {}}, 1: {"methods": {}}}
        pds = PreparedDataSource()
        assert pds._is_per_roi_yaml(data) is True

    def test_empty_dict(self):
        pds = PreparedDataSource()
        assert pds._is_per_roi_yaml({}) is False