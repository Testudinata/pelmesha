"""
Configuration management for mass spectrometry data processing pipelines.

Provides a Pydantic-based configuration system that supports:
- Flat and nested dict-style access to parameters
- Automatic validation and type coercion
- YAML serialization/deserialization
- Adaptive (m/z-to-dots) parameter conversion
- Dynamic field computation (headers, baseliner callable)
- Parameter distribution to function groups

Main class: Configs
Supporting classes: AdaptiveParameter, DatasetHeaders
"""
from __future__ import annotations
import ast
from pelmesha.utensils import printer
from pybaselines import Baseline
import inspect
import os
import warnings
from typing import Any, Callable
import numpy as np
import yaml
from h5py import File
from pydantic import BaseModel, Field, PrivateAttr, create_model
from threading import Thread
from multiprocessing import Pool, Manager, cpu_count
from pelmesha.filling import DataSource
from pelmesha.dough import Indexator, SliceIndexator
from pelmesha.kneading import preprocess_configuration_base, process_spectra_base, peakpicking_base
from tqdm.auto import tqdm
from functools import partial
import copy
import matplotlib.pyplot as plt
import pandas as pd
# --------------------------------------------------------------------------- #
#  Utility functions for adaptive parameter conversion                        #
# --------------------------------------------------------------------------- #




def _baseliner_prep(baseliner: str | None,
                    mz_scale: np.ndarray) -> Callable | None:
    """Prepare a pybaselines Baseline method callable."""
    if baseliner:
        return getattr(Baseline(mz_scale), baseliner)
    return None


def _default_config_path() -> str:
    """Return the path to the default Base_configs.yaml."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Base_configs.yaml")

def _inspect_defaults(functions: Callable | list | tuple) -> dict[str, inspect.Parameter]:
    """Return a dictionary of parameter names to inspect.Parameter objects.

    Each element in *functions* can be:
    - A plain callable (function or bound method)
    - A ``(callable, class)`` tuple - used when an unbound method is
      resolved together with its owning class (e.g. from AST extraction).
    """
    defaults = {}
    defaults['methods'] = {}
    methods_defaults = defaults['methods']
    defaults['functions'] = {}
    functions_defaults = defaults['functions']
    
    if not isinstance(functions, (list, tuple)):
        if callable(functions):
            functions = [functions]
        else:
            raise TypeError("func must be a callable or a list of callables")
    for item in functions:
        # Unpack (func, cls) tuple if present
        if isinstance(item, tuple) and len(item) == 2:
            func, cls = item
        else:
            func = item
            cls = None

        func_name = func.__name__
        
        # Skip callables that don't have an inspectable signature
        # (C extensions, builtins like np.where, etc.)
        try:
            func_sig = inspect.signature(func)
        except (ValueError, TypeError):
            continue

        # Получаем дефолтные параметры методов класса, функции и параметры инициализации класса
        if hasattr(func, "__self__"): # Bound methods
            cls = func.__self__.__class__
            cls_name = cls.__name__

            if cls_name not in methods_defaults:
                methods_defaults[cls_name] = {}
            methods_defaults[cls_name][func_name] = {}
            method = methods_defaults[cls_name][func_name]
            method["cls_init_params"] = {}

            for param in inspect.signature(cls).parameters.values(): # Получаем параметры инициализации класса
                if param.default is not inspect.Parameter.empty:
                    method["cls_init_params"][param.name] = param.default
            method['params'] = {}
            for param in func_sig.parameters.values():
                if param.default is not inspect.Parameter.empty:
                    method['params'][param.name] = param.default

        elif cls is not None and isinstance(cls, type): # Unbound method with class info
            cls_name = cls.__name__
            if cls_name not in methods_defaults:
                methods_defaults[cls_name] = {}
            methods_defaults[cls_name][func_name] = {}
            method = methods_defaults[cls_name][func_name]
            method["cls_init_params"] = {}
            for param in inspect.signature(cls).parameters.values():
                if param.default is not inspect.Parameter.empty:
                    method["cls_init_params"][param.name] = param.default
            method['params'] = {}
            for param in func_sig.parameters.values():
                if param.default is not inspect.Parameter.empty:
                    method['params'][param.name] = param.default

        else: # Plain functions
            functions_defaults[func_name] = {}
            for param in func_sig.parameters.values():
                if param.default is not inspect.Parameter.empty:
                    functions_defaults[func_name][param.name] = param.default
    return defaults

_KNOWN_BUCKETS: frozenset[str] = frozenset({"cls_init_params", "params"})
"""Known bucket names inside a method entry."""
# --------------------------------------------------------------------------- #
#  Dynamic parameter extraction from pybaselines                               #
# --------------------------------------------------------------------------- #

_MISSING: Any = object()
"""Sentinel for missing default values."""


_NP_TYPE_MAP: dict[type, type] = {
    float: float,
    int: int,
    bool: bool,
    str: str,
    list: list,
    tuple: tuple,
    np.float64: float,
    np.int64: int,
    np.bool_: bool,
    np.ndarray: list,
}
"""Map numpy types to Python types for Pydantic fields."""

# Parameters to skip when extracting from pybaselines methods
_SKIP_PARAMS: set[str] = {"self", "y", "x_data", "data", "weights",
                          "return_coef", "conserve_memory",
                          "pad_kwargs", "window_kwargs", "kwargs"}


def _map_annotation(annotation: Any, default: Any = _MISSING) -> type:
    """Map a type annotation to a Python type suitable for Pydantic fields.

    When ``annotation`` is unavailable (``inspect.Parameter.empty`` or
    ``None``), the type is inferred from the ``default`` value if provided.
    Falls back to ``float`` when neither annotation nor default is available.
    """
    if annotation is not inspect.Parameter.empty and annotation is not None:
        if annotation in _NP_TYPE_MAP:
            return _NP_TYPE_MAP[annotation]
        # Handle generic types like Optional[float], list[float], etc.
        origin = getattr(annotation, "__origin__", None)
        if origin is not None and origin in _NP_TYPE_MAP:
            return _NP_TYPE_MAP[origin]
        return float

    # No annotation available - infer from default value
    if default is not _MISSING and default is not None:
        if isinstance(default, bool):
            return bool
        if isinstance(default, int):
            return int
        if isinstance(default, float):
            return float
        if isinstance(default, str):
            return str
        if isinstance(default, (list, tuple)):
            return type(default)

    return float  # ultimate fallback


def _parse_numpy_docstring(doc: str) -> dict[str, str]:
    """
    Parse a numpy-style docstring to extract parameter descriptions.

    Returns a dict mapping parameter name -> description string.
    """
    if not doc:
        return {}
    params: dict[str, str] = {}
    lines = doc.split("\n")
    in_params = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "Parameters":
            in_params = True
            continue
        if stripped == "----------":
            continue
        if in_params:
            # Check if this line starts a new parameter
            if stripped and not stripped.startswith(" ") and ":" in stripped:
                # Save previous param
                if current_param and current_desc:
                    params[current_param] = " ".join(current_desc).strip()
                # Start new param
                current_param = stripped.split(":")[0].strip()
                current_desc = []
            elif current_param:
                # Continuation of description
                if stripped:
                    current_desc.append(stripped)
                elif current_desc:
                    # Empty line within description
                    current_desc.append("")

    # Save last param
    if current_param and current_desc:
        params[current_param] = " ".join(current_desc).strip()

    return params


def _extract_pybaselines_func_params(func: Callable) -> dict[str, dict[str, Any]]:
    """
    Extract parameter metadata from a pybaselines algorithm function.

    Returns a dict mapping parameter name -> metadata dict with keys:
    - ``"default"``: default value or ``_MISSING``
    - ``"annotation"``: type annotation or ``inspect.Parameter.empty``
    - ``"doc"``: description extracted from docstring (or ``""``)
    """
    sig = inspect.signature(func)
    doc_params = _parse_numpy_docstring(func.__doc__ or "")

    params: dict[str, dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if name in _SKIP_PARAMS:
            continue
        meta: dict[str, Any] = {
            "default": param.default if param.default is not inspect.Parameter.empty else _MISSING,
            "annotation": param.annotation,
            "doc": doc_params.get(name, ""),
        }
        params[name] = meta
    return params


_ALGO_MODEL_CACHE: dict[str, type[BaseModel]] = {}
"""Cache of dynamically created Pydantic models, keyed by algorithm name."""


def _build_algo_model(algo_name: str, func: Callable) -> type[BaseModel]:
    """
    Create a Pydantic model for a specific pybaselines algorithm.

    The model is cached by ``algo_name`` to avoid repeated creation.

    Parameters
    ----------
    algo_name : str
        Algorithm name (e.g. ``"asls"``, ``"modpoly"``).
    func : callable
        The pybaselines method (e.g. ``Baseline().asls``).

    Returns
    -------
    type[BaseModel]
        A dynamically created Pydantic model class.
    """
    if algo_name in _ALGO_MODEL_CACHE:
        return _ALGO_MODEL_CACHE[algo_name]

    params = _extract_pybaselines_func_params(func)
    fields: dict[str, Any] = {}
    for name, meta in params.items():
        annotation = meta.get("annotation", inspect.Parameter.empty)
        default = meta.get("default", _MISSING)
        doc = meta.get("doc", "")

        # Infer type from annotation or default value
        py_type = _map_annotation(annotation, default)

        if default is _MISSING:
            # Required parameter (no default)
            fields[name] = (py_type, Field(..., description=doc))
        else:
            fields[name] = (py_type, Field(default, description=doc))

    model = create_model(f"{algo_name}_params", **fields)
    _ALGO_MODEL_CACHE[algo_name] = model
    return model


# --------------------------------------------------------------------------- #
#  AdaptiveParameter - lazy m/z-to-dots conversion wrapper                    #
# --------------------------------------------------------------------------- #


class AdaptiveParameter:
    """
    Wraps a parameter with an optional adaptation rule for m/z-to-dots conversion.

    The adaptation rule is a callable ``f(parameter, *args)`` that transforms
    the parameter when the instance is called with data (e.g. ``dots_distance``).

    ``original`` always holds the user-provided value and never changes.
    ``parameter`` starts as ``original`` and is replaced by the adapted value
    after :meth:`__call__`.  On repeated calls the adaptation runs from
    ``original`` again, so multiple calls with different data produce correct
    results.

    Attributes
    ----------
    original : Any
        The original, user-provided parameter value (never changes).
    parameter : Any
        Current working value - equals ``original`` before adaptation,
        then the adapted value after :meth:`__call__`.
    adaptation_rule : callable or None
        Transformation function ``f(parameter, *args) -> adapted_value``.
    """

    def __init__(self, parameter: Any, adaptation_rule: Callable | None):
        self.original = parameter
        self.parameter = parameter
        self.adaptation_rule = adaptation_rule

    def __index__(self):
        return self.parameter

    def __repr__(self):
        return repr(self.parameter)

    def __call__(self, *args, **kwargs):
        if callable(self.adaptation_rule):
            self.parameter = self.adaptation_rule(self.original, *args)
        else:
            self.parameter = self.original
        return self.parameter

    def __len__(self):
        if callable(self.parameter):
            return True
        return len(self.parameter) if self.parameter is not None else False

    def __array__(self):
        return np.array(self.parameter)

    # Arithmetic
    def __add__(self, other): return self.parameter + other
    def __sub__(self, other): return self.parameter - other
    def __mul__(self, other): return self.parameter * other
    def __truediv__(self, other): return self.parameter / other
    def __floordiv__(self, other): return self.parameter // other
    def __mod__(self, other): return self.parameter % other
    def __pow__(self, other): return self.parameter ** other

    # Comparison
    def __eq__(self, other): return self.parameter == other
    def __ne__(self, other): return self.parameter != other
    def __lt__(self, other): return self.parameter < other
    def __le__(self, other): return self.parameter <= other
    def __gt__(self, other): return self.parameter > other
    def __ge__(self, other): return self.parameter >= other


# --------------------------------------------------------------------------- #
#  DatasetHeaders - bidirectional column-name/index mapping                   #
# --------------------------------------------------------------------------- #


class DatasetHeaders(list):
    """
    A list subclass that provides bidirectional mapping between column names
    and integer indices.

    Parameters
    ----------
    attrs : list of str
        Column header names in order.

    Examples
    --------
    >>> h = DatasetHeaders(["mz", "Intensity", "Area"])
    >>> h["Intensity"]
    1
    >>> h[0]
    'mz'
    >>> h(["Area", "mz"])
    [2, 0]
    """

    def __init__(self, attrs: list[str]):
        self.indexes: dict[str, int] = {}
        self.headnames: list[str] = [""] * len(attrs)
        for index, name in enumerate(attrs):
            self.headnames[index] = name
            self.indexes[name] = index
        super().__init__(self.headnames)

    def __call__(self, index_value: int | str | list) -> Any:
        if isinstance(index_value, list):
            result = [0] * len(self.headnames)
            if isinstance(index_value[0], int):
                for i, ind in enumerate(index_value):
                    result[i] = self.headnames[ind]
            elif isinstance(index_value[0], str):
                for i, ind in enumerate(index_value):
                    result[i] = self.indexes[ind]
            return result
        if isinstance(index_value, int):
            return self.headnames[index_value]
        if isinstance(index_value, str):
            return self.indexes[index_value]
        raise TypeError(f"Unsupported index type: {type(index_value)}")

    def __len__(self) -> int:
        return len(self.headnames)

    def __getitem__(self, index):
        return self.headnames[index]

    def __iter__(self):
        return iter(self.headnames)

    def __contains__(self, item):
        return item in self.headnames


# --------------------------------------------------------------------------- #
#  Pydantic parameter group models                                            #
# --------------------------------------------------------------------------- #


class BaselineParams(BaseModel):
    """
    Parameters for baseline correction using pybaselines_.

    .. _pybaselines: https://pybaselines.readthedocs.io/

    Algorithm-specific parameters (e.g. ``lam``, ``p``, ``diff_order`` for
    ``asls``) are stored in ``algo_params`` and validated through a
    dynamically created Pydantic model built from the pybaselines method
    signature.
    """
    baseline_algo: str | None = Field(
        None,
        description="Algorithm name for pybaselines Baseline (e.g. 'asls', 'penalized_poly')."
    )
    # Dynamic algorithm-specific parameters stored as a flat dict.
    # This is the serializable representation.
    algo_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm-specific parameters extracted from pybaselines."
    )
    # Dynamic Pydantic model for validation (not serialized by Pydantic).
    _algo_model: BaseModel | None = PrivateAttr(default=None)
    # Track which algorithm the current model was built for.
    _current_algo: str | None = PrivateAttr(default=None)


class SmoothingParams(BaseModel):
    """
    Parameters for spectral smoothing.
    """
    model_config = {"arbitrary_types_allowed": True}

    smooth_algo: str | None = Field(
        None,
        description="Smoothing algorithm: 'GA' (Gaussian), 'MA' (moving average), 'SG' (Savitzky-Golay), or None."
    )
    smooth_window: AdaptiveParameter = Field(
        default_factory=lambda: AdaptiveParameter(0.075, _smooth_window_to_dots),
        description="Smoothing window size in m/z units (adaptive)."
    )
    smooth_cycles: int = Field(
        1,
        description="Number of smoothing iterations.",
        ge=1
    )


class AlignmentParams(BaseModel):
    """
    Parameters for spectral alignment via msalign.
    """
    model_config = {"arbitrary_types_allowed": True}

    align_peaks: list[float] | None = Field(
        None,
        description="Reference peak m/z values for alignment. None = disabled."
    )
    align_pweights: list[float] | None = Field(
        None,
        description="Weights for each reference peak."
    )
    shift_range: AdaptiveParameter = Field(
        default_factory=lambda: AdaptiveParameter([-0.95, 0.95], _shift_range_to_dots),
        description="Maximum allowed shift in m/z units (adaptive)."
    )
    only_shift: bool = Field(
        True,
        description="If True, only shift spectra (no scaling)."
    )
    iterations: int = Field(
        3,
        description="Number of alignment iterations.",
        ge=1
    )
    width: float = Field(
        0.1,
        description="Peak width in m/z for alignment."
    )


class PeakPickingParams(BaseModel):
    """
    Parameters for peak detection and filtering.
    """
    oversegmentationfilter: float | None = Field(
        None,
        description="Filter threshold for over-segmented peaks (merges closer than this)."
    )
    fwhhfilter: float | None = Field(
        None,
        description="FWHH filter: exclude peaks with FWHH below this value."
    )
    heightfilter: float | None = Field(
        None,
        description="Absolute intensity filter: exclude peaks below this intensity."
    )
    rel_heightfilter: float | None = Field(
        None,
        description="Relative height filter (0-100%): exclude peaks below this percentile."
    )
    peaklocation: float = Field(
        1.0,
        description="Peak location parameter (0-1) for barycentric center computation.",
        ge=0.0,
        le=1.0
    )
    SNR_threshold: float = Field(
        3.5,
        description="SNR threshold for peak validation."
    )
    noise_func: str = Field(
        "std",
        description="Noise estimation function: 'std' (standard deviation) or 'MAD' (median absolute deviation)."
    )
    noise_est_iterations: int = Field(
        3,
        description="Number of noise estimation iterations.",
        ge=1
    )
    Calc_peak_area: bool = Field(
        True,
        description="If True, calculate peak area."
    )


# --------------------------------------------------------------------------- #
#  Main Configs class                                                         #
# --------------------------------------------------------------------------- #

# Set of all static (known) parameter names that are direct Pydantic fields.
# Dynamic pybaselines parameters (lam, p, diff_order, ...) are NOT in this set
# and are handled via ``extra='allow'`` -> ``__pydantic_extra__``.
_STATIC_PARAMS: set[str] = {
    # Baseline
    "baseline_algo",
    # Smoothing
    "smooth_algo", "smooth_window", "smooth_cycles",
    # Alignment
    "align_peaks", "align_pweights", "shift_range",
    "only_shift", "iterations", "width",
    # Peak picking
    "oversegmentationfilter", "fwhhfilter", "heightfilter",
    "rel_heightfilter", "peaklocation", "SNR_threshold",
    "noise_func", "noise_est_iterations", "Calc_peak_area",
    # Cross-cutting
    "resample_to_dots",
}

# Mapping from flat parameter names to their Pydantic sub-model field names.
_PARAM_TO_MODEL: dict[str, str] = {
    "baseline_algo": "baseline_params",
    "smooth_algo": "smoothing_params",
    "smooth_window": "smoothing_params",
    "smooth_cycles": "smoothing_params",
    "align_peaks": "alignment_params",
    "align_pweights": "alignment_params",
    "shift_range": "alignment_params",
    "only_shift": "alignment_params",
    "iterations": "alignment_params",
    "width": "alignment_params",
    "oversegmentationfilter": "peakpicking_params",
    "fwhhfilter": "peakpicking_params",
    "heightfilter": "peakpicking_params",
    "rel_heightfilter": "peakpicking_params",
    "peaklocation": "peakpicking_params",
    "SNR_threshold": "peakpicking_params",
    "noise_func": "peakpicking_params",
    "noise_est_iterations": "peakpicking_params",
    "Calc_peak_area": "peakpicking_params",
}

# Reverse mapping: nested config group name -> Pydantic model field name.
_NESTED_GROUP_TO_MODEL: dict[str, str] = {
    "baseline_configs": "baseline_params",
    "smoothing_configs": "smoothing_params",
    "msalign_configs": "alignment_params",
    "peaks_configs": "peakpicking_params",
}

class PydanticConfigs(BaseModel):
    """
    Unified configuration for mass spectrometry data processing.

    All static parameters are declared as direct Pydantic fields, so your IDE
    will show autocompletion hints when typing ``Configs(...``.

    Supports both **flat** access (``configs["smooth_window"]``) and
    **nested** access (``configs["smoothing_configs"]["smooth_window"]``).

    Parameters
    ----------
    functions_list : list of callable, optional
        List of pipeline functions whose parameter signatures are used to
        distribute configs into nested groups.
    config_path : str, optional
        Path to a YAML configuration file. If ``None``, loads
        ``Base_configs.yaml`` from the package directory.
    **kwargs
        Additional parameter overrides (flat keys). Dynamic pybaselines
        parameters (e.g. ``lam``, ``p``, ``diff_order``) are accepted here.

    Examples
    --------
    >>> cfg = Configs(smooth_window=0.15)  # IDE shows autocomplete
    >>> cfg["smooth_window"]
    0.15
    >>> cfg["smoothing_configs"]["smooth_window"]
    0.15
    >>> cfg.dump("./my_config.yaml")
    """

    # ================================================================== #
    #  Direct Pydantic fields - visible in IDE autocomplete               #
    # ================================================================== #

    # --- Baseline ---
    baseline_algo: str | None = Field(
        None,
        description="Algorithm name for pybaselines Baseline (e.g. 'asls', 'penalized_poly')."
    )

    # --- Smoothing ---
    smooth_algo: str | None = Field(
        None,
        description="Smoothing algorithm: 'GA' (Gaussian), 'MA' (moving average), 'SG' (Savitzky-Golay), or None."
    )
    smooth_window: AdaptiveParameter = Field(
        default_factory=lambda: AdaptiveParameter(0.075, _smooth_window_to_dots),
        description="Smoothing window size in m/z units (adaptive)."
    )
    smooth_cycles: int = Field(
        1,
        description="Number of smoothing iterations.",
        ge=1
    )

    # --- Alignment ---
    align_peaks: list[float] | None = Field(
        None,
        description="Reference peak m/z values for alignment. None = disabled."
    )
    align_pweights: list[float] | None = Field(
        None,
        description="Weights for each reference peak."
    )
    shift_range: AdaptiveParameter = Field(
        default_factory=lambda: AdaptiveParameter([-0.95, 0.95], _shift_range_to_dots),
        description="Maximum allowed shift in m/z units (adaptive)."
    )
    only_shift: bool = Field(
        True,
        description="If True, only shift spectra (no scaling)."
    )
    iterations: int = Field(
        3,
        description="Number of alignment iterations.",
        ge=1
    )
    width: float = Field(
        0.1,
        description="Peak width in m/z for alignment."
    )

    # --- Peak picking ---
    oversegmentationfilter: float | None = Field(
        None,
        description="Filter threshold for over-segmented peaks (merges closer than this)."
    )
    fwhhfilter: float | None = Field(
        None,
        description="FWHH filter: exclude peaks with FWHH below this value."
    )
    heightfilter: float | None = Field(
        None,
        description="Absolute intensity filter: exclude peaks below this intensity."
    )
    rel_heightfilter: float | None = Field(
        None,
        description="Relative height filter (0-100%): exclude peaks below this percentile."
    )
    peaklocation: float = Field(
        1.0,
        description="Peak location parameter (0-1) for barycentric center computation.",
        ge=0.0, le=1.0
    )
    SNR_threshold: float = Field(
        3.5,
        description="SNR threshold for peak validation."
    )
    noise_func: str = Field(
        "std",
        description="Noise estimation function: 'std' (standard deviation) or 'MAD' (median absolute deviation)."
    )
    noise_est_iterations: int = Field(
        3,
        description="Number of noise estimation iterations.",
        ge=1
    )
    Calc_peak_area: bool = Field(
        True,
        description="If True, calculate peak area."
    )

    # --- Cross-cutting ---
    resample_to_dots: bool | int = Field(
        False,
        description="Resample spectra to N dots. False = no resampling."
    )

    # ================================================================== #
    #  Pydantic sub-models (synced from flat fields by model_validator)    #
    # ================================================================== #

    baseline_params: BaselineParams = Field(default_factory=BaselineParams)
    smoothing_params: SmoothingParams = Field(default_factory=SmoothingParams)
    alignment_params: AlignmentParams = Field(default_factory=AlignmentParams)
    peakpicking_params: PeakPickingParams = Field(default_factory=PeakPickingParams)

    # ================================================================== #
    #  Private attributes (not serialized by Pydantic)                    #
    # ================================================================== #

    _headers: DatasetHeaders | None = PrivateAttr(default=None)
    _noise_func_callable: Callable | None = PrivateAttr(default=None)
    _align_by_index: bool = PrivateAttr(default=False)

    # ================================================================== #
    #  Model configuration                                                #
    # ================================================================== #

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",  # в†ђ dynamic pybaselines params go to __pydantic_extra__
    }

    # ================================================================== #
    #  Initialisation                                                     #
    # ================================================================== #

    def __init__(self,
                 config_path: str | None = None,
                 functions_list: list | None = None,
                 **kwargs):
        # 1. Load base config from YAML if functions_list is None (default functions is used)
        func_params: dict[str, Any] = {}
        if (config_path is None) and (functions_list is None):
            config_path = _default_config_path()

        if isinstance(config_path, str):
            if not config_path.endswith(".yaml"):
                config_path += ".yaml"
                    
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    loaded = yaml.load(f, Loader=yaml.FullLoader)
                    if loaded:
                        func_params.update(loaded)
        else:
            func_params = _inspect_defaults(functions_list)

        # 2. Override with kwargs
        for key, value in kwargs.items():
            func_params[key].update(value)
        
        # 3. Separate sub-model dicts from flat params.
        #    Known static params are passed directly to Pydantic (IDE sees them).
        #    Unknown params (dynamic pybaselines) go to __pydantic_extra__.
        
        # flat_data: dict[str, Any] = {}
        # for func_name, params in func_params.items():
        #     for param_name, value in params.items():
        #         flat_data[func_name + "__" + param_name] = value

        # 4. Convert float values to AdaptiveParameter for smooth_window and shift_range
        if "smooth_window" in flat_data and not isinstance(flat_data["smooth_window"], AdaptiveParameter):
            flat_data["smooth_window"] = AdaptiveParameter(flat_data["smooth_window"], _smooth_window_to_dots)
        if "shift_range" in flat_data and not isinstance(flat_data["shift_range"], AdaptiveParameter):
            raw = flat_data["shift_range"]
            if isinstance(raw, (int, float)):
                raw = [-raw, raw]
            flat_data["shift_range"] = AdaptiveParameter(raw, _shift_range_to_dots)

        # 5. Initialize Pydantic BaseModel
        super().__init__(**flat_data, **sub_model_data)
        # 6. Store init metadata
        self._config_path = config_path
        # 7. Post-init setup (syncs sub-models, validates baseline, etc.)
        self._post_init_setup()

    # ------------------------------------------------------------------ #
    #  Post-initialisation setup                                          #
    # ------------------------------------------------------------------ #

    def _post_init_setup(self) -> None:
        """Run after Pydantic validation to sync sub-models, set up adaptive
        parameters, dynamic fields, and cross-parameter validation."""
        # 0. Sync flat fields -> sub-models
        self._sync_submodels()

        # 1. Validate baseline_algo against pybaselines
        self._validate_baseline_algo()

        # 2. Convert noise_func string to callable
        self._setup_noise_func()

        # 3. Compute dynamic headers
        self._compute_headers()

    def _sync_submodels(self) -> None:
        """Synchronise direct flat fields -> Pydantic sub-models."""
        for flat_key, model_name in _PARAM_TO_MODEL.items():
            value = getattr(self, flat_key, None)
            model = getattr(self, model_name)
            if hasattr(model, flat_key):
                setattr(model, flat_key, value)

        # Sync dynamic extras -> algo_params
        extras = self.__pydantic_extra__ or {}
        for key, value in extras.items():
            if key not in _STATIC_PARAMS:
                self.baseline_params.algo_params[key] = value

    def _validate_baseline_algo(self) -> None:
        """Validate baseline_algo and build dynamic parameter model.

        When ``baseline_algo`` is set, this method:
        1. Checks that the method exists in pybaselines.Baseline
        2. Builds a dynamic Pydantic model from the method's signature
        3. Populates defaults, merging with any existing ``algo_params``
        4. Validates and stores the result in ``algo_params``
        """
        bp = self.baseline_params
        algo = bp.baseline_algo
        if algo:
            from pybaselines import Baseline
            baseline = Baseline()
            func = getattr(baseline, algo, None)
            if func is None:
                available = [m for m in dir(baseline) if not m.startswith('_')]
                raise ValueError(
                    f"Method '{algo}' not found in pybaselines.Baseline. "
                    f"Available methods: {', '.join(available)}"
                )

            # Build dynamic model for this algorithm
            model_class = _build_algo_model(algo, func)

            # Get defaults from the dynamic model
            defaults: dict[str, Any] = {}
            for field_name, field_info in model_class.model_fields.items():
                if field_info.default is not None and field_info.default is not ...:
                    defaults[field_name] = field_info.default

            # Merge: existing algo_params override defaults
            merged = {**defaults, **bp.algo_params}

            # Validate through the dynamic model
            try:
                validated = model_class(**merged)
                bp._algo_model = validated
                bp._current_algo = algo
                # Store validated values back to algo_params dict
                bp.algo_params = validated.model_dump()
            except Exception as e:
                raise ValueError(
                    f"Invalid parameters for baseline algorithm '{algo}': {e}"
                ) from e
        else:
            bp._algo_model = None
            bp._current_algo = None
            bp.algo_params = {}

    def _setup_noise_func(self) -> None:
        """Convert noise_func string to a callable."""
        nf = self.peakpicking_params.noise_func
        if nf == "MAD":
            from scipy.stats import median_abs_deviation
            import math

            def _mad(y, nan_policy="omit"):
                return (math.sqrt(2 * math.log(len(y)))
                        * median_abs_deviation(y, nan_policy) / 0.6745)
            self._noise_func_callable = _mad
        elif nf == "std":
            self._noise_func_callable = np.std
        else:
            self._noise_func_callable = np.std

    def _compute_headers(self) -> None:
        """Dynamically generate DatasetHeaders based on SNR and area flags."""
        pp = self.peakpicking_params
        if pp.SNR_threshold and pp.Calc_peak_area:
            self._headers = DatasetHeaders([
                "spectra_ind", "mz", "Intensity", "Area", "SNR",
                "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"
            ])
        elif pp.SNR_threshold:
            self._headers = DatasetHeaders([
                "spectra_ind", "mz", "Intensity", "SNR",
                "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"
            ])
        elif pp.Calc_peak_area:
            self._headers = DatasetHeaders([
                "spectra_ind", "mz", "Intensity", "Area",
                "PextL", "PextR", "FWHML", "FWHMR"
            ])
        else:
            self._headers = DatasetHeaders([
                "spectra_ind", "mz", "Intensity",
                "PextL", "PextR", "FWHML", "FWHMR"
            ])

    # ------------------------------------------------------------------ #
    #  Dict-style access (flat + nested)                                  #
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: str) -> Any:
        """Flat or nested dict-style access.

        Priority:
        1. Direct attribute on self (static Pydantic fields)
        2. Special computed keys: ``"headers"``, ``"noise_func"``,
           ``"align_by_index"``
        3. Nested config group name (e.g. ``"smoothing_configs"`` -> returns a
           flat dict view of that group)
        4. Parameter inside a nested group (searches all sub-models)
        5. Dynamic baseline algorithm parameters (``algo_params``)
        6. ``__pydantic_extra__`` (dynamic extras)
        """
        # 1. Direct root-level attribute (static Pydantic fields)
        if key in self.model_fields:
            return getattr(self, key)

        # 2. Special computed keys
        computed = {
            "headers": self._headers,
            "noise_func": self._noise_func_callable,
            "align_by_index": self._align_by_index,
        }
        if key in computed:
            return computed[key]

        # 3. Nested config group name -> return flat dict view
        if key in _NESTED_GROUP_TO_MODEL:
            model_name = _NESTED_GROUP_TO_MODEL[key]
            model = getattr(self, model_name)
            return model.model_dump()

        # 4. Search in sub-models
        for model_name in ("baseline_params", "smoothing_params",
                           "alignment_params", "peakpicking_params"):
            model = getattr(self, model_name)
            if hasattr(model, key):
                return getattr(model, key)

        # 5. Dynamic baseline algorithm parameters
        if key in self.baseline_params.algo_params:
            return self.baseline_params.algo_params[key]

        # 6. __pydantic_extra__ (dynamic extras)
        extras = self.__pydantic_extra__ or {}
        if key in extras:
            return extras[key]

        raise KeyError(f"Parameter '{key}' not found in Configs.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a parameter value, triggering re-validation and re-setup
        where necessary."""
        # Handle special computed keys
        if key in ("baseliner", "baseline_algo"):
            self.baseline_algo = value
            self.baseline_params.baseline_algo = value
            self._validate_baseline_algo()
            return

        if key == "shift_range":
            if isinstance(value, (int, float)):
                value = AdaptiveParameter([-value, value], _shift_range_to_dots)
            self.shift_range = value
            self.alignment_params.shift_range = value
            return

        if key == "smooth_window":
            if isinstance(value, (int, float)):
                value = AdaptiveParameter(value, _smooth_window_to_dots)
            self.smooth_window = value
            self.smoothing_params.smooth_window = value
            return

        if key == "noise_func":
            self.noise_func = value
            self.peakpicking_params.noise_func = value
            self._setup_noise_func()
            return

        if key in ("SNR_threshold", "Calc_peak_area"):
            setattr(self, key, value)
            setattr(self.peakpicking_params, key, value)
            self._compute_headers()
            return

        # Try root-level attribute (static Pydantic field)
        if key in self.model_fields:
            setattr(self, key, value)
            # Also sync to sub-model
            if key in _PARAM_TO_MODEL:
                model_name = _PARAM_TO_MODEL[key]
                model = getattr(self, model_name)
                if hasattr(model, key):
                    setattr(model, key, value)
            return

        # Try sub-models
        for model_name in ("baseline_params", "smoothing_params",
                           "alignment_params", "peakpicking_params"):
            model = getattr(self, model_name)
            if hasattr(model, key):
                setattr(model, key, value)
                return

        # Check if this is a dynamic baseline algo param
        bp = self.baseline_params
        if bp._algo_model is not None and key in bp._algo_model.model_fields:
            try:
                validated = bp._algo_model.model_copy(update={key: value})
                bp._algo_model = validated
                bp.algo_params[key] = value
            except Exception as e:
                raise ValueError(
                    f"Invalid value for baseline parameter '{key}': {e}"
                ) from e
            return

        # Store in __pydantic_extra__
        if self.__pydantic_extra__ is None:
            self.__pydantic_extra__ = {}
        self.__pydantic_extra__[key] = value

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------ #
    #  Flatten - return all params as a single flat dict                  #
    # ------------------------------------------------------------------ #

    def flatten(self) -> dict[str, Any]:
        """
        Return all configuration parameters as a single flat dictionary.

        AdaptiveParameter instances are unwrapped to their ``.original``
        value. DatasetHeaders are excluded.
        """
        result: dict[str, Any] = {}

        # Static Pydantic fields (direct on Configs)
        for field_name in self.model_fields:
            if field_name in ("baseline_params", "smoothing_params",
                              "alignment_params", "peakpicking_params"):
                continue
            value = getattr(self, field_name)
            if isinstance(value, AdaptiveParameter):
                result[field_name] = value.original
            else:
                result[field_name] = value

        # Dynamic baseline algorithm parameters
        for k, v in self.baseline_params.algo_params.items():
            result[k] = v

        # __pydantic_extra__
        extras = self.__pydantic_extra__ or {}
        for k, v in extras.items():
            if k not in result:
                result[k] = v

        return result

    # ------------------------------------------------------------------ #
    #  YAML serialisation (with section headers)                          #
    # ------------------------------------------------------------------ #

    # Mapping: section header -> list of parameter names in that section.
    # Parameters not listed here go to the end under "# Other".
    _YAML_SECTIONS: dict[str, list[str]] = {
        "# Baseline correction": [
            "baseline_algo",
            # algo_params are injected dynamically below
        ],
        "# Smoothing": [
            "smooth_algo", "smooth_window", "smooth_cycles",
        ],
        "# Alignment (msalign)": [
            "align_peaks", "align_pweights", "shift_range",
            "only_shift", "iterations", "width",
        ],
        "# Peak picking": [
            "oversegmentationfilter", "fwhhfilter", "heightfilter",
            "rel_heightfilter", "peaklocation", "SNR_threshold",
            "noise_func", "noise_est_iterations", "Calc_peak_area",
        ],
        "# Processing": [
            "resample_to_dots",
        ],
    }

    

    # ------------------------------------------------------------------ #
    #  Serialisation support (pickle, multiprocessing)                    #
    # ------------------------------------------------------------------ #

    def __getstate__(self) -> dict:
        """Return state for pickling (removes non-serializable attrs)."""
        state = self.__dict__.copy()

        # Convert AdaptiveParameter to serializable form.
        # Module-level functions (_smooth_window_to_dots, _shift_range_to_dots)
        # ARE picklable, so we serialize the adaptation_rule as-is.
        for attr in ("smooth_window", "shift_range"):
            ap = state.get(attr)
            if isinstance(ap, AdaptiveParameter):
                state[attr] = {
                    "original": ap.original,
                    "parameter": ap.parameter,
                    "adaptation_rule": ap.adaptation_rule,
                }

        # Remove callable from __pydantic_private__ (will be rebuilt by _post_init_setup).
        # Do NOT set state["_noise_func_callable"] directly - that would add a
        # plain __dict__ entry that shadows the PrivateAttr in __pydantic_private__.
        private = state.get("__pydantic_private__")
        if private is not None:
            private.pop("_noise_func_callable", None)

        # Remove dynamically created Pydantic model from BaselineParams
        # (asls_params etc. can't be pickled; _post_init_setup rebuilds it).
        # Pydantic v2 stores PrivateAttr values in __pydantic_private__ dict.
        bp = state.get("baseline_params")
        if bp is not None and hasattr(bp, "__pydantic_private__"):
            bp.__pydantic_private__.pop("_algo_model", None)
            bp.__pydantic_private__.pop("_current_algo", None)

        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle."""
        self.__dict__.update(state)

        # Pydantic v2 only creates __pydantic_extra__ during __init__,
        # which we bypass in __setstate__. Ensure it exists.
        if not hasattr(self, "__pydantic_extra__") or self.__pydantic_extra__ is None:
            object.__setattr__(self, "__pydantic_extra__", {})

        # Ensure __pydantic_private__ exists with all expected PrivateAttr keys.
        # After unpickling, some default-valued PrivateAttrs may be missing.
        if not hasattr(self, "__pydantic_private__") or self.__pydantic_private__ is None:
            object.__setattr__(self, "__pydantic_private__", {})
        expected_private = {
            "_headers": None,
            "_noise_func_callable": None,
            "_align_by_index": False,
        }
        for key, default in expected_private.items():
            if key not in self.__pydantic_private__:
                self.__pydantic_private__[key] = default

        # Restore AdaptiveParameter from serialized dicts.
        # Module-level functions are picklable, so adaptation_rule is preserved.
        for attr in ("smooth_window", "shift_range"):
            val = getattr(self, attr, None)
            if isinstance(val, dict):
                ap = AdaptiveParameter(val["original"], val.get("adaptation_rule"))
                ap.parameter = val["parameter"]
                object.__setattr__(self, attr, ap)
                # Also sync to sub-model
                if attr == "smooth_window":
                    object.__setattr__(self.smoothing_params, attr, ap)
                elif attr == "shift_range":
                    object.__setattr__(self.alignment_params, attr, ap)

        # Re-setup adaptive parameters, baseline algo, noise func, headers
        self._post_init_setup()

    # ------------------------------------------------------------------ #
    #  Representation                                                     #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lines = ["Configs("]
        for group_name, model_name in _NESTED_GROUP_TO_MODEL.items():
            model = getattr(self, model_name)
            lines.append(f"  {group_name}: {model.model_dump()}")
        lines.append(f"  resample_to_dots: {self.resample_to_dots}")
        lines.append(")")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()
class Configs():
    def __init__(self,
                 configs_source: str | dict = {},
                 **kwargs):
        self.configs: dict[str, Any] = {}
        self.update(configs_source, **kwargs)

    @staticmethod
    def _validate_configs_structure(data: dict) -> None:
        """Validate that *data* looks like a proper Configs dictionary.

        Accepts two top-level structures:

        * ``{"methods": {...}, "functions": {...}}`` - full config dump
        * ``{"methods": {...}}`` or ``{"functions": {...}}`` - partial

        Raises ``ValueError`` with a clear message if the structure is
        unrecognised.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a dict, got {type(data).__name__}."
            )
        top_keys = set(data.keys())
        allowed = {"methods", "functions"}
        if not top_keys:
            raise ValueError(
                "Config dict is empty. Expected 'methods' and/or 'functions' keys."
            )
        unknown = top_keys - allowed
        if unknown:
            raise ValueError(
                f"Unknown top-level key(s) in config dict: {unknown}. "
                f"Allowed keys: {allowed}. "
                f"If you meant to pass a callable or file path, use the "
                f"first positional argument instead."
            )
        # Validate methods structure if present
        methods = data.get("methods", {})
        if not isinstance(methods, dict):
            raise ValueError(
                f"'methods' must be a dict, got {type(methods).__name__}."
            )
        for cls_name, func_dict in methods.items():
            if not isinstance(func_dict, dict):
                raise ValueError(
                    f"methods['{cls_name}'] must be a dict, "
                    f"got {type(func_dict).__name__}."
                )
            for method_name, bucket in func_dict.items():
                if not isinstance(bucket, dict):
                    raise ValueError(
                        f"methods['{cls_name}']['{method_name}'] must be a dict, "
                        f"got {type(bucket).__name__}."
                    )
                allowed_bucket_keys = {"cls_init_params", "params"}
                bucket_keys = set(bucket.keys())
                unknown_bucket = bucket_keys - allowed_bucket_keys
                if unknown_bucket:
                    raise ValueError(
                        f"methods['{cls_name}']['{method_name}'] has unknown "
                        f"key(s): {unknown_bucket}. "
                        f"Allowed keys: {allowed_bucket_keys}."
                    )
        # Validate functions structure if present
        functions = data.get("functions", {})
        if not isinstance(functions, dict):
            raise ValueError(
                f"'functions' must be a dict, got {type(functions).__name__}."
            )
        for func_name, params in functions.items():
            if not isinstance(params, dict):
                raise ValueError(
                    f"functions['{func_name}'] must be a dict, "
                    f"got {type(params).__name__}."
                )
    # ------------------------------------------------------------------ #
    #  Name-based parameter access                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_params(bucket: dict[str, Any]) -> dict[str, Any]:
        """Extract the relevant parameter dict from a method bucket.

        For methods the bucket has ``params`` and ``cls_init_params`` keys.
        This helper returns only the actual parameters:
        - ``params`` for a method
        - ``cls_init_params`` for a class
        """
        if "params" in bucket and "cls_init_params" in bucket:
            return dict(bucket["params"])
        return dict(bucket)

    def _resolve_method_name(self, name: str,
                             methods: dict) -> tuple[str, str]:
        """Resolve a bare *method name* to ``(cls_name, method_name)``.

        Searches across all classes in *methods*.  Raises ``KeyError``
        when the name is not found or is ambiguous (appears in multiple
        classes).
        """
        matches: list[str] = []
        for cls_name, func_dict in methods.items():
            if name in func_dict:
                matches.append(f"{cls_name}.{name}")

        if not matches:
            raise KeyError(
                f"Method '{name}' not found in any class."
            )
        if len(matches) > 1:
            loc_list = "\n  - ".join(matches)
            raise KeyError(
                f"'{name}' is ambiguous - found in {len(matches)} classes:\n"
                f"  - {loc_list}\n"
                f"Use a qualified name to disambiguate:\n"
                f"    cfg[\"ClassName.{name}\"]\n"
                f"    cfg[(\"ClassName\", \"{name}\")]"
            )
        cls_name, func_name = matches[0].split(".")
        return cls_name, func_name

    def __getitem__(self, key: str | tuple[str, ...]) -> dict[str, Any]:
        """Retrieve parameters by function/method/class name.

        Returns only the **actual parameters** — for methods this is
        the ``params`` sub-dict, for classes the ``cls_init_params``.

        Supports several lookup forms:

        * ``cfg["msalign"]`` — plain function name
        * ``cfg["snip"]`` — method name (warns if ambiguous)
        * ``cfg["Baseline"]`` — class name → ``cls_init_params``
        * ``cfg["Baseline.snip"]`` — ``ClassName.methodName`` → ``params``
        * ``cfg[("Baseline", "snip")]`` — ``(ClassName, methodName)``
        * ``cfg[("Baseline", "snip", "cls_init_params")]`` — explicit bucket
        * ``cfg[("snip", "cls_init_params")]`` — ``(methodName, bucketName)``
        * ``cfg["snip.cls_init_params"]`` — ``methodName.bucketName``

        Parameters
        ----------
        key : str or tuple of str
            Name or names identifying the target.

        Returns
        -------
        dict
            The parameter dictionary for the matched function/method.

        Raises
        ------
        KeyError
            If the name cannot be resolved to any known group.
        """
        methods: dict = self.configs.get("methods", {})
        functions: dict = self.configs.get("functions", {})

        # --- (methodName, bucketName) or (ClassName, methodName[, bucket]) ---
        if isinstance(key, tuple):
            if len(key) == 2:
                first, second = key
                # If second is a known bucket name → (methodName, bucketName)
                if second in _KNOWN_BUCKETS:
                    cls_name, func_name = self._resolve_method_name(
                        first, methods
                    )
                    fb = methods[cls_name][func_name]
                    if second not in fb:
                        raise KeyError(
                            f"'{cls_name}.{func_name}' has no "
                            f"'{second}' bucket."
                        )
                    return dict(fb[second])
                # Otherwise → (ClassName, methodName)
                cls_name, func_name = first, second
                bucket = None
            elif len(key) == 3:
                cls_name, func_name, bucket = key
            else:
                raise KeyError(
                    f"Tuple key must have 2 or 3 elements, got {len(key)}."
                )
            if cls_name not in methods:
                raise KeyError(f"Unknown class '{cls_name}'.")
            if func_name not in methods[cls_name]:
                raise KeyError(
                    f"Unknown method '{cls_name}.{func_name}'."
                )
            fb = methods[cls_name][func_name]
            if bucket is not None:
                if bucket not in fb:
                    raise KeyError(
                        f"'{cls_name}.{func_name}' has no '{bucket}' bucket."
                    )
                return dict(fb[bucket])
            return self._extract_params(fb)

        # --- "methodName.bucketName" or "ClassName.methodName" string ---
        if isinstance(key, str) and "." in key:
            parts = key.split(".", 1)
            first, second = parts
            # If second is a known bucket name → methodName.bucketName
            if second in _KNOWN_BUCKETS:
                cls_name, func_name = self._resolve_method_name(
                    first, methods
                )
                fb = methods[cls_name][func_name]
                if second not in fb:
                    raise KeyError(
                        f"'{cls_name}.{func_name}' has no "
                        f"'{second}' bucket."
                    )
                return dict(fb[second])
            # Otherwise → ClassName.methodName
            cls_name, func_name = first, second
            if cls_name not in methods:
                raise KeyError(f"Unknown class '{cls_name}'.")
            if func_name not in methods[cls_name]:
                raise KeyError(
                    f"Unknown method '{cls_name}.{func_name}'."
                )
            return self._extract_params(methods[cls_name][func_name])

        # --- Plain name ---
        if isinstance(key, str):
            name = key

            # 1. Check plain functions
            if name in functions:
                return dict(functions[name])

            # 2. Collect matching methods
            method_matches: list[str] = []
            for cls_name, func_dict in methods.items():
                if name in func_dict:
                    method_matches.append(f"{cls_name}.{name}")

            # 3. Collect matching classes
            class_matches: list[str] = []
            if name in methods:
                class_matches.append(name)

            total = len(method_matches) + len(class_matches)

            if total == 0:
                raise KeyError(
                    f"'{name}' not found in any known function, method, "
                    f"or class."
                )

            if total == 1:
                if method_matches:
                    cls_name, func_name = method_matches[0].split(".")
                    return self._extract_params(methods[cls_name][func_name])
                # class match - return cls_init_params
                cls_methods = methods[name]
                if len(cls_methods) == 1:
                    func_name = next(iter(cls_methods))
                    fb = cls_methods[func_name]
                    return dict(fb.get("cls_init_params", {}))
                else:
                    # Multiple methods in class - return all cls_init_params
                    return {
                        m: dict(d.get("cls_init_params", {}))
                        for m, d in cls_methods.items()
                    }

            # Ambiguous - warn and raise
            all_matches = method_matches + class_matches
            loc_list = "\n  - ".join(all_matches)
            raise KeyError(
                f"'{name}' is ambiguous - found in {total} locations:\n"
                f"  - {loc_list}\n"
                f"Use a qualified name to disambiguate:\n"
                f"    cfg[\"ClassName.methodName\"]\n"
                f"    cfg[(\"ClassName\", \"methodName\")]"
            )

        raise KeyError(f"Unsupported key type: {type(key).__name__}.")
    
    def __repr__(self) -> str:
        """Human-readable representation grouped by function/method."""
        methods: dict = self.configs.get("methods", {})
        functions: dict = self.configs.get("functions", {})
        lines = ["PipelineConfigurator("]

        # --- Plain functions ---
        if functions:
            lines.append("  functions:")
            for func_name, params in functions.items():
                if params:
                    items = ", ".join(f"{k}={v!r}" for k, v in params.items())
                    lines.append(f"    {func_name}: {items}")
                else:
                    lines.append(f"    {func_name}: (no parameters)")

        # --- Methods grouped by class ---
        if methods:
            lines.append("  methods:")
            for cls_name, func_dict in methods.items():
                lines.append(f"    {cls_name}:")
                for method_name, bucket in func_dict.items():
                    # Class init params
                    init_params = bucket.get("cls_init_params", {})
                    if init_params:
                        items = ", ".join(f"{k}={v!r}" for k, v in init_params.items())
                        lines.append(f"      __init__({items})")
                    # Method params
                    method_params = bucket.get("params", {})
                    if method_params:
                        items = ", ".join(f"{k}={v!r}" for k, v in method_params.items())
                        lines.append(f"      {method_name}({items})")
                    else:
                        lines.append(f"      {method_name}: (no parameters)")

        if not functions and not methods:
            lines.append("  (empty)")

        lines.append(")")
        return "\n".join(lines)
    
    def update(self,
               params_source: str | dict[str, Any] | None = {},
               **kwargs) -> None:
        """Update configuration parameters.

        Accepts parameters from a YAML file, a dictionary, or keyword
        arguments.  When both *params_source* and **kwargs* are given,
        a warning is issued and **kwargs* take priority.

        Parameters
        ----------
        params_source : str or dict or None
            Path to a ``.yaml`` file, or a dictionary with parameter
            overrides.  The dictionary supports two modes:

            * **Structured mode** - keys are callables (or their string
              names), values are dicts of parameters for that specific
              function::

                  cfg.update({msalign: {"smooth_window": 0.15}})

            * **Flat mode** - keys are plain parameter names, values
              are new values.  Every known method/function group that
              contains that parameter name is updated::

                  cfg.update({"smooth_window": 0.15})

        **kwargs
            Additional parameter overrides applied on top of
            *params_source*.
        """
        # --- Merge all sources into a single dict ---
        func_params: dict[str, Any] = {}

        if params_source:
            if isinstance(params_source, str):
                path = params_source
                if not path.endswith(".yaml"):
                    path += ".yaml"
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        loaded = yaml.load(f, Loader=yaml.FullLoader)
                        if loaded:
                            func_params.update(loaded)
                else:
                    warnings.warn(
                        f"YAML file '{path}' not found - skipping."
                    )
            elif isinstance(params_source, dict):
                func_params.update(params_source)
            else:
                warnings.warn(
                    f"params_source must be a str (YAML path) or dict, "
                    f"got {type(params_source).__name__} - skipping."
                )

        if kwargs:
            if params_source:
                warnings.warn(
                    "Both params_source and **kwargs provided. "
                    "**kwargs will override params_source on conflict."
                )
            func_params.update(kwargs)

        if not func_params:
            return

        # --- Distribute to the correct function groups ---
        nested_groups: dict[str, dict[str, Any]] = {}
        flat_overrides: dict[str, Any] = {}

        for key, value in func_params.items():
            if key in ("cls_init_params", "params") and isinstance(value, dict):
                nested_groups[key] = value
            else:
                flat_overrides[key] = value

        structured_mode = False
        for key in flat_overrides:
            if callable(key):
                structured_mode = True
                break
            if isinstance(key, str):
                try:
                    candidate = eval(key)
                    if callable(candidate):
                        structured_mode = True
                        break
                except (ValueError, NameError):
                    continue

        if structured_mode:
            self._update_structured(flat_overrides)
            return

        self._update_flat(flat_overrides)

        if nested_groups:
            self._update_structured(nested_groups)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _update_structured(self,
                           func_params: dict[str | Callable, dict[str, Any]]
                           ) -> None:
        """Apply overrides where keys are callables (or their string names)."""
        methods = self.configs.setdefault("methods", {})
        functions = self.configs.setdefault("functions", {})

        for func_key, params in func_params.items():
            # Resolve string to callable (SAFE - no eval)
            if isinstance(func_key, str):
                try:
                    func = eval(func_key)
                    if not callable(func):
                        raise ValueError
                except ValueError:
                    warnings.warn(
                        f"Skipping unknown function '{func_key}' - "
                        f"cannot resolve to a callable."
                    )
                    continue
            elif callable(func_key):
                func = func_key
            else:
                warnings.warn(
                    f"Skipping invalid key '{func_key!r}' - "
                    f"expected a callable or a string name."
                )
                continue

            # Method (bound to a class)
            if hasattr(func, "__self__"):
                cls = func.__self__.__class__
                cls_name = cls.__name__
                func_name = func.__name__

                cls_bucket = methods.setdefault(cls_name, {})
                func_bucket = cls_bucket.setdefault(func_name, {})

                cls_init = params.pop("cls_init_params", None)
                method_params = params.pop("params", params)

                if cls_init is not None and isinstance(cls_init, dict):
                    init_bucket = func_bucket.setdefault("cls_init_params", {})
                    self._merge_params(init_bucket, cls_init, func_key)

                param_bucket = func_bucket.setdefault("params", {})
                self._merge_params(param_bucket, method_params, func_key)

            # Plain function
            else:
                func_name = func.__name__
                func_bucket = functions.setdefault(func_name, {})
                self._merge_params(func_bucket, params, func_key)

    def _find_param_locations(
        self, param_name: str
    ) -> tuple[list[str], list[tuple[dict[str, Any], str]]]:
        """Find all locations where *param_name* exists in the config.

        Searches through both ``methods`` and ``functions`` dicts.

        Parameters
        ----------
        param_name : str
            The parameter name to search for.

        Returns
        -------
        locations : list[str]
            Human-readable location descriptions (for warnings).
        refs : list[tuple[dict, str]]
            Mutable references ``(bucket_dict, key)`` for each occurrence,
            allowing the caller to update or delete in-place.
        """
        methods = self.configs.get("methods", {})
        functions = self.configs.get("functions", {})

        locations: list[str] = []
        refs: list[tuple[dict[str, Any], str]] = []

        for cls_name, func_dict in methods.items():
            for func_name, func_bucket in func_dict.items():
                param_bucket = func_bucket.get("params")
                if param_bucket is not None and param_name in param_bucket:
                    locations.append(f"{cls_name}.{func_name} (params)")
                    refs.append((param_bucket, param_name))
                init_bucket = func_bucket.get("cls_init_params")
                if init_bucket is not None and param_name in init_bucket:
                    locations.append(f"{cls_name}.{func_name} (cls_init_params)")
                    refs.append((init_bucket, param_name))

        for func_name, func_bucket in functions.items():
            if param_name in func_bucket:
                locations.append(f"{func_name} (function)")
                refs.append((func_bucket, param_name))

        return locations, refs

    def _update_flat(self, flat_overrides: dict[str, Any]) -> None:
        """Search all known method/function groups and update matching keys.

        When a parameter name is found in **multiple** functions, a warning
        is issued listing all matches.  Use **structured mode** to target a
        specific function unambiguously::

            cfg.update({msalign: {"smooth_window": 0.15}})
            cfg.update({"msalign": {"smooth_window": 0.15}})
        """
        for param_name, value in flat_overrides.items():
            locations, refs = self._find_param_locations(param_name)

            if not locations:
                warnings.warn(
                    f"Parameter '{param_name}' not found in any known "
                    f"function group - ignoring."
                )
                continue

            # Apply the value to all locations
            for bucket, key in refs:
                bucket[key] = value

            # Warn about ambiguity
            if len(locations) > 1:
                loc_list = "\n  - ".join(locations)
                warnings.warn(
                    f"Parameter '{param_name}' was found in {len(locations)} locations:\n"
                    f"  - {loc_list}\n"
                    f"Value '{value}' has been applied to ALL of them.\n"
                    f"To target a specific function, use structured mode:\n"
                    f"    cfg.update({{func_object: {{'{param_name}': {value!r}}}}})\n"
                    f"    cfg.update({{\"func_name\": {{'{param_name}': {value!r}}}}})"
                )

    def delete(self,
               param_name: str | tuple[str, ...]) -> None:
        """Delete a configuration parameter by name or path.

        Supports the same indexing as :meth:`__getitem__`:

        * **Flat mode** — plain parameter name::

              cfg.delete("smooth_window")

          If the name appears in **multiple** locations, deletion is
          **refused** and a warning is issued.

        * **Structured path** — dotted string or tuple::

              cfg.delete("msalign.smooth_window")
              cfg.delete("Baseline.snip.lam")
              cfg.delete(("msalign", "smooth_window"))
              cfg.delete(("Baseline", "snip", "params", "lam"))

        Parameters
        ----------
        param_name : str or tuple of str
            The parameter to delete.  A plain ``str`` triggers flat-mode
            lookup; a dotted ``str`` or ``tuple`` navigates the config
            hierarchy directly.
        """
        # --- Structured path (tuple or dotted string) ---
        if isinstance(param_name, tuple):
            self._delete_by_path(list(param_name))
            return

        if isinstance(param_name, str) and "." in param_name:
            parts = param_name.split(".")
            self._delete_by_path(parts)
            return

        # --- Flat mode (plain string) ---
        if isinstance(param_name, str):
            locations, refs = self._find_param_locations(param_name)

            if not locations:
                warnings.warn(
                    f"Parameter '{param_name}' not found in any known "
                    f"function group - ignoring."
                )
                return

            if len(locations) > 1:
                loc_list = "\n  - ".join(locations)
                warnings.warn(
                    f"Parameter '{param_name}' is ambiguous — found in "
                    f"{len(locations)} locations:\n"
                    f"  - {loc_list}\n"
                    f"Refusing to delete. Use a qualified path to "
                    f"disambiguate:\n"
                    f"    cfg.delete(\"func_name.{param_name}\")\n"
                    f"    cfg.delete(\"ClassName.method_name.{param_name}\")"
                )
                return

            # Exactly one location — safe to delete
            bucket, key = refs[0]
            deleted_value = bucket.pop(key, None)
            if deleted_value is not None:
                print(
                    f"Deleted '{param_name}' from {locations[0]}."
                )
            return

        warnings.warn(
            f"param_name must be a str or tuple of str, "
            f"got {type(param_name).__name__} - ignoring."
        )

    def _delete_by_path(self, parts: list[str]) -> None:
        """Navigate config hierarchy following *parts* and delete the last key.

        Minimal indexing — resolves class/method/function names automatically::

            cfg.delete("msalign.smooth_window")     # function.param
            cfg.delete("Baseline.snip.lam")          # class.method.param
            cfg.delete(("Baseline", "snip", "lam"))  # tuple form

        Parameters
        ----------
        parts : list of str
            Path components to the parameter to delete.
        """
        if not parts:
            return

        methods: dict = self.configs.get("methods", {})
        functions: dict = self.configs.get("functions", {})

        first, *rest = parts

        # --- Class.method.param or Class.method.bucket.param ---
        if first in methods:
            if not rest:
                warnings.warn(
                    f"Expected at least a method name after "
                    f"'{first}' - ignoring."
                )
                return
            second, *tail = rest
            if second not in methods[first]:
                warnings.warn(
                    f"Method '{second}' not found in class "
                    f"'{first}' - ignoring."
                )
                return
            bucket = methods[first][second]

            # If next part is a known bucket name, use it; else default to "params"
            if tail and tail[0] in _KNOWN_BUCKETS:
                *keys_to_navigate, last_key = tail[1:] if len(tail) > 1 else []
                current = bucket.get(tail[0])
                if current is None:
                    warnings.warn(
                        f"Bucket '{tail[0]}' not found in "
                        f"'{first}.{second}' - ignoring."
                    )
                    return
            else:
                keys_to_navigate = tail[:-1] if len(tail) > 1 else []
                last_key = tail[-1] if tail else None
                current = bucket.get("params")

            if last_key is None:
                warnings.warn(
                    f"Expected a parameter name after "
                    f"'{first}.{second}' - ignoring."
                )
                return

            if current is None:
                warnings.warn(
                    f"Parameter bucket not found in "
                    f"'{first}.{second}' - ignoring."
                )
                return

            # Navigate nested keys if any
            for k in keys_to_navigate:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    warnings.warn(
                        f"Key '{k}' not found in "
                        f"'{'.'.join(parts)}' - ignoring."
                    )
                    return

            if isinstance(current, dict) and last_key in current:
                current.pop(last_key)
            else:
                warnings.warn(
                    f"Key '{last_key}' not found in "
                    f"'{'.'.join(parts)}' - ignoring."
                )
            return

        # --- Function.param ---
        if first in functions:
            if not rest:
                warnings.warn(
                    f"Expected a parameter name after "
                    f"'{first}' - ignoring."
                )
                return
            *keys_to_navigate, last_key = rest
            current = functions[first]
            for k in keys_to_navigate:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    warnings.warn(
                        f"Key '{k}' not found in "
                        f"'{'.'.join(parts)}' - ignoring."
                    )
                    return
            if isinstance(current, dict) and last_key in current:
                current.pop(last_key)
            else:
                warnings.warn(
                    f"Key '{last_key}' not found in "
                    f"'{'.'.join(parts)}' - ignoring."
                )
            return

        # --- Unknown ---
        warnings.warn(
            f"Unknown function or class '{first}' - ignoring."
        )

    def _merge_params(self,
                      target: dict[str, Any],
                      source: dict[str, Any],
                      func_key: str | Callable) -> None:
        """Update *target* with *source*.

        When a key is not yet in *target*, inspects the actual callable's
        signature (if available).  The key is accepted if:

        * it is a named parameter of the callable, **or**
        * the callable accepts ``**kwargs`` (``VAR_KEYWORD``).

        Otherwise a warning is issued and the key is skipped.
        """
        # Resolve callable for signature inspection
        func: Callable | None = None
        if callable(func_key):
            func = func_key
        elif isinstance(func_key, str):
            try:
                candidate = eval(func_key)
                if callable(candidate):
                    func = candidate
            except (ValueError, NameError):
                pass

        # Pre-compute the set of valid parameter names and **kwargs flag
        valid_params: set[str] = set()
        has_kwargs: bool = False
        if func is not None:
            try:
                sig = inspect.signature(func)
                for p_name, p in sig.parameters.items():
                    valid_params.add(p_name)
                    if p.kind == inspect.Parameter.VAR_KEYWORD:
                        has_kwargs = True
            except (ValueError, TypeError):
                pass  # cannot inspect — fall through to strict mode

        for key, value in source.items():
            if key in target:
                target[key] = value
            elif key in valid_params:
                target[key] = value
            elif has_kwargs:
                target[key] = value
                warnings.warn(
                    f"Parameter '{key}' was not found among the known "
                    f"parameters of '{func_key}', but was added because "
                    f"the function accepts **kwargs."
                )
            else:
                warnings.warn(
                    f"Parameter '{key}' is not a known parameter of "
                    f"'{func_key}' and the function does not accept "
                    f"**kwargs - ignoring."
                )

    def set_method(self,
                   cls: type | str,
                   method_name: str,
                   delete_old: bool = True,
                   **params) -> None:
        """Set or replace a method's configuration in the config.

        Inspects the actual class to verify the method exists, extracts
        its default parameters, and applies any overrides from *params*.

        Parameters
        ----------
        cls : type or str
            The class (or its name as a string) that owns the method.
        method_name : str
            The method name to configure.
        delete_old : bool
            If ``True`` (default), removes all existing methods of this
            class before adding the new one.
        **params
            Parameter overrides.  If the dict contains ``cls_init_params``
            or ``params`` keys, those sub-dicts are used for the
            respective buckets.  Otherwise all keys are treated as
            method parameters (``params`` bucket).
        """
        # --- Resolve class ---
        if isinstance(cls, str):
            try:
                cls_obj = eval(cls)
            except (ValueError, NameError):
                warnings.warn(
                    f"Unknown class '{cls}' - cannot resolve. "
                    f"Skipping set_method."
                )
                return
        else:
            cls_obj = cls

        if not isinstance(cls_obj, type):
            warnings.warn(
                f"'{cls_obj}' is not a class. Skipping set_method."
            )
            return

        cls_name = cls_obj.__name__

        # --- Verify method exists on the actual class ---
        method_func = getattr(cls_obj, method_name, None)
        if method_func is None or not callable(method_func):
            warnings.warn(
                f"Method '{method_name}' not found in class "
                f"'{cls_name}' - skipping set_method."
            )
            return

        # --- Optionally delete old methods of this class ---
        methods = self.configs.setdefault("methods", {})
        if delete_old and cls_name in methods:
            del methods[cls_name]

        # --- Extract default parameters via _inspect_defaults ---
        defaults = _inspect_defaults([(method_func, cls_obj)])
        method_defaults = (
            defaults.get("methods", {})
            .get(cls_name, {})
            .get(method_name, {})
        )

        # Write defaults into config
        cls_bucket = methods.setdefault(cls_name, {})
        func_bucket = cls_bucket.setdefault(method_name, {})
        func_bucket.setdefault("cls_init_params",
                               dict(method_defaults.get("cls_init_params", {})))
        func_bucket.setdefault("params",
                               dict(method_defaults.get("params", {})))

        # --- Apply overrides from **params ---
        if not params:
            return

        if "cls_init_params" in params or "params" in params:
            cls_init = params.get("cls_init_params")
            if cls_init is not None and isinstance(cls_init, dict):
                init_bucket = func_bucket.setdefault("cls_init_params", {})
                self._merge_params(init_bucket, cls_init,
                                   f"{cls_name}.{method_name}")

            method_params = params.get("params", {})
            if isinstance(method_params, dict):
                param_bucket = func_bucket.setdefault("params", {})
                self._merge_params(param_bucket, method_params,
                                   f"{cls_name}.{method_name}")
        else:
            param_bucket = func_bucket.setdefault("params", {})
            self._merge_params(param_bucket, params,
                               f"{cls_name}.{method_name}")


class PipelineConfigurator(Configs):
    def __init__(self,
                configs_source: str | dict = {},
                preprocess_function: Callable = preprocess_configuration_base,
                process_pipeline: Callable = process_spectra_base,
                peakpick_function: Callable = peakpicking_base,
                **kwargs):
        # Store pipeline function references # TODO добавить базовые функции обработки в дефолтные значения
        self._preprocess_function: Callable = preprocess_function
        self._process_function: Callable = process_pipeline
        self._peakpick_function: Callable = peakpick_function

        # 1. Get default params from functions itself
        self.configs: dict[str, Any] = {}

        # Collect pipeline functions for default extraction,
        # keeping track of which step each function belongs to.
        self._step_func_names: dict[str, set[str]] = {
            'preprocess': set(),
            'process': set(),
            'peakpick': set(),
        }
        pipeline_funcs: list[Callable] = []

        for step_name, step_func in [
            ('preprocess', preprocess_function),
            ('process', process_pipeline),
            ('peakpick', peakpick_function),
        ]:
            extracted = PipelineConfigurator._extract_called_functions(step_func)
            pipeline_funcs.extend(extracted)
            # Record function names for this step
            for item in extracted:
                if isinstance(item, tuple) and len(item) == 2:
                    func, cls = item
                    self._step_func_names[step_name].add(func.__name__)
                elif callable(item):
                    self._step_func_names[step_name].add(item.__name__)

        # Extract defaults from pipeline functions
        self.configs = _inspect_defaults(pipeline_funcs)

        # 2. Load params from YAML or dict and override with kwargs
        self.update(configs_source, **kwargs)

    # ------------------------------------------------------------------ #
    #  Step-specific config access                                       #
    # ------------------------------------------------------------------ #

    def get_step_configs(self, step_name: str) -> dict[str, dict[str, Any]]:
        """Return configs for a specific pipeline step.

        Parameters
        ----------
        step_name : str
            One of ``'preprocess'``, ``'process'``, ``'peakpick'``.

        Returns
        -------
        dict
            A sub-dict of ``self.configs`` containing only the functions
            that belong to the requested step::

                {
                    "methods": {
                        "ClassName": {
                            "method_name": {"cls_init_params": ..., "params": ...}
                        }
                    },
                    "functions": {
                        "func_name": {param: value, ...}
                    }
                }

        Raises
        ------
        KeyError
            If *step_name* is not one of the known steps.
        """
        if step_name not in self._step_func_names:
            raise KeyError(
                f"Unknown pipeline step '{step_name}'. "
                f"Available steps: {list(self._step_func_names.keys())}"
            )

        func_names = self._step_func_names[step_name]
        result: dict[str, Any] = {}

        # Filter methods
        methods = self.configs.get("methods", {})
        filtered_methods: dict[str, Any] = {}
        for cls_name, func_dict in methods.items():
            for method_name in list(func_dict.keys()):
                if method_name in func_names:
                    if cls_name not in filtered_methods:
                        filtered_methods[cls_name] = {}
                    filtered_methods[cls_name][method_name] = func_dict[method_name]
        if filtered_methods:
            result["methods"] = filtered_methods

        # Filter functions
        functions = self.configs.get("functions", {})
        filtered_funcs = {
            fname: fparams
            for fname, fparams in functions.items()
            if fname in func_names
        }
        if filtered_funcs:
            result["functions"] = filtered_funcs

        return result


    # ------------------------------------------------------------------ #
    #  YAML serialisation                                                #
    # ------------------------------------------------------------------ #
    def save(self, path2dir: str = "",
             file_name: str = "processing_recipe") -> None:
        """Alias for :meth:`dump`."""
        self.dump(path2dir, file_name)

    def dump(self, path2dir: str = "",
             file_name: str = "processing_recipe") -> None:
        """Save the full ``self.configs`` dictionary to a YAML file.

        The output preserves the internal structure::

            methods:
              ClassName:
                method_name:
                  cls_init_params: {param: value, ...}
                  params: {param: value, ...}
            functions:
              func_name: {param: value, ...}

        Parameters
        ----------
        path2dir : str
            Directory path or full file path.  If a directory, the filename
            is auto-generated using ``file_end``.
        file_end : str
            Suffix appended to the filename (before ``.yaml``).
        """
        if not path2dir.endswith(".yaml"):
            if path2dir.endswith(file_name):
                path = path2dir + ".yaml"
            else:
                path = os.path.join(path2dir, file_name+ ".yaml") 
        else:
            path = path2dir

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.configs, f, default_flow_style=False, sort_keys=True)

        # Save companion .py file with custom pipeline functions
        py_path = path[:-5] + ".py" if path.endswith(".yaml") else path + ".py"
        self._dump_functions(py_path)

    # ------------------------------------------------------------------ #
    #  Custom functions serialisation (companion .py file)                #
    # ------------------------------------------------------------------ #

    def _dump_functions(self, py_path: str) -> None:
        """Save source code of custom pipeline functions to a companion .py file.

        The file contains the source of ``_preprocess_function``,
        ``_process_function``, and ``_peakpick_function``, followed by an
        ``__pipeline_functions__`` mapping dict that is used by
        :meth:`_load_functions_from_py` to restore them.

        Parameters
        ----------
        py_path : str
            Full path to the ``.py`` file to write.
        """
        func_map: dict[str, Callable] = {
            '_preprocess_function': self._preprocess_function,
            '_process_function': self._process_function,
            '_peakpick_function': self._peakpick_function,
        }

        lines: list[str] = []
        lines.append("# Custom pipeline functions for PipelineConfigurator\n")
        lines.append("# Auto-generated - do not edit manually\n\n")
        lines.append("import numpy as np\n\n")

        for attr_name, func in func_map.items():
            source = self._get_source(func)
            if source:
                lines.append(f"# --- {attr_name} ---\n")
                lines.append(source)
                lines.append("\n\n")
            else:
                lines.append(f"# {attr_name}: {func.__name__!r} (source not available)\n\n")

        # Append the mapping dict at the end
        lines.append("# Auto-generated function mapping\n")
        lines.append("__pipeline_functions__ = {\n")
        for attr_name, func in func_map.items():
            source = self._get_source(func)
            if source:
                lines.append(f"    '{attr_name}': {func.__name__},\n")
        lines.append("}\n")

        with open(py_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def _load_functions_from_py(py_path: str) -> dict[str, Callable]:
        """Load custom pipeline functions from a companion ``.py`` file.

        Expects the file to contain an ``__pipeline_functions__`` dict
        mapping attribute names (e.g. ``'_preprocess_function'``) to
        callables.

        Parameters
        ----------
        py_path : str
            Full path to the ``.py`` file.

        Returns
        -------
        dict[str, Callable]
            Mapping of attribute name → callable, or empty dict on failure.
        """
        if not os.path.exists(py_path):
            return {}

        with open(py_path, "r", encoding="utf-8") as f:
            source = f.read()

        namespace: dict[str, Any] = {}
        try:
            exec(source, namespace)
        except Exception as e:
            warnings.warn(
                f"Failed to load custom functions from '{py_path}': {e}",
                UserWarning,
            )
            return {}

        return namespace.get('__pipeline_functions__', {})

    # ------------------------------------------------------------------ #
    #  Full config replacement from YAML                                  #
    # ------------------------------------------------------------------ #

    def _load(self, yaml_path: str) -> None:
        """Fully replace the current configuration from a YAML file.

        Unlike :meth:`update`, this method **completely replaces**
        ``self.configs`` with the contents of the YAML file (no merging).

        If a companion ``.py`` file with the same base name exists next to
        the YAML file, custom pipeline functions are restored from it via
        :meth:`_load_functions_from_py`.

        Parameters
        ----------
        yaml_path : str
            Path to the ``.yaml`` file (``.yaml`` extension is added
            automatically if missing).
        """
        if not yaml_path.endswith(".yaml"):
            yaml_path += ".yaml"

        py_path = yaml_path[:-5] + ".py"

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"YAML file not found: {yaml_path}"
            )

        with open(yaml_path, "rb") as f:
            loaded = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(loaded, dict):
            raise ValueError(
                f"YAML file must contain a top-level mapping (dict), "
                f"got {type(loaded).__name__}."
            )

        self._validate_configs_structure(loaded)

        # Fully replace configs (no merge)
        self.configs = loaded

        # Restore custom functions from companion .py
        funcs = self._load_functions_from_py(py_path)
        if funcs:
            if '_preprocess_function' in funcs:
                self._preprocess_function = funcs['_preprocess_function']
            if '_process_function' in funcs:
                self._process_function = funcs['_process_function']
            if '_peakpick_function' in funcs:
                self._peakpick_function = funcs['_peakpick_function']

    # ------------------------------------------------------------------ #
    #  Override update — YAML path triggers full replacement              #
    # ------------------------------------------------------------------ #

    def update(self,
               params_source: str | dict[str, Any] | None = None,
               **kwargs) -> None:
        """Update or replace configuration parameters.

        **Behaviour change from** :meth:`Configs.update`:

        * If *params_source* is a **YAML file path** (``str``), the current
          configuration is **fully replaced** (not merged) with a warning.
        * For a ``dict`` or ``**kwargs``, delegates to the parent
          :meth:`Configs.update` (partial update / merge).

        Parameters
        ----------
        params_source : str or dict or None
            YAML file path (triggers full replacement) or a dict with
            parameter overrides (partial update).
        **kwargs
            Additional parameter overrides.
        """
        if isinstance(params_source, str):
            warnings.warn(
                f"Loading from YAML file '{params_source}' will "
                f"**fully replace** the current configuration. "
                f"Use a dict for partial updates instead.",
                UserWarning,
            )
            self._load(params_source)
            # Apply kwargs on top if provided
            if kwargs:
                super().update(kwargs)
            return

        # Delegate to parent for dict / None / kwargs
        super().update(params_source, **kwargs)

    @staticmethod
    def _get_source(func: Callable) -> str | None:
        """Get source code of *func*, with fallbacks for Jupyter/IPython."""
        # 1. Standard inspect
        try:
            return inspect.getsource(func)
        except (OSError, TypeError):
            pass

        # 2. dill fallback (works in many Jupyter environments)
        try:
            import dill
            return dill.source.getsource(func)
        except Exception:
            pass

        # 3. IPython fallback
        try:
            import IPython
            ip = IPython.get_ipython()
            if ip is not None:
                return ip.object_inspect(func).source
        except Exception:
            pass

        return None

    @staticmethod
    def _extract_called_functions(func: Callable) -> list[Callable | tuple[Callable, type]]:
        """Parse the source of *func* with ``ast`` and return all callables
        that are invoked inside it.

        Supports:
        - Plain name calls: ``msalign(data, ...)``
        - Attribute calls on local variables: ``baseline.method(data, ...)``
          (traces variable assignments back to class instantiation)

        Python builtins (``enumerate``, ``len``, ...) and names that cannot be
        resolved are silently skipped.

        Parameters
        ----------
        func : callable
            An orchestrator function (e.g. ``Pipeline.process``).

        Returns
        -------
        list of callable or (callable, type) tuple
            Resolved callable objects for every name found in the AST.
            When a method is resolved via its owning class, a
            ``(method, class)`` tuple is returned so that the caller
            can extract ``cls_init_params``.
        """
        import ast
        import builtins

        source = PipelineConfigurator._get_source(func)
        if source is None:
            warnings.warn(
                f"Cannot get source of '{func.__name__}'. "
                f"Pass the list of functions directly instead."
            )
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            warnings.warn(
                f"Syntax error while parsing {func.__name__}: {e}."
            )
            return []

        # --- Build a variable -> class/module map from simple assignments ---
        # e.g.  baseline = Baseline(mz_scale)  ->  {"baseline": "Baseline"}
        var_to_class: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Name):
                            var_to_class[target.id] = node.value.func.id

        # Pre-compute the set of Python builtins for fast lookup
        builtin_names: set[str] = set(dir(builtins))
        func_globals = getattr(func, "__globals__", {})

        resolved: list[Callable | tuple[Callable, type]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Name):
                # Plain name call: msalign(data, ...)
                name = node.func.id
                if name in builtin_names:
                    continue
                obj = func_globals.get(name) or globals().get(name)
                if obj is not None and callable(obj) and not inspect.isclass(obj):
                    resolved.append(obj)

            elif isinstance(node.func, ast.Attribute):
                # Attribute call: obj.method(data, ...)
                method_name = node.func.attr

                # Case A: obj is a simple name - could be a variable, module, or class
                if isinstance(node.func.value, ast.Name):
                    obj_name = node.func.value.id

                    # A1: obj is a local variable assigned from a class instantiation
                    cls_name = var_to_class.get(obj_name)
                    if cls_name is not None:
                        cls_obj = func_globals.get(cls_name) or globals().get(cls_name)
                        if cls_obj is not None and isinstance(cls_obj, type):
                            method_obj = getattr(cls_obj, method_name, None)
                            if method_obj is not None and callable(method_obj):
                                resolved.append((method_obj, cls_obj))
                                continue

                    # A2: obj is a module (e.g. np.linspace)
                    mod_obj = func_globals.get(obj_name) or globals().get(obj_name)
                    if mod_obj is not None:
                        method_obj = getattr(mod_obj, method_name, None)
                        if method_obj is not None and callable(method_obj):
                            resolved.append(method_obj)
                            continue

                    # A3: obj is a class (e.g. Baseline.asls)
                    if mod_obj is not None and isinstance(mod_obj, type):
                        method_obj = getattr(mod_obj, method_name, None)
                        if method_obj is not None and callable(method_obj):
                            resolved.append((method_obj, mod_obj))
                            continue

                # Case B: obj is a chained call - Baseline().asls(...)
                elif isinstance(node.func.value, ast.Call):
                    inner_call = node.func.value
                    if isinstance(inner_call.func, ast.Name):
                        cls_name = inner_call.func.id
                        cls_obj = func_globals.get(cls_name) or globals().get(cls_name)
                        if cls_obj is not None and isinstance(cls_obj, type):
                            method_obj = getattr(cls_obj, method_name, None)
                            if method_obj is not None and callable(method_obj):
                                resolved.append((method_obj, cls_obj))
                                continue

        return resolved

# --------------------------------------------------------------------------- #
#  PreparedDataSource - per-ROI configuration manager linked to a DataSource       #
# --------------------------------------------------------------------------- #


class PreparedDataSource:
    """
    Manages per-ROI :class:`PipelineConfigurator` linked to a :class:`DataSource`.

    Unlike :class:`PipelineConfigurator`, all modifications are **ROI-specific**:

    * Specify which ROI(s) to apply changes to (all if omitted).
    * Access per-ROI :class:`PipelineConfigurator` via ``cm[roi_name]``.
    * Saves to ``processed_pelmesha/<sample_name>_processing_recipe.yaml``.

    Parameters
    ----------
    configs_source : str or PipelineConfigurator or None
        Source for the base configuration.  Can be:

        * A :class:`PipelineConfigurator` instance - used directly as the base
          config template for all ROIs.
        * Path to a ``*_processing_recipe.yaml`` file (per-ROI or flat).
        * Path to a datasource file - the recipe is auto-resolved from
          ``processed_pelmesha/`` next to the datasource.
        * ``None`` - creates an empty manager; link later via :meth:`set_link`.

    datasource : DataSource or str or None
        :class:`DataSource` instance or path to a data-source file.
        When provided, per-ROI configs are created automatically.

    **kwargs
        Additional parameter overrides forwarded to each ROI's
        :class:`PipelineConfigurator`.

    Examples
    --------
    >>> cm = PreparedDataSource(datasource="/data/sample.imzml")
    >>> cm["00"]["msalign"]["shift_range"]
    [-0.95, 0.95]
    >>> cm.update({"smooth_window": 0.15}, rois=["00", "01"])
    >>> cm.save()
    """

    def __init__(self,
                 datasource: DataSource | str | None = None,
                 configs_source: str | PipelineConfigurator | None = None,
                 **kwargs):
        #: Path to the source file that was used to load configs.
        self._configs_source_path: str | None = None
        #: Linked DataSource (optional).
        self._datasource: DataSource | None = None
        #: Per-ROI PipelineConfigurator instances.
        self._roi_configs: dict[str, PipelineConfigurator] = {}
        #: Base config used as a template when new ROIs are added via set_link.
        self._base_configs: PipelineConfigurator | None = None

        # --- Resolve configs_source ---
        if configs_source is not None:
            self._load(configs_source, **kwargs)

        # --- Link datasource if provided ---
        if datasource is not None:
            self.set_link(datasource)

    # ------------------------------------------------------------------ #
    #  Internal loading helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_recipe_path(datasource_path: str) -> str:
        """Resolve the recipe path relative to a datasource file.

        Looks for ``processed_pelmesha/<sample_name>_processing_recipe.yaml``
        next to the datasource.
        """
        sample_name = os.path.splitext(os.path.basename(datasource_path))[0]
        folder_name = os.path.basename(os.path.dirname(datasource_path))
        if folder_name != sample_name:
            sample_name = folder_name + "_" + sample_name
        return os.path.join(
            os.path.dirname(datasource_path),
            "processed_pelmesha",
            f"{sample_name}_processing_recipe.yaml",
        )

    def _is_per_roi_yaml(self, data: dict) -> bool:
        """Heuristic: does *data* look like a per-ROI config dump?"""
        if not isinstance(data, dict):
            return False
        # Per-ROI files have top-level keys like "roi_00", "roi_01", ...
        # Flat PipelineConfigurator files have "methods" / "functions" at the top.
        top_keys = set(data.keys())
        if top_keys & {"methods", "functions"}:
            return False
        # If at least one key looks like a ROI name, treat as per-ROI.
        Not_roi_nests_keys = ("methods", "functions")
        return any(
            isinstance(k, str) and (k not in Not_roi_nests_keys)
            for k in data
        )

    def _load(self, source: str | PipelineConfigurator, **kwargs) -> None:
        """Load configuration from *source* into ``_roi_configs`` or ``_base_configs``.

        * If *source* is a :class:`PipelineConfigurator` instance, it is used
          directly as the base config template.
        * If *source* is a YAML file path, the companion ``.py`` file (same
          base name) is also loaded to restore custom pipeline functions.
        * Per-ROI YAML files (multiple ROIs in one file) create one
          :class:`PipelineConfigurator` per ROI, restoring per-ROI functions
          from the companion ``.py`` file.
        """
        # --- PipelineConfigurator instance: use directly as base config ---
        if isinstance(source, PipelineConfigurator):
            self._base_configs = source
            return

        # --- String source: YAML file or datasource path ---
        if isinstance(source, str):
            path = source
            if not path.endswith(".yaml"):
                path += ".yaml"

            if os.path.exists(path):
                with open(path, "rb") as f:
                    loaded = yaml.load(f, Loader=yaml.FullLoader)

                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"YAML file '{path}' must contain a top-level mapping (dict), "
                        f"got {type(loaded).__name__}."
                    )

                # Resolve companion .py path
                py_path = path[:-5] + ".py"

                if self._is_per_roi_yaml(loaded):
                    # Per-ROI config file - create one PipelineConfigurator per ROI
                    # Load per-ROI functions from companion .py
                    roi_funcs = PipelineConfigurator._load_functions_from_py(py_path)
                    for roi_name, roi_data in loaded.items():
                        Configs._validate_configs_structure(roi_data)
                        cc = PipelineConfigurator(roi_data)
                        # Restore per-ROI functions from companion .py
                        if roi_name in roi_funcs:
                            funcs = roi_funcs[roi_name]
                            if '_preprocess_function' in funcs:
                                cc._preprocess_function = funcs['_preprocess_function']
                            if '_process_function' in funcs:
                                cc._process_function = funcs['_process_function']
                            if '_peakpick_function' in funcs:
                                cc._peakpick_function = funcs['_peakpick_function']
                        if kwargs:
                            cc.update(kwargs)
                        self._roi_configs[roi_name] = cc
                    self._configs_source_path = path
                    return

                # Flat YAML - delegate to PipelineConfigurator (handles companion .py)
                self._base_configs = PipelineConfigurator(path, **kwargs)
                self._configs_source_path = path
                return

            # File does not exist - try as a datasource path
            recipe = self._resolve_recipe_path(source)
            if recipe and os.path.exists(recipe):
                self._load(recipe, **kwargs)
                return

            # Fallback: treat as a callable name (unlikely but safe)
            self._base_configs = PipelineConfigurator(source, **kwargs)
            self._configs_source_path = source
            return

        self._base_configs = PipelineConfigurator(source, **kwargs)

    # ------------------------------------------------------------------ #
    #  DataSource linking                                                #
    # ------------------------------------------------------------------ #

    def set_link(self, datasource: DataSource | str) -> None:
        """Link this :class:`PreparedDataSource` to a :class:`DataSource`.

        Creates per-ROI :class:`PipelineConfigurator` for every ROI found in the
        datasource's metadata.  Existing per-ROI configs are preserved.

        Parameters
        ----------
        datasource : DataSource or str
            :class:`DataSource` instance or path to a data-source file.
        """
        if isinstance(datasource, str):
            datasource = DataSource(datasource)

        self._datasource = datasource

        # Get ROI names from the datasource metadata
        roi_names: list[str] = list(datasource.roi_metadata.index)

        for roi in roi_names:
            if roi not in self._roi_configs:
                # Create a fresh PipelineConfigurator for this ROI
                if self._base_configs is not None:
                    self._roi_configs[roi] = copy.deepcopy(self._base_configs)
                else:
                    self._roi_configs[roi] = PipelineConfigurator()

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def path(self) -> str | None:
        """Path to the linked datasource file, if available."""
        if self._datasource is not None:
            return self._datasource.file_path
        return self._configs_source_path

    @property
    def sample_name(self) -> str | None:
        """Sample name derived from the linked datasource or source path."""
        if self._datasource is not None:
            return self._datasource.sample_name
        if self._configs_source_path:
            name = os.path.splitext(os.path.basename(self._configs_source_path))[0]
            if name.endswith("_processing_recipe"):
                name = name.replace("_processing_recipe", "")
            return name
        return None

    @property
    def rois(self) -> list[str]:
        """List of ROI names currently managed."""
        return list(self._roi_configs.keys())

    # ------------------------------------------------------------------ #
    #  ROI-specific access                                                 #
    # ------------------------------------------------------------------ #

    def __getitem__(self, roi: str) -> PipelineConfigurator:
        """Get the :class:`PipelineConfigurator` for a specific ROI.

        Parameters
        ----------
        roi : str
            ROI name (e.g. ``"00"``, ``"01"``).

        Returns
        -------
        PipelineConfigurator
            The full configuration object for this ROI.

        Raises
        ------
        KeyError
            If the ROI is not found.
        """
        if roi not in self._roi_configs:
            raise KeyError(
                f"ROI '{roi}' not found. "
                f"Available ROIs: {list(self._roi_configs.keys())}"
            )
        return self._roi_configs[roi]

    def __contains__(self, roi: str) -> bool:
        """Check whether a ROI is managed."""
        return roi in self._roi_configs

    def __len__(self) -> int:
        """Number of managed ROIs."""
        return len(self._roi_configs)

    def __iter__(self):
        """Iterate over ROI names."""
        return iter(self._roi_configs)

    # ------------------------------------------------------------------ #
    #  Parameter modification (ROI-aware)                                  #
    # ------------------------------------------------------------------ #

    def update(self,
               params_source: str | dict[str, Any] | None = None,
               rois: str | list[str] | None = None,
               **kwargs) -> None:
        """Update configuration parameters for specific ROI(s).

        If *rois* is ``None``, the update is applied to **all** ROIs.

        Parameters
        ----------
        params_source : str or dict or None
            Path to a YAML file, or a dictionary with parameter overrides.
            Same format as :meth:`PipelineConfigurator.update`.
        rois : str or list of str or None
            ROI(s) to update.  ``None`` means all ROIs.
        **kwargs
            Additional parameter overrides applied on top.
        """
        targets: list[str]
        if rois is None:
            targets = list(self._roi_configs.keys())
        elif isinstance(rois, str):
            targets = [rois]
        else:
            targets = list(rois)

        for roi in targets:
            if roi in self._roi_configs:
                self._roi_configs[roi].update(params_source, **kwargs)

    def delete(self,
               param_name: str | tuple[str, ...],
               rois: str | list[str] | None = None) -> None:
        """Delete a configuration parameter from specific ROI(s).

        If *rois* is ``None``, the deletion is applied to **all** ROIs.

        Parameters
        ----------
        param_name : str or tuple of str
            The parameter to delete.  Same format as
            :meth:`Configs.delete`.
        rois : str or list of str or None
            ROI(s) to delete from.  ``None`` means all ROIs.
        """
        targets: list[str]
        if rois is None:
            targets = list(self._roi_configs.keys())
        elif isinstance(rois, str):
            targets = [rois]
        else:
            targets = list(rois)

        for roi in targets:
            if roi in self._roi_configs:
                self._roi_configs[roi].delete(param_name)

    def setdefault(self, roi: str) -> PipelineConfigurator:
        """Ensure a ROI exists, creating it from base configs if needed.

        Parameters
        ----------
        roi : str
            ROI name.

        Returns
        -------
        PipelineConfigurator
            The config for this ROI (existing or newly created).
        """
        if roi not in self._roi_configs:
            if self._base_configs is not None:
                self._roi_configs[roi] = copy.deepcopy(self._base_configs)
            else:
                self._roi_configs[roi] = PipelineConfigurator()
        return self._roi_configs[roi]

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def _default_save_path(self) -> str:
        """Compute the default save path.

        Returns
        -------
        str
            ``<datasource_dir>/processed_pelmesha/<sample_name>_processing_recipe.yaml``
        """
        if self._datasource is not None:
            base_dir = os.path.dirname(self._datasource.file_path)
            name = self._datasource.sample_name
        # elif self._configs_source_path:
        #     base_dir = os.path.dirname(os.path.dirname(self._configs_source_path))
        #     name = self.sample_name or "unknown"
        else:
            base_dir = "."
            name = "unknown"

        return os.path.join(
            base_dir,
            "processed_pelmesha",
            f"{name}_processing_recipe.yaml",
        )

    def dump(self, path: str | None = None) -> str:
        """Save all per-ROI configurations to a YAML file.

        The output structure preserves each ROI's full config::

            roi_00:
              methods: ...
              functions: ...

        Also saves a companion ``.py`` file with the same base name
        containing the custom pipeline functions for each ROI.

        Parameters
        ----------
        path : str or None
            Full save path.  If ``None``, saves to
            ``processed_pelmesha/<sample_name>_processing_recipe.yaml``
            relative to the datasource file.

        Returns
        -------
        str
            The path the file was saved to.
        """
        if path is None:
            path = self._default_save_path()

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Build the full per-ROI config dict
        full_config: dict[str, Any] = {}
        for roi_name, config in self._roi_configs.items():
            full_config[roi_name] = config.configs

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=True)

        # Save companion .py with per-ROI pipeline functions
        py_path = path[:-5] + ".py" if path.endswith(".yaml") else path + ".py"
        self._dump_roi_functions(py_path)

        return path

    def _dump_roi_functions(self, py_path: str) -> None:
        """Save per-ROI custom pipeline functions to a companion .py file.

        The file contains an ``__pipeline_functions__`` dict mapping
        ROI names to their function mappings, e.g.::

            __pipeline_functions__ = {
                "roi_00": {
                    "_preprocess_function": <func>,
                    "_process_function": <func>,
                    "_peakpick_function": <func>,
                },
                ...
            }

        Parameters
        ----------
        py_path : str
            Full path to the ``.py`` file to write.
        """
        lines: list[str] = []
        lines.append("# Per-ROI custom pipeline functions for PreparedDataSource\n")
        lines.append("# Auto-generated - do not edit manually\n\n")
        lines.append("import numpy as np\n\n")

        # Collect all unique function sources
        seen_sources: dict[str, str] = {}  # func_name -> source
        roi_func_map: dict[str, dict[str, str]] = {}  # roi_name -> {attr_name: func_name}

        for roi_name, config in self._roi_configs.items():
            roi_func_map[roi_name] = {}
            for attr_name in ('_preprocess_function', '_process_function', '_peakpick_function'):
                func = getattr(config, attr_name, None)
                if func is None:
                    continue
                func_name = func.__name__
                roi_func_map[roi_name][attr_name] = func_name
                if func_name not in seen_sources:
                    source = config._get_source(func)
                    if source:
                        seen_sources[func_name] = source

        # Write function sources
        for func_name, source in seen_sources.items():
            lines.append(f"# --- {func_name} ---\n")
            lines.append(source)
            lines.append("\n\n")

        # Write the mapping dict
        lines.append("# Auto-generated per-ROI function mapping\n")
        lines.append("__pipeline_functions__ = {\n")
        for roi_name, funcs in roi_func_map.items():
            lines.append(f"    '{roi_name}': {{\n")
            for attr_name, func_name in funcs.items():
                lines.append(f"        '{attr_name}': {func_name},\n")
            lines.append("    },\n")
        lines.append("}\n")

        with open(py_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def save(self, path: str | None = None) -> str:
        """Alias for :meth:`dump`."""
        return self.dump(path)

    # ------------------------------------------------------------------ #
    #  Representation                                                     #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lines = ["PreparedDataSource("]
        if self._datasource is not None:
            lines.append(f"  datasource: {self._datasource.sample_name}")
        lines.append(f"  rois: {list(self._roi_configs.keys())}")
        lines.append(")")
        return "\n".join(lines)
        

class Pipeline:
    def __init__(self,
                 prepdata: PreparedDataSource):
        '''WIP
        Unified interface for running MSI data processing.

        Pipeline is a thin orchestrator that accepts a PreparedDataSource 
        and runs processing methods using the pipeline functions stored in the configs object.

        Parameters
        ----------
        configs : PreparedDataSource
            Configuration object with datasource and pipeline functions.
        '''
        if isinstance(prepdata, PreparedDataSource):
            self.prepdata = prepdata
            self._roi_configs = prepdata._roi_configs
            self._configs_source_path = prepdata._configs_source_path
            self._datasource = prepdata._datasource
        else:
            raise ValueError(
                "Provide a PreparedDataSource."
            )
    def _multistream_pipeline(self,
                              process_wrapper: Callable,
                              roi: str = None,
                              cpu_num: int = 1, 
                              Ram_GB_limit: int = 2,
                              dtypeconv:  np.dtype | str | None = None,
                              idxs: Indexator | SliceIndexator | int | None = None):
        """Основная функция генератор результатов мультипроцессинга"""

        datasource = self._datasource
        if dtypeconv is None:
            dtypeconv = datasource.metadata["dtype_raw"]
        dtypeconv = np.dtype(dtypeconv)
        roi_metadata = datasource.roi_metadata
        rmeta = roi_metadata.loc[r]
        if idxs is None:
            idxs = rmeta["idxroi"]
        # Get per-ROI PipelineConfigurator pipeline functions and its configs from PreparedDataSource
        roi_configs = self._roi_configs[roi]
        preprocess_function = roi_configs._preprocess_function

        internal_configs = {}
        if process_wrapper.__name__ == "_procfunc_wrapper":
            internal_configs['process_pipeline']= roi_configs._process_function
            wrapper_configs = roi_configs.get_step_configs("process")
        elif process_wrapper.__name__ == "_peakpick_wrapper":
            internal_configs['process_pipeline']  = roi_configs._process_function
            internal_configs['peakpick_function']  = roi_configs._peakpick_function
            wrapper_configs = {"peakpick":roi_configs.get_step_configs("peakpick"), 
                               "process":roi_configs.get_step_configs("process")}
        else:
            raise ValueError("Unknown process_wrapper function")
        
        mz = None
        if preprocess_function:
            preprocess_configs = roi_configs.get_step_configs("preprocess")
            mz, internal_configs = preprocess_function(datasource, roi, rmeta, **preprocess_configs)
        else:
            if datasource.loader.dcont:
                mz = datasource.loader.mz_scale_cont

        yield mz # return common mz to spectra. if mz is not common, return None

        idxs_batches = datasource.split_idxs(idxs = idxs,cpu_count=cpu_num, Ramcap_GB = Ram_GB_limit)
        partial_worker = partial(process_wrapper,
                    datasource = datasource,
                    configs = wrapper_configs,
                    dtypeconv = dtypeconv,
                    **internal_configs
                    )

        with Pool(cpu_num) as p:
            for loc_idxs, data_int in tqdm(p.imap_unordered(partial_worker, idxs_batches), total=len(idxs_batches), unit = 'batch'):
                yield loc_idxs, data_int


    def process(self,
                free_cpus: int = 1, 
                draw: bool = False, 
                draw_mz_range: tuple[float, float] | None = None,
                draw_spctrum_idx: int | None = None,
                Ram_GB_limit: float = 2,
                h5chunk_size_MB: int = 10,
                dtypeconv: np.dtype | str | None = None):
        datasource = self._datasource
        
        hdf5_save_path = os.path.join(os.path.split(datasource.file_path)[0],'processed_pelmesha',datasource.sample_name + '_processed_spectra.hdf5')
        if os.path.exists(hdf5_save_path):
            os.remove(hdf5_save_path)
        cpu_num = cpu_count()-free_cpus
        bytes_flsize = dtypeconv.itemsize
        chunk_size_by_elements = int(max(1,np.ceil(h5chunk_size_MB*(1024**2)/bytes_flsize)))
        roi_metadata = datasource.roi_metadata
        for roi in roi_metadata.index:
            print(f'Processing ROI {roi}')
            processing_stream = self._multistream_pipeline(self._procfunc_wrapper,
                                                           roi = roi,
                                                           cpu_num = cpu_num,
                                                           Ram_GB_limit = Ram_GB_limit,
                                                           dtypeconv = dtypeconv)
            gen_mz = next(processing_stream)
            if gen_mz is None:
                processing_stream.close()
                warnings.warn(f"Discontinuous data detected for sample '{datasource.sample_name}' (ROI: {roi}). "  
                              "Resampling is required before writing to HDF5.")
                
                continue
            
            with File(hdf5_save_path,"a") as hdf5:
                dots_num = len(gen_mz)
                hdf5.create_dataset(roi+"/int", (Indexator(roi_metadata.loc[roi,"idxroi"]).count, dots_num), chunks=(chunk_size_by_elements/dots_num, dots_num), dtype = dtypeconv)
                hdf5.create_dataset(roi+"/mz", data = gen_mz, dtype = dtypeconv)
                
                for loc_idxs, data_int in processing_stream:
                    hdf5[roi]["int"][loc_idxs,:] = data_int

            if draw:
                Drawer(self.prepdata).draw_processed_data(roi, draw_mz_range, draw_spctrum_idx)

        self.prepdata.save()
        

    def peakpick(self,
                 free_cpus: int = 1, 
                 draw: bool = False, 
                 draw_mz_range: tuple[float, float] | None = None,
                 draw_spctrum_idx: int | None = None,
                 Ram_GB_limit: float = 2,
                 h5chunk_size_MB: int = 10,
                 dtypeconv: np.dtype | str | None = None):         
        datasource = self._datasource
        
        hdf5_save_path = os.path.join(os.path.split(datasource.file_path)[0],'processed_pelmesha',datasource.sample_name + '_peaklists.hdf5')
        if os.path.exists(hdf5_save_path):
            os.remove(hdf5_save_path)
        cpu_num = cpu_count()-free_cpus
        bytes_flsize = dtypeconv.itemsize
        chunk_size_by_elements = int(max(1,np.ceil(h5chunk_size_MB*(1024**2)/bytes_flsize)))
        roi_metadata = datasource.roi_metadata
        for roi in roi_metadata.index:
            print(f'Processing ROI {roi}')
            peakpicking_stream = self._multistream_pipeline(self._peakpick_wrapper,
                                                           roi = roi,
                                                           cpu_num = cpu_num,
                                                           Ram_GB_limit = Ram_GB_limit,
                                                           dtypeconv = dtypeconv)
            gen_mz = next(peakpicking_stream)
            
            roi_configs = self._roi_configs[roi]
            peakpick_function = roi_configs._peakpick_function
            headers = roi_configs[peakpick_function.__name__]['headers']
            with File(hdf5_save_path,"a") as hdf5:
                n_heads = len(headers)
                hdf5.create_dataset(roi + "/peaklists",(0, n_heads), maxshape = (None, n_heads), chunks=(chunk_size_by_elements/n_heads, n_heads), dtype=dtypeconv)
                hdf5[roi][peaklists].attrs["Column headers"] = headers
                for peaklists in peakpicking_stream:
                    list_size = len(peaklists)
                    hdf5[roi]["peaklists"].resize((hdf5[roi]["peaklists"].shape[0] + list_size, n_heads))
                    hdf5[roi]["peaklists"][-list_size:,:] = peaklists

            if draw:
                Drawer(self.prepdata).draw_processed_data(roi, draw_mz_range, draw_spctrum_idx)

        self.prepdata.save()

        

    @staticmethod
    def _procfunc_wrapper(idxs: Indexator | SliceIndexator | tuple| np.ndarray,
                          datasource: DataSource,
                          configs: dict | Configs | PipelineConfigurator,
                          dtypeconv: np.dtype | None = None,
                          **internal_configs
                          ):
        process_function = internal_configs.pop("process_pipeline")
        loc_idxs = datasource._get_local_roi_idx(idxs)
        if datasource.metadata['continuous']:
            mz = datasource.source.get_mz(idxs[0])
            data_int = np.asarray(np.vstack(tuple(datasource.source.get_intensities_stream(idxs))), dtype=dtypeconv)
            
            ## proccessing array
            mz, data_int = process_function(mz, data_int, configs, **internal_configs)
            yield SliceIndexator(loc_idxs), data_int
        else:
            for loc_idx, (mz, data_int) in Indexator(loc_idxs), datasource.source.get_batch(Indexator(idxs)):
                mz, data_int = process_function(mz, np.asarray(data_int, dtype=dtypeconv), configs, **internal_configs)
                yield loc_idx, data_int
    
    @staticmethod
    def _peakpick_wrapper(idxs: Indexator | SliceIndexator | tuple| np.ndarray,
                          datasource: DataSource,
                          configs: dict | Configs | PipelineConfigurator,
                          dtypeconv: np.dtype | None = None,
                          **internal_configs
                          ):
        process_function = internal_configs.pop("process_pipeline")
        peakpick_function = internal_configs.pop("peakpick_function")
        proc_configs = configs['process']
        internal_proc_configs = internal_configs['process']
        peakpick_configs = configs['peakpick']
        internal_peakpick_configs = internal_configs['peakpick']

        if datasource.metadata['continuous']:
            mz = datasource.source.get_mz(idxs[0])
            data_int = np.asarray(np.vstack(tuple(datasource.source.get_intensities_stream(idxs))), dtype=dtypeconv)
            
            ## proccessing array
            data_int = process_function(mz, data_int, proc_configs, **internal_proc_configs)
            ## getting peaklist
            yield peakpick_function(mz, data_int, peakpick_configs, **internal_peakpick_configs)
        else:
            for mz, data_int in datasource.source.get_batch(Indexator(idxs)):
                mz, data_int = process_function(mz, data_int, proc_configs, **internal_proc_configs)
                
                yield peakpick_function(mz, np.asarray(data_int, dtype=dtypeconv), peakpick_configs, **internal_peakpick_configs)
        
    
class Drawer():
    def __init__(self, datasource: str | DataSource | PreparedDataSource):
        if isinstance(datasource, (DataSource, str)):
            if isinstance(datasource, str):
                datasource = DataSource(datasource)
            self.processed_spectra_path = datasource.processed_spectra_path
            self.peaklist_path = datasource.peaklist_path
            self.datasource = datasource
            if datasource.configs_path is not None:
                self.prepdata = PreparedDataSource(datasource, datasource.configs_path)
            else:
                self.prepdata = None
                if self.processed_spectra_path is None and self.peaklist_path is None:
                    raise ValueError("No processed spectra, peaklist or configs found. Only raw datasource")
        elif isinstance(datasource, PreparedDataSource):
            self.prepdata = datasource
            self.datasource = datasource._datasource
            self.processed_spectra_path = self.datasource.processed_spectra_path
            self.peaklist_path = self.datasource.peaklist_path
    def _draw(self,
              mz: np.ndarray,
              data_int: np.ndarray,
              peaklist: np.ndarray | None = None,
              headers: list[str] | None = None,
              roi: str | None = None,
              mz_range: tuple[float, float] | None = None,
              spectrum_idx: int | None = None):
        
        plt.figure().set_figwidth(25)
        plt.gcf().set_figheight(5)
        datasource = self.datasource
        diapcalc = lambda mz, plot_mz_range: (np.array(mz>plot_mz_range[0]) & np.array(mz<plot_mz_range[1])) if plot_mz_range is not None else range(len(mz))
        
        # Draw raw
        Label = ["Raw mass spectrum"]
        mz_raw, intens_raw = datasource.get_spectrum(spectrum_idx) 
        diap_raw = diapcalc(mz_raw, mz_range)
        plt.plot(mz_raw[diap_raw], intens_raw[diap_raw],alpha=0.75)
        
        # Draw processed
        Label.append("Processed mass spectrum")
        diap = diapcalc(mz, mz_range)
        plt.plot(mz[diap], data_int[diap],alpha=0.75)
        
        # Draw peaklist
        if peaklist is not None:
            if isinstance(peaklist, np.ndarray):
                peaklist = pd.DataFrame(peaklist.T, headers).T
            peaklist = peaklist.astype({"spectra_ind": int})
            peaklist.query("mz>@plot_mz_range[0] and mz<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx").plot(x="mz",y="Intensity",ax = plt.gca(),style = "x", color = "k")
            left_intens=[]
            for left_base in peaklist.query("PextL>@plot_mz_range[0] and PextL<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx")['PextL']:
                left_intens.append(data_int[mz>=left_base][0])
            right_intens = []
            for right_base in peaklist.query("PextR>@plot_mz_range[0] and PextR<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx")['PextR']:
                right_intens.append(data_int[mz<=right_base][-1])
            plt.plot(peaklist.query("PextL>@plot_mz_range[0] and PextL<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx")['PextL'],
            left_intens,'v')
            plt.plot(peaklist.query("PextR>@plot_mz_range[0] and PextR<@plot_mz_range[1] and spectra_ind == @sample_spectra_idx")['PextR'],
            right_intens,'^')
            Label=Label+[f'Peaks', 'Left peak base','Right peak base']
        plt.grid(visible=True,which="both")
        plt.xlim(mz_range)
        plt.legend([*Label])
        plt.minorticks_on()
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.title(f"Sample: {datasource.sample_name}, roi: {roi}, spectrum idx: {spectrum_idx}")
        plt.show()

    def audit_processing(self,
                         roi: str | list | None = None,
                         draw_mz_range: tuple[float, float] | None = None,
                         draw_spectrum_idx: int | None = None,
                         dtypeconv: np.dtype | None = None):
        if roi is None:
            roi = list(self.prepdata.roi_metadata.index)
        elif isinstance(roi, str):
            roi = [roi]
        elif isinstance(roi, list):
            pass
        else:
            raise ValueError("Invalid roi")

        pipeline = Pipeline(self.prepdata)
        datasource = self.datasource
        roi_metadata = datasource.roi_metadata
        for roi in roi_metadata.index:
            if draw_spectrum_idx is None:
                rmeta = roi_metadata.loc[roi]
                idxs = Indexator(rmeta["idxroi"])
                spectrum_idx = list(idxs)[np.random.randint(0,idxs.count)]
            else:
                spectrum_idx = draw_spectrum_idx

            processing_stream = pipeline._multistream_pipeline(Pipeline._procfunc_wrapper,roi, idxs = spectrum_idx, dtypeconv=dtypeconv)
            mz = next(processing_stream)
            loc_idx, data_int = next(processing_stream)
            roi_configs = self.prepdata._roi_configs[roi]
            peaklist_function = roi_configs._peaklist_function
            if peaklist_function:
                peaklist_configs = roi_configs.get_step_configs('peaklist')
                peaklist = peaklist_function(mz,
                                             data_int,
                                             [spectrum_idx],
                                             peaklist_configs,
                                             )
                headers = peaklist_configs['headers']
            else:
                peaklist = None
            if draw_mz_range is None:
                mz_range = (mz[0], mz[-1])
            else:
                mz_range = draw_mz_range
            self._draw(mz, data_int, peaklist, headers, mz_range, spectrum_idx)
    def draw_processed_data(self,
                            roi: str | list | None = None,
                            draw_mz_range: tuple[float, float] | None = None,
                            draw_spectrum_idx: int | None = None):
        if roi is None:
            roi = list(self.prepdata.roi_metadata.index)
        elif isinstance(roi, str):
            roi = [roi]
        elif isinstance(roi, list):
            pass
        else:
            raise ValueError("Invalid roi")
        datasource = self.datasource

        for r in roi:
            if draw_spectrum_idx is None:
                rmeta = datasource.roi_metadata.loc[roi]
                idxs = Indexator(rmeta["idxroi"])
                spectrum_idx = list(idxs)[np.random.randint(0,idxs.count)]
            else:
                spectrum_idx = draw_spectrum_idx
            
            if self.processed_spectra_path is None:
                if self.prepdata is None:
                    raise ValueError("No processed spectra path or configs to get processed spectrum")
                else:
                    pipeline = Pipeline(self.prepdata)
                    mz, data_int = pipeline._spectra_processing(r,idxs = spectrum_idx)
            else:
                with File(self.processed_spectra_path, "r") as hdf5:
                    mz, data_int = hdf5[r]["int"][datasource._get_local_roi_idx(spectrum_idx), :]

            if self.peaklist_path is not None:
                with File(self.peaklist_path, "r") as hdf5:
                    peaklist = pd.DataFrame(hdf5[r]["peaklist"][:].T, hdf5[r]['peaklist'].attrs["Column headers"]).query('spectra_ind == @spectrum_idx')
            else:
                peaklist = None
            if draw_mz_range is None:
                mz_range = (mz[0], mz[-1])
            else:
                mz_range = draw_mz_range
            self._draw(mz, data_int, peaklist, None, mz_range, spectrum_idx)
    