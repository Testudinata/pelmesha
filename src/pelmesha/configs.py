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
from pybaselines import Baseline
import inspect
import os
import warnings
from typing import Any, Callable
from pelmesha.pspectra import msalign,smoothing,DataProc_array,peaks_prop_array
import numpy as np
import yaml
from pydantic import BaseModel, Field, PrivateAttr, create_model

# --------------------------------------------------------------------------- #
#  Utility functions for adaptive parameter conversion                        #
# --------------------------------------------------------------------------- #

BASE_FUNCTIONS_FOR_PROCCESING = [msalign,smoothing,DataProc_array]
BASE_PEAKPICKING_FUNCTION = peaks_prop_array
def _shift_range_to_dots(window_shift_mz: tuple | list | None,
                         dots_distance: float) -> tuple | None:
    """Convert m/z shift range to dot-based shift range."""
    if window_shift_mz:
        return tuple(int(shift_mz / dots_distance) for shift_mz in window_shift_mz)
    return None


def _smooth_window_to_dots(smooth_window_mz: float | None,
                           dots_distance: float) -> int | None:
    """Convert m/z smoothing window to dot-based window."""
    if smooth_window_mz:
        return int(smooth_window_mz / dots_distance)
    return None


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
    - A ``(callable, class)`` tuple — used when an unbound method is
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

    # No annotation available — infer from default value
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

    Returns a dict mapping parameter name → description string.
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

    Returns a dict mapping parameter name → metadata dict with keys:
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
#  AdaptiveParameter — lazy m/z-to-dots conversion wrapper                    #
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
        Current working value — equals ``original`` before adaptation,
        then the adapted value after :meth:`__call__`.
    adaptation_rule : callable or None
        Transformation function ``f(parameter, *args) → adapted_value``.
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
#  DatasetHeaders — bidirectional column-name/index mapping                   #
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
# Dynamic pybaselines parameters (lam, p, diff_order, …) are NOT in this set
# and are handled via ``extra='allow'`` → ``__pydantic_extra__``.
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

# Reverse mapping: nested config group name → Pydantic model field name.
_NESTED_GROUP_TO_MODEL: dict[str, str] = {
    "baseline_configs": "baseline_params",
    "smoothing_configs": "smoothing_params",
    "msalign_configs": "alignment_params",
    "peaks_configs": "peakpicking_params",
}

class Configs(BaseModel):
    """
    Unified configuration for mass spectrometry data processing.

    All static parameters are declared as direct Pydantic fields, so your IDE
    will show autocompletion hints when typing ``Configs(…``.

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
    #  Direct Pydantic fields — visible in IDE autocomplete               #
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
        "extra": "allow",  # ← dynamic pybaselines params go to __pydantic_extra__
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
        # 0. Sync flat fields → sub-models
        self._sync_submodels()

        # 1. Validate baseline_algo against pybaselines
        self._validate_baseline_algo()

        # 2. Convert noise_func string to callable
        self._setup_noise_func()

        # 3. Compute dynamic headers
        self._compute_headers()

    def _sync_submodels(self) -> None:
        """Synchronise direct flat fields → Pydantic sub-models."""
        for flat_key, model_name in _PARAM_TO_MODEL.items():
            value = getattr(self, flat_key, None)
            model = getattr(self, model_name)
            if hasattr(model, flat_key):
                setattr(model, flat_key, value)

        # Sync dynamic extras → algo_params
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
        3. Nested config group name (e.g. ``"smoothing_configs"`` → returns a
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

        # 3. Nested config group name → return flat dict view
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
    #  Flatten — return all params as a single flat dict                  #
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

    # Mapping: section header → list of parameter names in that section.
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
        # Do NOT set state["_noise_func_callable"] directly — that would add a
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
    
class CustomConfigs():
    def __init__(self,
                configs_source: str | Callable | list | tuple | Pipeline,
                **kwargs):
        # 1. Load params from YAML or get default params from functions itself if configs_source is callable or list|tuple of callables
        self.configs: dict[str, Any] = {}
        if isinstance(configs_source, str):
            if not configs_source.endswith(".yaml"):
                configs_source += ".yaml"
                    
            if os.path.exists(configs_source):
                with open(configs_source, "rb") as f:
                    loaded = yaml.load(f, Loader=yaml.FullLoader)
                    if loaded:
                        self.configs.update(loaded)
        
        elif isinstance(configs_source, Pipeline):
            # Extract functions from Pipeline's process/peakpick methods via AST
            funcs: list[Callable] = []
            if configs_source.process_function is not None:
                funcs.extend(self._extract_called_functions(configs_source.process_function))
            if configs_source.peakpick_function is not None:
                funcs.extend(self._extract_called_functions(configs_source.peakpick_function))
            self.configs = _inspect_defaults(funcs)
            
        else:
            self.configs = _inspect_defaults(configs_source)

        # 2. Override with kwargs
        self.update(kwargs)

    # ------------------------------------------------------------------ #
    #  Name-based parameter access                                         #
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

        # --- (ClassName, methodName[, bucket]) tuple ---
        if isinstance(key, tuple):
            if len(key) == 2:
                cls_name, func_name = key
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

        # --- "ClassName.methodName" string ---
        if isinstance(key, str) and "." in key:
            parts = key.split(".", 1)
            cls_name, func_name = parts
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
                # class match — return cls_init_params
                cls_methods = methods[name]
                if len(cls_methods) == 1:
                    func_name = next(iter(cls_methods))
                    fb = cls_methods[func_name]
                    return dict(fb.get("cls_init_params", {}))
                else:
                    # Multiple methods in class — return all cls_init_params
                    return {
                        m: dict(d.get("cls_init_params", {}))
                        for m, d in cls_methods.items()
                    }

            # Ambiguous — warn and raise
            all_matches = method_matches + class_matches
            loc_list = "\n  - ".join(all_matches)
            raise KeyError(
                f"'{name}' is ambiguous — found in {total} locations:\n"
                f"  - {loc_list}\n"
                f"Use a qualified name to disambiguate:\n"
                f"    cfg[\"ClassName.methodName\"]\n"
                f"    cfg[(\"ClassName\", \"methodName\")]"
            )

        raise KeyError(f"Unsupported key type: {type(key).__name__}.")

    # ------------------------------------------------------------------ #
    #  YAML serialisation                                                  #
    # ------------------------------------------------------------------ #
    def save(self, path2dir: str = "",
             file_name: str = "Processing_settings") -> None:
        """Alias for :meth:`dump`."""
        self.dump(path2dir, file_name)

    def dump(self, path2dir: str = "",
             file_name: str = "Processing_settings") -> None:
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

        Python builtins (``enumerate``, ``len``, …) and names that cannot be
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

        source = CustomConfigs._get_source(func)
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

        # --- Build a variable → class/module map from simple assignments ---
        # e.g.  baseline = Baseline(mz_scale)  →  {"baseline": "Baseline"}
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

                # Case A: obj is a simple name — could be a variable, module, or class
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

                # Case B: obj is a chained call — Baseline().asls(...)
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

    def update(self,
               params_source: str | dict[str, Any] | None = None,
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

            * **Structured mode** — keys are callables (or their string
              names), values are dicts of parameters for that specific
              function::

                  cfg.update({msalign: {"smooth_window": 0.15}})

            * **Flat mode** — keys are plain parameter names, values
              are new values.  Every known method/function group that
              contains that parameter name is updated::

                  cfg.update({"smooth_window": 0.15})

        **kwargs
            Additional parameter overrides applied on top of
            *params_source*.
        """
        # --- Merge all sources into a single dict ---
        func_params: dict[str, Any] = {}

        if params_source is not None:
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
                        f"YAML file '{path}' not found — skipping."
                    )
            elif isinstance(params_source, dict):
                func_params.update(params_source)
            else:
                warnings.warn(
                    f"params_source must be a str (YAML path) or dict, "
                    f"got {type(params_source).__name__} — skipping."
                )

        if kwargs:
            if params_source is not None:
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
            # Resolve string to callable (SAFE — no eval)
            if isinstance(func_key, str):
                try:
                    func = eval(func_key)
                    if not callable(func):
                        raise ValueError
                except ValueError:
                    warnings.warn(
                        f"Skipping unknown function '{func_key}' — "
                        f"cannot resolve to a callable."
                    )
                    continue
            elif callable(func_key):
                func = func_key
            else:
                warnings.warn(
                    f"Skipping invalid key '{func_key!r}' — "
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

    def _update_flat(self, flat_overrides: dict[str, Any]) -> None:
        """Search all known method/function groups and update matching keys.

        When a parameter name is found in **multiple** functions, a warning
        is issued listing all matches.  Use **structured mode** to target a
        specific function unambiguously::

            cfg.update({msalign: {"smooth_window": 0.15}})
            cfg.update({"msalign": {"smooth_window": 0.15}})
        """
        methods = self.configs.get("methods", {})
        functions = self.configs.get("functions", {})

        for param_name, value in flat_overrides.items():
            # Collect all locations where this parameter exists
            locations: list[str] = []

            for cls_name, func_dict in methods.items():
                for func_name, func_bucket in func_dict.items():
                    param_bucket = func_bucket.get("params")
                    if param_bucket is not None and param_name in param_bucket:
                        locations.append(f"{cls_name}.{func_name} (params)")
                    init_bucket = func_bucket.get("cls_init_params")
                    if init_bucket is not None and param_name in init_bucket:
                        locations.append(f"{cls_name}.{func_name} (cls_init_params)")

            for func_name, func_bucket in functions.items():
                if param_name in func_bucket:
                    locations.append(f"{func_name} (function)")

            if not locations:
                warnings.warn(
                    f"Parameter '{param_name}' not found in any known "
                    f"function group — ignoring."
                )
                continue

            # Apply the value to all locations
            for cls_name, func_dict in methods.items():
                for func_name, func_bucket in func_dict.items():
                    param_bucket = func_bucket.get("params")
                    if param_bucket is not None and param_name in param_bucket:
                        param_bucket[param_name] = value
                    init_bucket = func_bucket.get("cls_init_params")
                    if init_bucket is not None and param_name in init_bucket:
                        init_bucket[param_name] = value

            for func_name, func_bucket in functions.items():
                if param_name in func_bucket:
                    func_bucket[param_name] = value

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

    @staticmethod
    def _merge_params(target: dict[str, Any],
                      source: dict[str, Any],
                      func_key: str | Callable) -> None:
        """Update *target* with *source*, warning about unknown keys."""
        for key, value in source.items():
            if key in target:
                target[key] = value
            else:
                warnings.warn(
                    f"Parameter '{key}' is not a known parameter of "
                    f"'{func_key}' — ignoring."
                )

class Pipeline:
    def __init__(self, 
                 process_function: Callable | None = None, 
                 peakpick_function: Callable | None = None,
                 configs: Configs| dict = {}):
        self.configs = configs
        self.process_function = process_function
        self.peakpick_function = peakpick_function
    def process(self, DataSource):
        pass

    def peakpick(self, DataSource):
        pass
    