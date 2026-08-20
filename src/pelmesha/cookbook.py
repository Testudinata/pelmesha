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
Special classes: PreparedDataSource and PipelineConfigurator
Supporting classes: AdaptiveParameter
"""
from __future__ import annotations
import ast
import inspect
import textwrap
import os
import warnings
from typing import Any, Callable
import numpy as np
import yaml
import importlib
from pybaselines import Baseline #Необходимо для set_method PipelineConfigurator
from pelmesha.filling import DataSource
from pelmesha.dough import SliceIndexator, Indexator
from pelmesha.kneading import preprocess_configuration_base, process_spectra_base, peakpicking_base
from pydantic import BaseModel, Field, field_validator
# Names of the built-in (base) pipeline step functions from pelmesha.kneading.
# Used to detect custom vs. base functions during YAML save/load.
_BASE_PIPELINE_FUNC_NAMES: frozenset[str] = frozenset({
    "preprocess_configuration_base",
    "process_spectra_base",
    "peakpicking_base",
})

import copy
import pandas as pd
# --------------------------------------------------------------------------- #
#  Utility functions for adaptive parameter conversion                        #
# --------------------------------------------------------------------------- #

def _inspect_defaults(
    functions: Callable | list | tuple,
) -> dict[str, inspect.Parameter]:
    """Return a dictionary of parameter names to inspect.Parameter objects.

    Each element in *functions* can be:
    - A plain callable (function or bound method)
    - A ``(callable, class)`` tuple - used when an unbound method is
      resolved together with its owning class (e.g. from AST extraction).
    - A ``(callable, class, config_key, explicit_params)`` tuple - returned
      by :meth:`PipelineConfigurator._extract_called_functions` when the
      callable was found via a ``**configs['key']`` unfold pattern.  In this
      case *config_key* overrides the storage key and *explicit_params* is a
      set of parameter names that are already provided in the call site and
      should be skipped.
    - A ``(callable, config_key, explicit_params)`` tuple - same as above
      but for plain functions (no class).
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
        # New format from _extract_called_functions:
        #   (callable, config_key, explicit_params)  for plain functions
        #   ((method, cls), config_key, explicit_params)  for methods
        if (isinstance(item, tuple) and len(item) == 3
                and isinstance(item[2], (set, frozenset))):
            # New format with config_key and explicit_params
            callable_or_pair = item[0]
            config_key: str = item[1]
            explicit_params: set[str] = item[2]
            if isinstance(callable_or_pair, tuple) and len(callable_or_pair) == 2:
                func, cls = callable_or_pair
            else:
                func = callable_or_pair
                cls = None
        elif isinstance(item, tuple) and len(item) == 2:
            func, cls = item
            config_key = None
            explicit_params = set()
        else:
            func = item
            cls = None
            config_key = None
            explicit_params = set()

        func_name = func.__name__
        # When a config_key is provided, use it as the storage key
        # instead of the function name (e.g. 'smoothing' config key
        # for the 'smoothing' function).
        storage_key = config_key if config_key is not None else func_name
        
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
            methods_defaults[cls_name][storage_key] = {}
            method = methods_defaults[cls_name][storage_key]
            method["cls_init_params"] = {}

            for param in inspect.signature(cls).parameters.values(): # Получаем параметры инициализации класса
                if (param.default is not inspect.Parameter.empty
                        and param.name not in explicit_params):
                    method["cls_init_params"][param.name] = param.default
            method['params'] = {}
            for param in func_sig.parameters.values():
                if (param.default is not inspect.Parameter.empty
                        and param.name not in explicit_params):
                    method['params'][param.name] = param.default

        elif cls is not None and isinstance(cls, type): # Unbound method with class info
            cls_name = cls.__name__
            if cls_name not in methods_defaults:
                methods_defaults[cls_name] = {}
            methods_defaults[cls_name][storage_key] = {}
            method = methods_defaults[cls_name][storage_key]
            method["cls_init_params"] = {}
            for param in inspect.signature(cls).parameters.values():
                if (param.default is not inspect.Parameter.empty
                        and param.name not in explicit_params):
                    method["cls_init_params"][param.name] = param.default
            method['params'] = {}
            for param in func_sig.parameters.values():
                if (param.default is not inspect.Parameter.empty
                        and param.name not in explicit_params):
                    method['params'][param.name] = param.default

        else: # Plain functions
            functions_defaults[storage_key] = {}
            for param in func_sig.parameters.values():
                if (param.default is not inspect.Parameter.empty
                        and param.name not in explicit_params):
                    functions_defaults[storage_key][param.name] = param.default
    return defaults

_KNOWN_BUCKETS: frozenset[str] = frozenset({"cls_init_params", "params"})
"""Known bucket names inside a method entry."""


def _format_param_value(value: Any) -> str:
    """Format a parameter value for human-readable display.

    * Callables → qualified name (e.g. ``np.std``, ``scipy.signal.savgol_filter``)
    * Everything else → ``repr()``
    """
    if callable(value) and not inspect.isclass(value):
        # Try to get a qualified name like "np.std"
        name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
        if name:
            # Check if the callable lives in a module
            mod = getattr(value, "__module__", None)
            if mod and mod != "builtins":
                # Shorten common module paths
                if mod.startswith("numpy"):
                    mod = "np"
                elif mod.startswith("scipy"):
                    mod = "scipy"
                return f"{mod}.{name}"
            return name
        return repr(value)
    return repr(value)


# --------------------------------------------------------------------------- #
#  YAML representer / constructor for callable values                         #
# --------------------------------------------------------------------------- #

def _callable_to_yaml(dumper: yaml.Dumper, data: Callable) -> yaml.Node:
    """Serialize a callable as ``!obj <qualified_name>``.

    Examples in YAML::

        noise_func: !obj np.std
        baseliner: !obj pybaselines.Baseline
    """
    mod = getattr(data, "__module__", None)
    name = getattr(data, "__qualname__", None) or getattr(data, "__name__", None)
    if mod and name and mod != "builtins":
        # Shorten common aliases
        if mod.startswith("numpy"):
            mod = "np"
        return dumper.represent_scalar("!obj", f"{mod}.{name}")
    # Fallback: use repr (e.g. for lambdas or runtime-defined functions)
    return dumper.represent_scalar("!obj", repr(data))


def _yaml_to_callable(loader: yaml.Loader, node: yaml.Node) -> Callable:
    """Restore a callable from a ``!obj`` tag.

    Resolution order:
    1. ``globals()`` of this module (catches user-defined functions
       that were imported or defined at runtime).
    2. ``importlib.import_module`` → ``getattr`` (catches library
       functions like ``np.std``, ``scipy.signal.savgol_filter``).
    """
    value: str = loader.construct_scalar(node)

    # 1. Try globals() first — catches runtime-defined / imported names
    obj = globals().get(value)
    if obj is not None and callable(obj):
        return obj

    # 2. Try importlib — split "np.std" → module="numpy", attr="std"
    parts = value.split(".")
    if len(parts) >= 2:
        # Reverse common aliases
        module_name = parts[0]
        if module_name == "np":
            module_name = "numpy"
        try:
            module = importlib.import_module(module_name)
            for attr in parts[1:]:
                module = getattr(module, attr)
            if callable(module):
                return module
        except (ImportError, AttributeError):
            pass

    # 3. Try the full value as a module name (e.g. "pelmesha.kneading")
    try:
        module = importlib.import_module(value)
        if callable(module):
            return module
    except ImportError:
        pass

    raise yaml.YAMLError(
        f"Cannot resolve callable '{value}'. "
        f"Ensure the module is importable and the name is correct."
    )


# Register YAML representers for callable types.
# We register for specific types first (fast path), then add a fallback
# multi-representer for ``object`` that catches any remaining callable
# types (e.g. ``numpy._ArrayFunctionDispatcher`` for ``np.std``).
#
# We do NOT use ``yaml.add_multi_representer(Callable, ...)`` because
# the ``Callable`` ABC can match ``str`` and other non-callable types.
import types as _types
yaml.add_representer(_types.FunctionType, _callable_to_yaml)
yaml.add_representer(_types.BuiltinFunctionType, _callable_to_yaml)
yaml.add_representer(_types.MethodType, _callable_to_yaml)
yaml.add_representer(_types.LambdaType, _callable_to_yaml)
# Also handle classes (they are callable too, e.g. Baseline)
yaml.add_representer(type, _callable_to_yaml)


def _callable_fallback(dumper: yaml.Dumper, data: object) -> yaml.Node:
    """Fallback multi-representer: if *data* is callable, use our
    ``!obj`` tag; otherwise delegate to the default YAML behaviour."""
    if callable(data) and not isinstance(data, (str, bytes, int, float,
                                                  bool, type(None), list,
                                                  tuple, dict)):
        return _callable_to_yaml(dumper, data)
    # Fall through to YAML's default object serialisation
    return dumper.represent_object(data)


yaml.add_multi_representer(object, _callable_fallback)

yaml.add_constructor("!obj", _yaml_to_callable, Loader=yaml.FullLoader)
yaml.add_constructor("!obj", _yaml_to_callable, Loader=yaml.UnsafeLoader)
yaml.add_constructor("!obj", _yaml_to_callable, Loader=yaml.Loader)


class Configs():
    def __init__(self,
                 configs_source: str | dict | None = None,
                 **kwargs):
        self.configs: dict[str, Any] = {}
        # Maps (cls_name, storage_key) -> set[str] of parameter names that
        # are explicitly provided at the call site and should be excluded
        # from default extraction. Populated by PipelineConfigurator during
        # AST-based extraction; used by set_method to preserve exclusions.
        self._explicit_params_map: dict[tuple[str, str], set[str]] = {}
        if configs_source:
            if isinstance(configs_source, str):
                self.load_config(configs_source)
            elif isinstance(configs_source, dict):
                self.replace_config(configs_source)
            else:
                warnings.warn(
                    f"configs_source must be a str (YAML path) or dict, "
                    f"got {type(configs_source).__name__} - ignoring."
                )
        if kwargs:
            self.update(kwargs)

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
    #  Construct methods: full config replacement                         #
    # ------------------------------------------------------------------ #

    def replace_config(self, configs_dict: dict[str, Any]) -> None:
        """Replace the entire configuration with *configs_dict*.

        Validates the overall structure of *configs_dict* via
        :meth:`_validate_configs_structure` before replacing ``self.configs``.
        """
        self._validate_configs_structure(configs_dict)
        self.configs = copy.deepcopy(configs_dict)

    def load_config(self, yaml_path: str) -> None:
        """Load a full configuration from a YAML file and replace the current one.

        Loads the YAML file into a dict and delegates to
        :meth:`replace_config` after validating the structure.
        """
        if not yaml_path.endswith(".yaml"):
            yaml_path += ".yaml"
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        with open(yaml_path, "rb") as f:
            loaded = yaml.load(f, Loader=yaml.FullLoader)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"YAML file must contain a top-level mapping (dict), "
                f"got {type(loaded).__name__}."
            )
        self.replace_config(loaded)

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
    def __setitem__(self, key: str | tuple[str, ...], value: Any):
        if isinstance(key, tuple):
            self._set_by_path(list(key), value)
        elif isinstance(key, str) and "." in key:
            self._set_by_path(key.split("."), value)
        else:
            self._update_flat({key: value})
    def get(self, key: str | tuple[str, ...], default: Any = None) -> dict[str, Any]:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
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
        lines = ["Configs("]

        # --- Plain functions ---
        if functions:
            lines.append("  functions:")
            for func_name, params in functions.items():
                if params:
                    items = ", ".join(f"{k}={_format_param_value(v)}" for k, v in params.items())
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
                        items = ", ".join(f"{k}={_format_param_value(v)}" for k, v in init_params.items())
                        lines.append(f"      __init__({items})")
                    # Method params
                    method_params = bucket.get("params", {})
                    if method_params:
                        items = ", ".join(f"{k}={_format_param_value(v)}" for k, v in method_params.items())
                        lines.append(f"      {method_name}({items})")
                    else:
                        lines.append(f"      {method_name}: (no parameters)")

        if not functions and not methods:
            lines.append("  (empty)")

        lines.append(")")
        return "\n".join(lines)
    
    def update(self,
               params_source: str | dict[str, Any] | None = None,
               **kwargs) -> None:
        """Update configuration parameters (change-only).

        Only changes **existing** parameter values; never creates new
        entries.  If a method, function, or parameter does not exist, a
        warning is issued and it is skipped.

        Accepts parameters from a YAML file, a dictionary, or keyword
        arguments.  When both *params_source* and **kwargs* are given,
        a warning is issued and **kwargs* take priority.

        The dictionary supports several modes:

        * **Full config dict** - top-level ``methods`` / ``functions``
          keys, deep-merged into the existing config (only existing keys
          are touched)::

              cfg.update({"methods": {"Baseline": {"asls": {"params": {"lam": 1e5}}}}})

        * **Partial nested path** - keys are class or function names,
          values are nested dicts navigating the config hierarchy::

              cfg.update({"Baseline": {"asls": {"params": {"lam": 1e5}}}})
              cfg.update({"msalign": {"smooth_window": 0.15}})

        * **Structured mode** - keys are callables (or their string
          names), values are dicts of parameters for that specific
          function::

              cfg.update({msalign: {"smooth_window": 0.15}})

        * **Flat mode** - keys are plain parameter names, values are new
          values.  Every known method/function group that contains that
          parameter name is updated::

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

        # --- Full config dict {"methods": ..., "functions": ...} ---
        if "methods" in func_params or "functions" in func_params:
            self._validate_configs_structure(func_params)
            self._update_nested(self.configs, func_params, "configs")
            return

        # --- Partial nested path (class/function names as keys) ---
        if self._is_partial_path_dict(func_params):
            self._update_partial_path(func_params)
            return

        # --- Structured mode (callable keys) ---
        if self._has_callable_keys(func_params):
            self._update_structured(func_params)
            return

        # --- Flat mode ---
        self._update_flat(func_params)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _update_nested(self,
                       target: dict[str, Any],
                       source: dict[str, Any],
                       label: str) -> None:
        """Recursively update *target* with *source*, touching only existing keys.

        When a key is missing from *target*, a warning is issued and the
        key is skipped.  Nested dicts are merged recursively.
        """
        if not isinstance(source, dict):
            return
        for key, value in source.items():
            if key not in target:
                warnings.warn(
                    f"'{key}' not found in '{label}' - ignoring."
                )
                continue
            if isinstance(value, dict) and isinstance(target[key], dict):
                self._update_nested(target[key], value, f"{label}.{key}")
            else:
                target[key] = value

    def _is_partial_path_dict(self, d: dict[str, Any]) -> bool:
        """Return ``True`` if *d* looks like a partial nested-path dict.

        A partial path dict has class or function names as keys with dict
        values (e.g. ``{"Baseline": {"asls": {...}}}``).
        """
        methods = self.configs.get("methods", {})
        functions = self.configs.get("functions", {})
        for key, value in d.items():
            if not isinstance(key, str):
                return False
            if (key in methods or key in functions) and isinstance(value, dict):
                return True
        return False

    def _update_partial_path(self, partial: dict[str, Any]) -> None:
        """Update configs following a partial nested path.

        Keys are class or function names; values are nested dicts that
        navigate the config hierarchy.  Only existing keys are changed.
        """
        methods = self.configs.get("methods", {})
        functions = self.configs.get("functions", {})
        for key, value in partial.items():
            if key in methods:
                self._update_nested(methods[key], value, f"methods.{key}")
            elif key in functions:
                self._update_nested(functions[key], value, f"functions.{key}")
            else:
                warnings.warn(
                    f"Unknown class or function '{key}' - ignoring."
                )

    def _has_callable_keys(self, d: dict[str, Any]) -> bool:
        """Return ``True`` if any key of *d* is a callable or resolves to one."""
        for key in d:
            if callable(key):
                return True
            if isinstance(key, str):
                try:
                    candidate = eval(key)
                    if callable(candidate):
                        return True
                except (ValueError, NameError):
                    continue
        return False

    def _update_structured(self,
                           func_params: dict[str | Callable, dict[str, Any]]
                           ) -> None:
        """Apply overrides where keys are callables (or their string names).

        Change-only: existing entries are updated, missing ones are
        skipped with a warning.
        """
        methods = self.configs.get("methods", {})
        functions = self.configs.get("functions", {})

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

                cls_bucket = methods.get(cls_name)
                if cls_bucket is None:
                    warnings.warn(
                        f"Class '{cls_name}' not found in config - ignoring."
                    )
                    continue
                func_bucket = cls_bucket.get(func_name)
                if func_bucket is None:
                    warnings.warn(
                        f"Method '{cls_name}.{func_name}' not found in "
                        f"config - ignoring."
                    )
                    continue
                self._update_method(func_bucket, params, f"{cls_name}.{func_name}")

            # Plain function
            else:
                func_name = func.__name__
                func_bucket = functions.get(func_name)
                if func_bucket is None:
                    warnings.warn(
                        f"Function '{func_name}' not found in config - ignoring."
                    )
                    continue
                self._update_function(func_bucket, params, func_name)

    def _update_method(self,
                       func_bucket: dict[str, Any],
                       params: dict[str, Any],
                       label: str) -> None:
        """Update a method bucket's ``params`` / ``cls_init_params``.

        If *params* contains neither ``params`` nor ``cls_init_params``
        keys, all keys are treated as method parameters.
        """
        if not isinstance(params, dict):
            return
        cls_init = params.get("cls_init_params")
        method_params = params.get("params")
        if cls_init is None and method_params is None:
            method_params = params

        if cls_init is not None:
            if not isinstance(cls_init, dict):
                warnings.warn(
                    f"'cls_init_params' for '{label}' must be a dict - ignoring."
                )
            else:
                init_bucket = func_bucket.get("cls_init_params")
                if init_bucket is None:
                    warnings.warn(
                        f"'cls_init_params' not found in '{label}' - ignoring."
                    )
                else:
                    self._update_nested(init_bucket, cls_init,
                                        f"{label}.cls_init_params")

        if method_params is not None:
            if not isinstance(method_params, dict):
                warnings.warn(
                    f"'params' for '{label}' must be a dict - ignoring."
                )
            else:
                param_bucket = func_bucket.get("params")
                if param_bucket is None:
                    warnings.warn(
                        f"'params' not found in '{label}' - ignoring."
                    )
                else:
                    self._update_nested(param_bucket, method_params,
                                        f"{label}.params")

    def _update_function(self,
                         func_bucket: dict[str, Any],
                         params: dict[str, Any],
                         label: str) -> None:
        """Update a plain function's parameters (change-only)."""
        if not isinstance(params, dict):
            return
        self._update_nested(func_bucket, params, label)

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

    def _set_by_path(self, parts: list[str], value: Any) -> None:
        """Set a value at a nested path, creating intermediate dicts as needed.

        Used by :meth:`__setitem__` for tuple / dotted keys.  The path is
        navigated relative to ``self.configs``; missing intermediate keys
        are created as dicts.
        """
        if not parts:
            return
        current = self.configs
        for part in parts[:-1]:
            nxt = current.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                current[part] = nxt
            current = nxt
        current[parts[-1]] = value

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

        * **Delete entire class / method / function** — bare name or
          ``ClassName.method_name``::

              cfg.delete("Baseline")          # remove entire Baseline class
              cfg.delete("Baseline.snip")     # remove only the snip method
              cfg.delete("msalign")           # remove entire msalign function

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
            # Check if it is a bare class name → delete entire class
            methods = self.configs.get("methods", {})
            if param_name in methods:
                del methods[param_name]
                print(f"Deleted entire class '{param_name}' from methods.")
                return

            # Check if it is a bare function name → delete entire function
            functions = self.configs.get("functions", {})
            if param_name in functions:
                del functions[param_name]
                print(f"Deleted entire function '{param_name}' from functions.")
                return

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

        Supports deleting entire classes, methods, or functions::

            cfg.delete("Baseline")               # delete entire Baseline class
            cfg.delete("Baseline.snip")           # delete only the snip method
            cfg.delete("msalign")                 # delete entire msalign function
            cfg.delete("msalign.smooth_window")   # function.param
            cfg.delete("Baseline.snip.lam")       # class.method.param
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
                # Delete entire class
                del methods[first]
                print(f"Deleted entire class '{first}' from methods.")
                return

            second, *tail = rest

            if not tail:
                # Only two parts: "ClassName.method_name" → delete entire method
                if second not in methods[first]:
                    warnings.warn(
                        f"Method '{second}' not found in class "
                        f"'{first}' - ignoring."
                    )
                    return
                del methods[first][second]
                print(f"Deleted method '{first}.{second}'.")
                # Clean up empty class
                if not methods[first]:
                    del methods[first]
                    print(f"Class '{first}' is now empty — removed as well.")
                return

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

        # --- Function.param or entire function ---
        if first in functions:
            if not rest:
                # Delete entire function
                del functions[first]
                print(f"Deleted entire function '{first}' from functions.")
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
        # Look up stored explicit params to preserve exclusions from
        # AST-based extraction (e.g. params provided at the call site).
        # Collect explicit params from ALL methods of this class because
        # cls_init_params (__init__ params) are shared across all methods.
        stored_explicit: set[str] = set()
        for (_cls_name, _method_name), exp in self._explicit_params_map.items():
            if _cls_name == cls_name:
                stored_explicit.update(exp)
        if stored_explicit:
            defaults = _inspect_defaults([((method_func, cls_obj), method_name, stored_explicit)])
        else:
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

    def set_function(self,
                     func_name: str,
                     delete_old: bool = True,
                     **params) -> None:
        """Set or replace a plain function's configuration in the config.

        Inspects the actual callable (if resolvable) to extract default
        parameters, then applies any overrides from *params*.

        Parameters
        ----------
        func_name : str
            Name of the function to configure.  If the name resolves to a
            callable in the current scope (via ``eval``), its signature is
            inspected for default extraction and parameter validation.
        delete_old : bool
            If ``True`` (default), removes all existing entries for this
            function name before adding the new one.
        **params
            Parameter overrides.  All keys are treated as function
            parameters.
        """
        # --- Optionally delete old entry ---
        functions = self.configs.setdefault("functions", {})
        if delete_old and func_name in functions:
            del functions[func_name]

        # --- Try to resolve the callable for default extraction ---
        func_obj = None
        try:
            candidate = eval(func_name)
            if callable(candidate):
                func_obj = candidate
        except (ValueError, NameError):
            pass

        if func_obj is not None:
            # Extract defaults via _inspect_defaults
            defaults = _inspect_defaults([func_obj])
            func_defaults = defaults.get("functions", {}).get(func_name, {})
            func_bucket = functions.setdefault(func_name, {})
            func_bucket.update(dict(func_defaults))
        else:
            func_bucket = functions.setdefault(func_name, {})

        # --- Apply overrides from **params ---
        if params:
            if func_obj is not None:
                self._merge_params(func_bucket, params, func_obj)
            else:
                func_bucket.update(params)

class PipelineConfigurator(Configs):
    # TODO Написать про кастомные функции «Загружайте конфигурационные файлы и кастомные функции только из доверенных источников, в коде есть слабости».
    def __init__(self,
                configs_source: str | dict | None = None,
                preprocess_function: Callable = preprocess_configuration_base,
                process_pipeline: Callable = process_spectra_base,
                peakpick_function: Callable = peakpicking_base,
                **kwargs):
        # Store pipeline function references
        self._preprocess_function: Callable = preprocess_function
        self._process_function: Callable = process_pipeline
        self._peakpick_function: Callable = peakpick_function

        # 1. Get default params from functions itself
        self.configs: dict[str, Any] = {}
        # Maps (cls_name, storage_key) -> set[str] of parameter names that
        # are explicitly provided at the call site and should be excluded
        # from default extraction. Populated during AST-based extraction;
        # used by set_method to preserve exclusions.
        self._explicit_params_map: dict[tuple[str, str], set[str]] = {}

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
            # All items are now 3-tuples: (callable_or_pair, config_key, explicit_params)
            for item in extracted:
                callable_or_pair = item[0]
                if isinstance(callable_or_pair, tuple) and len(callable_or_pair) == 2:
                    func, cls = callable_or_pair
                else:
                    func = callable_or_pair
                self._step_func_names[step_name].add(func.__name__)

        # Extract defaults from pipeline functions
        self.configs = _inspect_defaults(pipeline_funcs)

        # Populate _explicit_params_map from extracted items so that
        # set_method can preserve the exclusion of explicitly-provided params.
        for item in pipeline_funcs:
            if isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], (set, frozenset)):
                callable_or_pair = item[0]
                config_key: str = item[1]
                explicit_params: set[str] = item[2]
                if not explicit_params:
                    continue
                if isinstance(callable_or_pair, tuple) and len(callable_or_pair) == 2:
                    _func, cls = callable_or_pair
                    if isinstance(cls, type):
                        key = (cls.__name__, config_key)
                        self._explicit_params_map[key] = explicit_params
                else:
                    # Plain functions — store by (None, config_key)
                    key = (None, config_key)
                    self._explicit_params_map[key] = explicit_params

        # 2. Load params from YAML or dict and override with kwargs
        if configs_source:
            if isinstance(configs_source, str):
                self.load_config(configs_source)
            elif isinstance(configs_source, dict):
                self.replace_config(configs_source)
            else:
                warnings.warn(
                    f"configs_source must be a str (YAML path) or dict, "
                    f"got {type(configs_source).__name__} - ignoring."
                )
        if kwargs:
            self.update(kwargs)

    # ------------------------------------------------------------------ #
    #  Override set_method / set_function to keep _step_func_names in    #
    #  sync so that get_step_configs can find replaced methods/functions. #
    # ------------------------------------------------------------------ #

    def set_method(self,
                   cls: type | str,
                   method_name: str,
                   delete_old: bool = True,
                   **params) -> None:
        """Set/replace a method and keep ``_step_func_names`` in sync."""
        # --- Resolve class name (same logic as parent) ---
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
        cls_name = cls_obj.__name__

        # Collect old method names for this class BEFORE parent modifies configs
        old_method_names: set[str] = set()
        methods = self.configs.setdefault("methods", {})
        if cls_name in methods:
            old_method_names = set(methods[cls_name].keys())

        # Determine which pipeline steps contain the old method names
        affected_steps: set[str] = set()
        for step_name, func_names in self._step_func_names.items():
            if old_method_names & func_names:
                affected_steps.add(step_name)

        # Call parent's set_method
        super().set_method(cls, method_name, delete_old=delete_old, **params)

        # --- Update _step_func_names ---
        if delete_old:
            # Remove old method names from affected steps
            for step_name in affected_steps:
                self._step_func_names[step_name].difference_update(old_method_names)

        # Add the new method_name to the same steps
        if affected_steps:
            for step_name in affected_steps:
                self._step_func_names[step_name].add(method_name)
        else:
            # Completely new class — add to all steps as a best guess
            for func_names in self._step_func_names.values():
                func_names.add(method_name)

    def set_function(self,
                     func_name: str,
                     delete_old: bool = True,
                     **params) -> None:
        """Set/replace a function and keep ``_step_func_names`` in sync."""
        # Determine which pipeline steps contain the old function name
        affected_steps: set[str] = set()
        for step_name, func_names in self._step_func_names.items():
            if func_name in func_names:
                affected_steps.add(step_name)

        # Call parent's set_function
        super().set_function(func_name, delete_old=delete_old, **params)

        # --- Update _step_func_names ---
        if delete_old:
            # Remove old func_name from affected steps
            for step_name in affected_steps:
                self._step_func_names[step_name].discard(func_name)

        # Add the (possibly new) func_name to the same steps
        if affected_steps:
            for step_name in affected_steps:
                self._step_func_names[step_name].add(func_name)
        else:
            # Completely new function — add to all steps as a best guess
            for func_names in self._step_func_names.values():
                func_names.add(func_name)

    def delete(self,
               param_name: str | tuple[str, ...]) -> None:
        """Delete a parameter, method, class, or function and keep
        ``_step_func_names`` in sync.

        Same indexing as :meth:`Configs.delete`, plus support for
        deleting entire classes and functions::

            pc.delete("Baseline")          # remove entire Baseline class
            pc.delete("Baseline.snip")     # remove only the snip method
            pc.delete("msalign")           # remove entire msalign function

        Parameters
        ----------
        param_name : str or tuple of str
            The parameter to delete.  See :meth:`Configs.delete`.
        """
        # --- Resolve what is being deleted BEFORE parent modifies configs ---
        methods = self.configs.get("methods", {})
        functions = self.configs.get("functions", {})

        # Collect names to remove from _step_func_names
        removed_method_names: set[str] = set()
        removed_func_names: set[str] = set()

        if isinstance(param_name, str) and "." not in param_name:
            # Bare name — could be a class or function
            if param_name in methods:
                removed_method_names = set(methods[param_name].keys())
            elif param_name in functions:
                removed_func_names.add(param_name)
        elif isinstance(param_name, (tuple, str)):
            # Path — extract the first component
            first = param_name[0] if isinstance(param_name, tuple) else param_name.split(".")[0]
            if first in methods:
                if isinstance(param_name, tuple) and len(param_name) == 2:
                    # "ClassName.method_name" — delete single method
                    removed_method_names.add(param_name[1])
                elif isinstance(param_name, str) and param_name.count(".") == 1:
                    # "ClassName.method_name" — delete single method
                    removed_method_names.add(param_name.split(".")[1])
                else:
                    # Deeper path or bare class — collect all method names
                    removed_method_names = set(methods[first].keys())
            elif first in functions:
                if isinstance(param_name, tuple) and len(param_name) == 1:
                    removed_func_names.add(first)
                elif isinstance(param_name, str) and "." not in param_name:
                    removed_func_names.add(first)

        # Call parent's delete
        super().delete(param_name)

        # --- Update _step_func_names ---
        for step_name, func_names in self._step_func_names.items():
            func_names.difference_update(removed_method_names)
            func_names.difference_update(removed_func_names)

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
        return Configs(result)


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

        The output includes comments that group parameters by pipeline step
        (preprocess / process / peakpick) and show which step function they
        are passed to.  Built-in (kneading) functions are marked ``(BASE)``.

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

        yaml_str = self._yaml_with_comments()
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        # Save companion .py file with custom pipeline functions
        py_path = path[:-5] + ".py" if path.endswith(".yaml") else path + ".py"
        self._dump_functions(py_path)

    # ------------------------------------------------------------------ #
    #  YAML string builder with step comments                             #
    # ------------------------------------------------------------------ #

    def _yaml_with_comments(self) -> str:
        """Build a YAML string with minimal step/function comments.

        Comments are inserted by scanning the raw YAML output and detecting
        when the pipeline step changes between consecutive function entries.
        The same step header may appear multiple times if its functions are
        interleaved by YAML's alphabetical sorting::

            # PREPROCESS step  →  preprocess_configuration_base  (BASE)

            functions:
              resample_mz_scale:
                ...

            # PROCESS step  →  process_spectra_base  (BASE)

              msalign:
                ...
              smoothing:
                ...

            # PEAKPICK step  →  peakpicking_base  (BASE)

              peakpicker:
                ...

            methods:
              ...
        """
        import io

        # --- 1. Build reverse mapping: short-name → step ---
        func_to_step: dict[str, str] = {}
        for step_name, func_names in self._step_func_names.items():
            for name in func_names:
                func_to_step[name] = step_name

        # --- 2. Step header templates ---
        step_headers: dict[str, str] = {}
        step_order: list[str] = ["preprocess", "process", "peakpick"]
        for step_name in step_order:
            step_func = getattr(self, f"_{step_name}_function", None)
            func_label = step_func.__name__ if step_func else step_name
            is_base = (step_func is not None
                       and step_func.__name__ in _BASE_PIPELINE_FUNC_NAMES)
            base_marker = "BASE " if is_base else ""
            step_headers[step_name] = (
                f"# {base_marker}{step_name.upper()} step. Function:  {func_label}"
            )

        # --- 3. Dump raw YAML ---
        buf = io.StringIO()
        yaml.dump(self.configs, buf,
                  default_flow_style=False, sort_keys=True)
        raw_lines = buf.getvalue().split("\n")

        # --- 4. Scan lines and insert headers when step changes ---
        lines: list[str] = []
        in_functions = False
        current_func_step: str | None = None

        for line in raw_lines:
            stripped = line.rstrip()

            # Track which section we're in
            if stripped == "functions:":
                in_functions = True
                lines.append(line)
                continue
            elif stripped == "methods:":
                in_functions = False
                lines.append(line)
                continue

            # In the functions section, detect function-name lines (indent 2)
            if in_functions and line.startswith("  ") and not line.startswith("    "):
                colon_pos = stripped.find(":")
                if colon_pos != -1:
                    candidate = stripped[2:colon_pos].strip()
                    if candidate and candidate in func_to_step:
                        step = func_to_step[candidate]
                        if step != current_func_step:
                            header = step_headers.get(step, f"# {step}")
                            lines.append("")
                            lines.append(header)
                            current_func_step = step

            lines.append(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Custom functions serialisation (companion .py file)                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_custom_functions(config: PipelineConfigurator) -> dict[str, Callable]:
        """Extract custom (non-builtin) pipeline functions from *config*.

        Returns a dict mapping attribute names (``'_preprocess_function'``,
        ``'_process_function'``, ``'_peakpick_function'``) to the callable,
        but only for functions whose ``__name__`` is **not** in
        :data:`_BASE_PIPELINE_FUNC_NAMES`.

        Parameters
        ----------
        config : PipelineConfigurator
            The config instance to inspect.

        Returns
        -------
        dict[str, Callable]
        """
        func_map: dict[str, Callable] = {
            '_preprocess_function': config._preprocess_function,
            '_process_function': config._process_function,
            '_peakpick_function': config._peakpick_function,
        }
        return {
            attr_name: func
            for attr_name, func in func_map.items()
            if func.__name__ not in _BASE_PIPELINE_FUNC_NAMES
        }

    def _get_functions_content(self) -> str | None:
        """Build the companion ``.py`` content for custom pipeline functions.

        Returns the full source text (including the ``__pipeline_functions__``
        mapping dict), or ``None`` when all three step functions are built-in
        (kneading) — meaning no companion file is needed.

        Returns
        -------
        str or None
        """
        custom_funcs = self._get_custom_functions(self)

        if not custom_funcs:
            return None

        lines: list[str] = []
        lines.append("# Custom pipeline functions for PipelineConfigurator\n")
        lines.append("# Auto-generated - do not edit manually\n\n")
        lines.append("import numpy as np\n\n")

        for attr_name, func in custom_funcs.items():
            source = self._get_source(func)
            if source:
                lines.append(f"# --- {attr_name} ---\n")
                lines.append(textwrap.dedent(source))
                lines.append("\n\n")
            else:
                lines.append(f"# {attr_name}: {func.__name__!r} (source not available)\n\n")

        # Append the mapping dict at the end
        lines.append("# Auto-generated function mapping\n")
        lines.append("__pipeline_functions__ = {\n")
        for attr_name, func in custom_funcs.items():
            source = self._get_source(func)
            if source:
                lines.append(f"    '{attr_name}': {func.__name__},\n")
        lines.append("}\n")

        return "".join(lines)

    def _dump_functions(self, py_path: str) -> None:
        """Save source code of **custom** pipeline functions to a companion .py file.

        Built-in (kneading) functions are **skipped** — they are imported
        directly from ``pelmesha.kneading`` on load.

        Delegates content generation to :meth:`_get_functions_content`.

        Parameters
        ----------
        py_path : str
            Full path to the ``.py`` file to write.
        """
        content = self._get_functions_content()
        if content is None:
            if os.path.exists(py_path):
                os.remove(py_path)
            return

        with open(py_path, "w", encoding="utf-8") as f:
            f.write(content)

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

    @staticmethod
    def _restore_functions_from_dict(
        config: PipelineConfigurator,
        funcs: dict[str, Callable],
    ) -> None:
        """Set pipeline function attributes on *config* from a *funcs* dict.

        Sets ``_preprocess_function``, ``_process_function``,
        ``_peakpick_function`` on *config* if present in *funcs*.
        Missing keys are left unchanged.

        Parameters
        ----------
        config : PipelineConfigurator
            The config instance to modify.
        funcs : dict[str, Callable]
            Dict mapping attribute names to callables (e.g. from
            :meth:`_load_functions_from_py`).
        """
        for attr_name in ('_preprocess_function', '_process_function', '_peakpick_function'):
            if attr_name in funcs:
                setattr(config, attr_name, funcs[attr_name])

    # ------------------------------------------------------------------ #
    #  Full config replacement from YAML                                  #
    # ------------------------------------------------------------------ #

    def load_config(self, yaml_path: str) -> None:
        """Fully replace the current configuration from a YAML file.

        Unlike :meth:`Configs.update`, this method **completely replaces**
        ``self.configs`` with the contents of the YAML file (no merging).

        If a companion ``.py`` file with the same base name exists next to
        the YAML file, custom pipeline functions are restored from it via
        :meth:`_load_functions_from_py`.  If no ``.py`` file is found, the
        step functions fall back to the built-in kneading defaults.

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

        # Validate and fully replace configs (no merge)
        self.replace_config(loaded)

        # Restore custom functions from companion .py (if any)
        funcs = self._load_functions_from_py(py_path)
        if funcs:
            self._restore_functions_from_dict(self, funcs)
        else:
            # No companion .py — fall back to built-in kneading defaults
            self._preprocess_function = preprocess_configuration_base
            self._process_function = process_spectra_base
            self._peakpick_function = peakpicking_base

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
        """Parse the source of *func* with ``ast`` and return only callables
        that receive configs via a ``**configs`` unfold pattern.

        The method identifies the ``configs`` parameter in the function
        signature, then traces how config values are passed to sub-calls:

        * ``func(x, y, **configs['key'])`` — the sub-call is stored under
          config key ``'key'`` (instead of ``func.__name__``), and ``x, y``
          are marked as *explicitly provided* (their defaults are skipped).
        * ``func(x, **var)`` where ``var = configs['key']`` — same logic
          via variable tracing.
        * ``func(x, y, **configs)`` — the whole configs dict is passed;
          the sub-call is stored under its own name, with ``x, y`` marked
          as explicit.

        Calls that do **not** involve a ``**configs`` unfold are silently
        skipped — only configs-fed callables are returned.

        Returns
        -------
        list of tuple
            Each element is one of:

            * ``(callable, config_key, explicit_params)`` — plain function
              with configs unfold
            * ``((method, class), config_key, explicit_params)`` — method
              with configs unfold
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

        # --------------------------------------------------------------- #
        #  1. Find the configs parameter name from the function signature  #
        # --------------------------------------------------------------- #
        configs_param_name: str | None = None
        try:
            sig = inspect.signature(func)
            for p_name, p in sig.parameters.items():
                # Look for a parameter typed as Configs | PipelineConfigurator
                # or simply named 'configs'
                if p_name == 'configs':
                    configs_param_name = p_name
                    break
                # Also check annotation
                ann = p.annotation
                if ann is not inspect.Parameter.empty:
                    ann_str = str(ann)
                    if 'Configs' in ann_str or 'PipelineConfigurator' in ann_str:
                        configs_param_name = p_name
                        break
        except (ValueError, TypeError):
            pass

        # --------------------------------------------------------------- #
        #  2. Build variable tracing maps                                  #
        # --------------------------------------------------------------- #
        # var -> class name  (e.g. baseline = Baseline(mz) -> {"baseline": "Baseline"})
        var_to_class: dict[str, str] = {}
        # var -> config key  (e.g. smooth_configs = configs['smoothing'] -> {"smooth_configs": "smoothing"})
        var_to_config_key: dict[str, str] = {}
        # var -> step name  (e.g. step_cfg = configs.get_step_configs('process') -> {"step_cfg": "process"})
        var_to_step_name: dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    rhs = node.value

                    # --- var = ClassName(...) ---
                    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name):
                        var_to_class[target.id] = rhs.func.id

                    # --- var = configs.get_step_configs('step_name') ---
                    if (configs_param_name is not None
                            and isinstance(rhs, ast.Call)
                            and isinstance(rhs.func, ast.Attribute)
                            and isinstance(rhs.func.value, ast.Name)
                            and rhs.func.value.id == configs_param_name
                            and rhs.func.attr == 'get_step_configs'
                            and rhs.args
                            and isinstance(rhs.args[0], ast.Constant)
                            and isinstance(rhs.args[0].value, str)):
                        var_to_step_name[target.id] = rhs.args[0].value

                    # --- var = configs['key'] ---
                    if (configs_param_name is not None
                            and isinstance(rhs, ast.Subscript)
                            and isinstance(rhs.value, ast.Name)
                            and rhs.value.id == configs_param_name
                            and isinstance(rhs.slice, ast.Constant)
                            and isinstance(rhs.slice.value, str)):
                        var_to_config_key[target.id] = rhs.slice.value

                    # --- var = configs.get('key')  or  configs.get('key', default) ---
                    if (configs_param_name is not None
                            and isinstance(rhs, ast.Call)
                            and isinstance(rhs.func, ast.Attribute)
                            and isinstance(rhs.func.value, ast.Name)
                            and rhs.func.value.id == configs_param_name
                            and rhs.func.attr == 'get'
                            and rhs.args
                            and isinstance(rhs.args[0], ast.Constant)
                            and isinstance(rhs.args[0].value, str)):
                        var_to_config_key[target.id] = rhs.args[0].value

                    # --- var = step_var['key']  (where step_var comes from get_step_configs) ---
                    if (isinstance(rhs, ast.Subscript)
                            and isinstance(rhs.value, ast.Name)
                            and rhs.value.id in var_to_step_name
                            and isinstance(rhs.slice, ast.Constant)
                            and isinstance(rhs.slice.value, str)):
                        var_to_config_key[target.id] = rhs.slice.value

        # --------------------------------------------------------------- #
        #  3. Helper: extract config_key and explicit params from a Call   #
        # --------------------------------------------------------------- #
        def _get_configs_unfold_info(
            call_node: ast.Call,
        ) -> tuple[str | None, set[int], set[str], bool]:
            """Inspect a Call node for ``**configs[key]`` or ``**var`` patterns.

            Returns
            -------
            config_key : str or None
                The config key being unfolded, or None if whole ``**configs``
                is passed (or no unfold detected).
            explicit_pos_indices : set of int
                Indices of positional arguments that are explicitly provided
                (simple names, not ``*args``).  These are resolved to actual
                parameter names in step 4 after the callable is identified.
            explicit_kwarg_names : set of str
                Names of keyword arguments that are explicitly provided
                (not via ``**unfold``).
            has_configs_unfold : bool
                ``True`` if a ``**configs``-based unfold was detected
                (either ``**configs['key']``, ``**configs``, or ``**var``
                where ``var`` traces back to ``configs['key']``).
            """
            config_key: str | None = None
            explicit_pos_indices: set[int] = set()
            explicit_kwarg_names: set[str] = set()
            has_configs_unfold: bool = False

            for kw in call_node.keywords:
                if kw.arg is None:  # **kwargs unfold
                    if isinstance(kw.value, ast.Subscript):
                        # **configs['key']  or  **step_var['key']
                        sub = kw.value
                        if (isinstance(sub.value, ast.Name)
                                and isinstance(sub.slice, ast.Constant)
                                and isinstance(sub.slice.value, str)):
                            if sub.value.id == configs_param_name:
                                config_key = sub.slice.value
                                has_configs_unfold = True
                            elif sub.value.id in var_to_step_name:
                                config_key = sub.slice.value
                                has_configs_unfold = True
                    elif isinstance(kw.value, ast.Name):
                        # **var  (could be **configs or **smooth_configs)
                        var_name = kw.value.id
                        if var_name == configs_param_name:
                            config_key = None  # whole configs passed
                            has_configs_unfold = True
                        elif var_name in var_to_config_key:
                            config_key = var_to_config_key[var_name]
                            has_configs_unfold = True
                        elif var_name in var_to_step_name:
                            config_key = None  # whole step configs passed
                            has_configs_unfold = True

            # Collect indices of positional args that are simple names
            for idx, arg in enumerate(call_node.args):
                if isinstance(arg, ast.Name):
                    explicit_pos_indices.add(idx)
                elif isinstance(arg, ast.Starred):
                    # *args — skip, these are not explicit param names
                    pass

            # Collect explicitly provided keyword arg names
            for kw in call_node.keywords:
                if kw.arg is not None:  # regular kwarg, not **unfold
                    explicit_kwarg_names.add(kw.arg)

            return config_key, explicit_pos_indices, explicit_kwarg_names, has_configs_unfold

        # --------------------------------------------------------------- #
        #  4. Main resolution loop — only keep calls with configs unfold   #
        # --------------------------------------------------------------- #
        builtin_names: set[str] = set(dir(builtins))
        func_globals = getattr(func, "__globals__", {})

        def _resolve_explicit_params(
            callable_obj: Callable,
            pos_indices: set[int],
            kwarg_names: set[str],
            cls: type | None = None,
        ) -> set[str]:
            """Resolve positional argument indices to parameter names.

            For unbound methods (where *cls* is provided and the callable
            is not ``__init__``), the first parameter (``self``) is
            automatically skipped since it is never present in the AST
            call.  For ``__init__`` or when *callable_obj* is itself a
            class, the class signature is used directly (no ``self``).
            """
            param_names: set[str] = set(kwarg_names)
            if not pos_indices:
                return param_names
            try:
                # callable_obj is a class → use its own signature
                if inspect.isclass(callable_obj):
                    sig = inspect.signature(callable_obj)
                elif callable_obj.__name__ == '__init__' and cls is not None:
                    # __init__ → use class signature (no 'self')
                    sig = inspect.signature(cls)
                elif cls is not None:
                    # Unbound method → skip 'self' at index 0
                    sig = inspect.signature(callable_obj)
                    param_list = list(sig.parameters.keys())
                    for idx in sorted(pos_indices):
                        # +1 to skip 'self'
                        actual_idx = idx + 1
                        if actual_idx < len(param_list):
                            param_names.add(param_list[actual_idx])
                    return param_names
                else:
                    sig = inspect.signature(callable_obj)
                param_list = list(sig.parameters.keys())
                for idx in sorted(pos_indices):
                    if idx < len(param_list):
                        param_names.add(param_list[idx])
            except (ValueError, TypeError):
                pass
            return param_names

        resolved: list[Callable | tuple[Callable, type]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # --- Get configs-unfold info for this call ---
            config_key, pos_indices, kwarg_names, has_configs_unfold = _get_configs_unfold_info(node)

            # Skip calls that don't involve a **configs unfold
            if not has_configs_unfold:
                continue

            if isinstance(node.func, ast.Name):
                # Plain name call: msalign(data, ...)
                name = node.func.id
                if name in builtin_names:
                    continue
                obj = func_globals.get(name) or globals().get(name)
                if obj is not None and callable(obj):
                    if inspect.isclass(obj):
                        # Class instantiation — skip, constructor params are
                        # merged into method entries via chained call handling
                        continue
                    else:
                        explicit_params = _resolve_explicit_params(obj, pos_indices, kwarg_names)
                        resolved.append((obj, config_key, explicit_params))

            elif isinstance(node.func, ast.Attribute):
                # Attribute call: obj.method(data, ...)
                method_name = node.func.attr

                # Case A: obj is a simple name
                if isinstance(node.func.value, ast.Name):
                    obj_name = node.func.value.id

                    # A1: obj is a local variable from class instantiation
                    cls_name = var_to_class.get(obj_name)
                    if cls_name is not None:
                        cls_obj = func_globals.get(cls_name) or globals().get(cls_name)
                        if cls_obj is not None and isinstance(cls_obj, type):
                            method_obj = getattr(cls_obj, method_name, None)
                            if method_obj is not None and callable(method_obj):
                                explicit_params = _resolve_explicit_params(method_obj, pos_indices, kwarg_names, cls=cls_obj)
                                resolved.append(((method_obj, cls_obj), config_key, explicit_params))
                                continue

                    # A2: obj is a module (e.g. np.linspace)
                    mod_obj = func_globals.get(obj_name) or globals().get(obj_name)
                    if mod_obj is not None:
                        method_obj = getattr(mod_obj, method_name, None)
                        if method_obj is not None and callable(method_obj):
                            explicit_params = _resolve_explicit_params(method_obj, pos_indices, kwarg_names)
                            resolved.append((method_obj, config_key, explicit_params))
                            continue

                    # A3: obj is a class (e.g. Baseline.asls)
                    if mod_obj is not None and isinstance(mod_obj, type):
                        method_obj = getattr(mod_obj, method_name, None)
                        if method_obj is not None and callable(method_obj):
                            explicit_params = _resolve_explicit_params(method_obj, pos_indices, kwarg_names, cls=mod_obj)
                            resolved.append(((method_obj, mod_obj), config_key, explicit_params))
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
                                # Resolve outer call's positional args against the method
                                explicit_params = _resolve_explicit_params(
                                    method_obj, pos_indices, kwarg_names, cls=cls_obj
                                )
                                # Resolve inner call's positional args against the class constructor
                                inner_pos_indices: set[int] = set()
                                for idx, arg in enumerate(inner_call.args):
                                    if isinstance(arg, ast.Name):
                                        inner_pos_indices.add(idx)
                                # Collect explicit keyword args from the inner call
                                inner_kwarg_names: set[str] = set()
                                for kw in inner_call.keywords:
                                    if kw.arg is not None:  # regular kwarg, not **unfold
                                        inner_kwarg_names.add(kw.arg)
                                if inner_pos_indices or inner_kwarg_names:
                                    inner_explicit = _resolve_explicit_params(
                                        cls_obj, inner_pos_indices, inner_kwarg_names, cls=cls_obj
                                    )
                                    explicit_params |= inner_explicit
                                resolved.append(((method_obj, cls_obj), config_key, explicit_params))
                                continue

        return resolved

class KDEConfigs(BaseModel):
    """
    WIP
    Планы на класс:
    1) Не конфигс как для пайплайна: здесь функция всего только одна для коррекции - такую сложность не делаем
    2) Валидация конфигов для KDE
    3) Методы загрузки/сохранения
    4) Создание дефолтных конфигов простой инициацией
    5) Быстрой настройки (работает скорее как dict)

    Pydantic config for KDE-based m/z peak correction (Pgrouping_KD).
`
    Designed to be:
    - Easy to instantiate with defaults: ``KDEConfigs()``
    - Easy to override: ``KDEConfigs(CountF=5)``
    - Easy to unpack into a function: ``func(**cfg.to_kwargs())``
    - Easy to save/load: ``cfg.save_yaml("kde.yaml")``
    - Easy to update from a dict: ``cfg.update(CountF=5)``
    """
     # --- Bandwidth ---
    KD_bandwidth: str | float = Field("fwhm",
                                      description="Bandwidth selection method or value."
                                      "Options: 'fwhm', 'mz_discret', 'ISJ', 'silverman', 'scott', or a float.")
    bwc: float = Field(
        1.0,
        ge=0.01, le=100.0,
        description="Bandwidth coefficient (multiplier for the selected/computed bandwidth).")

    # --- KDE algorithm ---
    KD_kernel: str = Field(
        "gaussian",
        description="KDE kernel name. See KDEpy documentation for available kernels."
    )
    KDE_algo: str = Field(
        "Tree",
        description="Explicit KDE algorithm (FFTKDE or TreeKDE). Default: 'Tree'"
        "Options: 'FFT', 'Tree'."
    )

    # # --- Peak filtering ---
    # CountF: int = Field(
    #     10, ge=0,
    #     description="Minimum number of occurrences for a peak to be kept."
    # )
    # dupl_drop: bool = Field(
    #     True,
    #     description="Drop duplicate peaks from the result."
    # )
    # min_resolution_ppm: float = Field(
    #     10.0, ge=0,
    #     description="Minimum instrument resolution in ppm. "
    #                 "Controls minimum bandwidth for 'mz_discret' method."
    # )

   # --- m/z splitting ---
    split_mz_min: float = Field(
        10.0, ge=0.0,
        description="Minimum m/z gap to split into separate segments."
    )
    split_peaks_min: int = Field(
        25, ge=1,
        description="Minimum number of peaks per segment."
    )

 # --- m/z scale ---
    account_mzscale: bool = Field(
        True,
        description="Account for m/z discretisation when computing bandwidth."
    )

# # --- Extra kwargs for mspeaks_KD ---
#     params2mspeaks_KD: dict[str, Any] = Field(
#         default_factory=dict,
#         description="Additional keyword arguments passed to mspeaks_KD."
#     )

    @field_validator("KD_bandwidth")
    @classmethod
    def _validate_bandwidth(cls, v):
        valid_strings = {"fwhm", "mz_discret", "isj", "silverman", "scott"}
        if isinstance(v, str) and v.lower() not in valid_strings:
            raise ValueError(
                f"Unknown KD_bandwidth '{v}'. "
                f"Valid strings: {valid_strings}. "
                f"Or pass a float directly."
            )
        return v
    @field_validator('KDE_algo')
    @classmethod
    def _validate_KDE_algo(cls, v):
        valid_strings = {"fft", "tree"}
        if isinstance(v, str) and v.lower() not in valid_strings:
            raise ValueError(
                f"Unknown KDE_algo '{v}'. "
                f"Valid strings: {valid_strings}. "
            )
        return v
    @property
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a flat dict suitable for ``**unpacking`` into
        :func:`Pgrouping_KD` or :meth:`DataSet.compute_KDE`.

        Excludes ``params2mspeaks_KD`` (merged into the configs).
        """
        configs = self.model_dump(exclude={"params2mspeaks_KD"})
        # configs.update(self.params2mspeaks_KD)
        return configs
    
    def save_yaml(self, sample: str, dirpath: str) -> None:
        """Save config to a YAML file.

        Callable values (e.g. ``KDE_algo``) are serialised using the
        ``!obj`` YAML tag — the same mechanism used by
        :class:`PipelineConfigurator`.

        Parameters
        ----------
        sample : str
            Sample name used to construct the filename
            ``<sample>_kde_recipe.yaml``.
        dirpath : str
            Directory to write the YAML file into.
        """
        path = os.path.join(dirpath, sample + "_kde_recipe.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._serializable_dict(), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> "KDEConfigs":
        """Load config from a YAML file.

        Supports the ``!obj`` YAML tag for restoring callable values
        (e.g. ``KDE_algo: !obj KDEpy.FFTKDE``) — the same mechanism used
        by :class:`PipelineConfigurator`.

        Parameters
        ----------
        path : str
            Path to the ``*_kde_recipe.yaml`` file.

        Returns
        -------
        KDEConfigs
            A new instance populated from the YAML file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**data)

    def update(self, **overrides: Any) -> None:
        """
        Update config fields in-place (dict-like convenience).

        Example::

            cfg = KDEConfigs()
            cfg.update(CountF=5, draw=False)
        """
        for key, value in overrides.items():
            setattr(self, key, value)

    def with_overrides(self, **overrides: Any) -> "KDEConfigs":
        """
        Return a **copy** with the given fields overridden (immutable-style).

        Example::

            cfg = KDEConfigs()
            cfg2 = cfg.with_overrides(CountF=5, draw=False)
        """
        return self.model_copy(update=overrides)

    def _serializable_dict(self) -> dict:
        """Return a YAML-safe dict."""
        d = self.model_dump()
        return d


# --------------------------------------------------------------------------- #
#  PreparedDataSource - per-ROI configuration manager linked to a DataSource  #
# --------------------------------------------------------------------------- #

class PreparedDataSource():
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
                 kde_configs: str | KDEConfigs | None = None,
                 rebuild_metadata: bool = False,
                 **kwargs):
        #: Path to the source file that was used to load configs.
        self._configs_source_path: str | None = None
        #: Linked DataSource (optional).
        self._datasource: DataSource | None = None
        #: Per-ROI PipelineConfigurator instances.
        self.roi_configs: dict[str, PipelineConfigurator] = {}
        #: Per-ROI KDEConfigs instances.
        self.roi_kde_configs: dict[str, KDEConfigs] = {}
        #: Base config used as a template when new ROIs are added via set_link.
        self._base_configs: PipelineConfigurator | None = None
        self._base_kde_configs: KDEConfigs | None = None
        
        kwargs_kde, kwargs = self._resolve_kwargs(kwargs)
        # --- Resolve kde_configs_source ---
        self._load_kde_configs(kde_configs, **kwargs_kde)
        # --- Resolve configs_source ---
        if configs_source is not None:
            self._load(configs_source, **kwargs)
        # --- Link datasource if provided ---
        if datasource is not None:
            self.set_link(datasource, rebuild_metadata=rebuild_metadata)
        
        # for compatability with Pipeline methods

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
        # Accept both string keys and integer keys (PyYAML loads "00" as int 0).
        Not_roi_nests_keys = ("methods", "functions")
        return any(
            (isinstance(k, str) and (k not in Not_roi_nests_keys))
            or isinstance(k, int)
            for k in data
        )

    def _load(self, source: str | PipelineConfigurator | dict[str: PipelineConfigurator], **kwargs) -> None:
        """Load configuration from *source* into ``roi_configs`` or ``_base_configs``.

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
                            PipelineConfigurator._restore_functions_from_dict(
                                cc, roi_funcs[roi_name]
                            )
                        if kwargs:
                            cc.update(kwargs)
                            
                        self.roi_configs[roi_name] = cc
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
        if isinstance(source, dict):
            for roi, configs in source.items():
                if roi in self.roi_configs:
                    self.roi_configs[roi] = configs

        self._base_configs = PipelineConfigurator(source, **kwargs)
    
    def _load_kde_configs(self,
                          source: str | KDEConfigs | dict[str, KDEConfigs] | None = None,
                          **kwargs) -> None:
        
        if source is None:
            return
        if isinstance(source, KDEConfigs):
            if self._datasource:
                for roi_name in self.roi_kde_configs:
                    self.roi_kde_configs[roi_name] = source
                    self.roi_kde_configs[roi_name].update(**kwargs)
            else:
                self._base_kde_configs = source
                self._base_kde_configs.update(**kwargs)
        elif isinstance(source, str):
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
                if self._is_per_roi_kde_yaml(loaded):
                    for roi_name, roi_data in loaded.items():
                        # PyYAML loads unquoted keys like "00" as integer 0.
                        # ROI names are always strings in the application.
                        if isinstance(roi_name, int):
                            roi_name = str(roi_name)
                        if self._datasource:
                            if roi_name in self.roi_kde_configs:
                                self.roi_kde_configs[roi_name] = KDEConfigs(**roi_data)
                                self.roi_kde_configs[roi_name].update(**kwargs)
                            else:
                                warnings.warn(
                                    f"YAML file '{path}' contains per-ROI configs for ROI '{roi_name}', "
                                    f"but sample doesn't have ROI '{roi_name}'. Skipping ROI '{roi_name}'."
                                )
                        else:
                            self.roi_kde_configs[roi_name] = KDEConfigs(**roi_data)
                
                else:
                    if self._datasource:
                        for roi_name in self.roi_kde_configs:
                            self.roi_kde_configs[roi_name] = KDEConfigs(**loaded)
                            self.roi_kde_configs[roi_name].update(**kwargs)
                    else:
                        self._base_kde_configs = KDEConfigs(**loaded)
                        self._base_kde_configs.update(**kwargs)
            else:
                raise FileNotFoundError(f"YAML file '{path}' not found.")

        else:
            for roi_name, roi_data in source.items():
                if self._datasource:
                    if roi_name in self.roi_kde_configs:
                        self.roi_kde_configs[roi_name] = KDEConfigs(**roi_data)
                        self.roi_kde_configs[roi_name].update(**kwargs)
                    else:
                        warnings.warn(
                            f"YAML file '{path}' contains per-ROI configs for ROI '{roi_name}', "
                            f"but sample doesn't have ROI '{roi_name}'. Skipping ROI '{roi_name}'."
                        )
                else:
                    self.roi_kde_configs[roi_name] = KDEConfigs(**roi_data)
                    self.roi_kde_configs[roi_name].update(**kwargs)
                
        
    def _is_per_roi_kde_yaml(self, data: dict) -> bool:
        """Heuristic: does *data* look like a per-ROI config dump?"""
        if not isinstance(data, dict):
            return False
        # Per-ROI files have top-level keys like "roi_00", "roi_01", ...
        for value in data.values():
            if isinstance(value, dict):
                if any(k in KDEConfigs().to_dict.keys() for k in value.keys()):
                    return True
            else:
                return False
    # ------------------------------------------------------------------ #
    #  DataSource linking                                                #
    # ------------------------------------------------------------------ #

    def set_link(self, 
                 datasource: DataSource | str, 
                 rebuild_metadata: bool = False) -> None:
        """Link this :class:`PreparedDataSource` to a :class:`DataSource`.

        Creates per-ROI :class:`PipelineConfigurator` for every ROI found in the
        datasource's metadata.  Existing per-ROI configs are preserved.

        Parameters
        ----------
        datasource : DataSource or str
            :class:`DataSource` instance or path to a data-source file.
        """
        if isinstance(datasource, str):
            datasource = DataSource(datasource,rebuild_metadata)

        self._datasource = datasource
        datasource.create_metafile(rebuild_metadata = rebuild_metadata)
        # Get ROI names from the datasource metadata
        roi_names: list[str] = list(datasource.roi_metadata.index)
        
        for roi in roi_names:
            if roi not in self.roi_configs:
                # Create a fresh PipelineConfigurator for this ROI
                if self._base_configs is not None:
                    self.roi_configs[roi] = copy.deepcopy(self._base_configs)
                else:
                    self.roi_configs[roi] = PipelineConfigurator()
            if roi not in self.roi_kde_configs:
                # Create a fresh KDEConfigs for this ROI
                if self._base_kde_configs is not None:
                    self.roi_kde_configs[roi] = copy.deepcopy(self._base_kde_configs)
                else:
                    self.roi_kde_configs[roi] = KDEConfigs()
        for roi in list(self.roi_configs.keys()):
            if roi not in roi_names:
                del self.roi_configs[roi]
        for roi in list(self.roi_kde_configs.keys()):
            if roi not in roi_names:
                del self.roi_kde_configs[roi]

    def peaklists(self, roi):
        return self._datasource.peaklists(roi)
    def get_mean_spectrum(self, roi: str | None = None, 
                          idxs: np.ndarray | None = None, 
                          mz_range: tuple[float, float] | None = None):
        return self._datasource.get_mean_spectrum(roi, idxs, mz_range)
    def get_coords(self, 
                   idxs: np.ndarray | SliceIndexator | Indexator | int | str | None = None, 
                   extract: list[str] | None = None):
        return self._datasource.get_coords(idxs, extract)
    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #
    
    @property
    def roi_metadata(self) -> pd.DataFrame:
        """ROI metadata from the linked datasource."""
        return self._datasource.roi_metadata
    
    @property
    def file_path(self) -> str | None:
        """Path to the linked datasource file."""
        return self._datasource.file_path
    
    
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
    def peaklists_path(self) -> str | None:
        """Path to the peaklist file."""
        path = self._default_save_path('peaklists.hdf5')
        return path
    @property
    def processed_spectra_path(self) -> str | None:
        """Path to the peaklist file."""
        path = self._default_save_path('processed_spectra.hdf5')
        return path
    @property
    def peaks_density_path(self) -> str | None:
        """Path to the peaklist file."""
        path = self._default_save_path('peaks_density.hdf5')
        return path
    @property
    def rois(self) -> list[str]:
        """List of ROI names currently managed."""
        return list(self.roi_configs.keys())
    @property
    def configs_path(self):
        return self._datasource.configs_path

    # ------------------------------------------------------------------ #
    #  ROI-specific access                                                 #
    # ------------------------------------------------------------------ #

    def __getitem__(self, roi: str) -> PipelineConfigurator:
        """Get the :class:`PipelineConfigurator` for a specific ROI.

        Parameters
        ----------
        roi : str
            ROI name (e.g. ``"R00"``, ``"R01"``).

        Returns
        -------
        PipelineConfigurator
            The full configuration object for this ROI.

        Raises
        ------
        KeyError
            If the ROI is not found.
        """
        if roi not in self.roi_configs:
            raise KeyError(
                f"ROI '{roi}' not found. "
                f"Available ROIs: {list(self.roi_configs.keys())}"
            )
        return self.roi_configs[roi]

    def __contains__(self, roi: str) -> bool:
        """Check whether a ROI is managed."""
        return roi in self.roi_configs

    def __len__(self) -> int:
        """Number of managed ROIs."""
        return len(self.roi_configs)

    def __iter__(self):
        """Iterate over ROI names."""
        return iter(self.roi_configs)

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
            targets = list(self.roi_configs.keys())
        elif isinstance(rois, str):
            targets = [rois]
        else:
            targets = list(rois)

        for roi in targets:
            if roi in self.roi_configs:
                self.roi_configs[roi].update(params_source, **kwargs)
    
    def update_kde(self,
                     params_source: str | None = None,
                    rois: str | list[str] | None = None,
                   **kwargs) -> None:            
        """Update configuration parameters for specific ROI(s).

        If *rois* is ``None``, the update is applied to **all** ROIs.

        Parameters
        ----------
        params_source : str or dict or None
            Path to a YAML file, or a dictionary with parameter overrides.
            Same format as :meth:`KDEConfigs.update`.
        rois : str or list of str or None
            ROI(s) to update.  ``None`` means all ROIs.
        **kwargs
            Parameter overrides applied on top.
        """
        targets: list[str]
        if rois is None:
            targets = list(self.roi_kde_configs.keys())
        elif isinstance(rois, str):
            targets = [rois]
        else:
            targets = list(rois)
        for roi in targets:
            if roi in self.roi_kde_configs:
                if params_source is not None:
                    self.roi_kde_configs[roi].load_yaml(params_source)
                self.roi_kde_configs[roi].update(**kwargs)

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
            targets = list(self.roi_configs.keys())
        elif isinstance(rois, str):
            targets = [rois]
        else:
            targets = list(rois)

        for roi in targets:
            if roi in self.roi_configs:
                self.roi_configs[roi].delete(param_name)

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
        if roi not in self.roi_configs:
            if self._base_configs is not None:
                self.roi_configs[roi] = copy.deepcopy(self._base_configs)
            else:
                self.roi_configs[roi] = PipelineConfigurator()
        return self.roi_configs[roi]
    @staticmethod
    def _resolve_kwargs(kwargs: dict) -> tuple[dict, dict]:
        """Resolve keyword arguments to a KDEconfigs and Configs.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        KDEkwargs
            Resolved kwargs for KDEconfigs.
        Configkwargs
            Resolved kwargs for Configs.
        """
        KDEkwargs = {}
        Configkwargs = {}

        KDEkwargs_list = list(KDEConfigs().to_dict.keys())
        for key in kwargs:
            if key in KDEkwargs_list:
                KDEkwargs[key] = kwargs[key]
            else:
                Configkwargs[key] = kwargs[key]
        return KDEkwargs, Configkwargs

    # ------------------------------------------------------------------ #
    #  Serialisation                                                     #
    # ------------------------------------------------------------------ #

    def _default_save_path(self, suffix = "", prefix = "") -> str:
        """Compute the default save path by delegating to the linked
        :class:`DataSource`.

        Parameters
        ----------
        suffix : str, optional
            File suffix / extension (e.g. ``"peaklists.hdf5"``).  An empty
            string results in no trailing underscore.
        prefix : str, optional
            Optional name prefix prepended as ``<prefix>_`` to ``<sample_name>``.
            Default ``""`` (no prefix).

        Returns
        -------
        str
            ``<datasource_dir>/processed_pelmesha/<prefix>_<sample_name>_<suffix>``
        """
        if self._datasource is not None:
            return self._datasource._default_save_path(suffix, prefix)
        else:
            raise ValueError("No datasource linked to object `PreparedDatasource`")
    def dump(self, path: str | None = None) -> str:
        """Save all per-ROI configurations to a YAML file.

        The output structure preserves each ROI's full config, using
        :meth:`PipelineConfigurator._yaml_with_comments` to include
        pipeline-step comments per ROI::

            roi_00:
              # PREPROCESS step  →  preprocess_configuration_base  (BASE)
              functions:
                ...

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
            path = self._default_save_path('processing_recipe.yaml')

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Build per-ROI YAML using each ROI's PipelineConfigurator._yaml_with_comments()
        lines: list[str] = []
        roi_names = sorted(self.roi_configs.keys())
        for idx, roi_name in enumerate(roi_names):
            config = self.roi_configs[roi_name]
            roi_yaml = config._yaml_with_comments()
            # Always quote ROI name keys so that PyYAML does not interpret
            # numeric-looking names (e.g. "00") as integers on re-load.
            lines.append(f'"{roi_name}":')
            for line in roi_yaml.split("\n"):
                if line.strip() == "" and not line:
                    lines.append("")
                else:
                    lines.append(f"  {line}")
            # Remove trailing blank line(s) from the last ROI block
            while lines and lines[-1].strip() == "":
                lines.pop()
            # Add a blank line between ROIs (except after the last)
            if idx < len(roi_names) - 1:
                lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

        # Save companion .py with per-ROI pipeline functions
        py_path = path[:-5] + ".py" if path.endswith(".yaml") else path + ".py"
        self._dump_roi_functions(py_path)

        return path
    
    def dump_kde_configs(self, path: str | None = None):
        if path is None:
            path = self._default_save_path('kde_recipe.yaml')
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Build per-ROI YAML with quoted keys (same approach as save())
        lines = []
        for roi_name in sorted(self.roi_kde_configs.keys()):
            roi_data = self.roi_kde_configs[roi_name].to_dict
            # Always quote ROI name keys so that PyYAML does not interpret
            # numeric-looking names (e.g. "00") as integers on re-load.
            lines.append(f'"{roi_name}":')
            roi_yaml = yaml.dump(roi_data, default_flow_style=False).rstrip("\n")
            for line in roi_yaml.split("\n"):
                lines.append(f"  {line}")
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")
        
    def _dump_roi_functions(self, py_path: str) -> None:
        """Save per-ROI custom pipeline functions to a companion .py file.

        Reuses :meth:`PipelineConfigurator._get_custom_functions` for
        built-in filtering and :meth:`PipelineConfigurator._get_source`
        for source extraction — the same logic as
        :meth:`PipelineConfigurator._dump_functions`.

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
        # Collect per-ROI custom functions using PipelineConfigurator's own filter
        roi_func_map: dict[str, dict[str, Callable]] = {}
        seen_funcs: dict[str, Callable] = {}  # func_name -> func (deduplicated)

        for roi_name, config in self.roi_configs.items():
            custom = PipelineConfigurator._get_custom_functions(config)
            if custom:
                roi_func_map[roi_name] = custom
                for attr_name, func in custom.items():
                    if func.__name__ not in seen_funcs:
                        seen_funcs[func.__name__] = func

        # If all functions are built-in, do not write the .py file at all
        if not seen_funcs:
            if os.path.exists(py_path):
                os.remove(py_path)
            return

        lines: list[str] = []
        lines.append("# Per-ROI custom pipeline functions for PreparedDataSource\n")
        lines.append("# Auto-generated - do not edit manually\n\n")
        lines.append("import numpy as np\n\n")

        # Write deduplicated function sources
        for func_name, func in seen_funcs.items():
            source = PipelineConfigurator._get_source(func)
            if source:
                lines.append(f"# --- {func_name} ---\n")
                lines.append(textwrap.dedent(source))
                lines.append("\n\n")
            else:
                lines.append(f"# {func_name!r} (source not available)\n\n")

        # Write the per-ROI mapping dict
        lines.append("# Auto-generated per-ROI function mapping\n")
        lines.append("__pipeline_functions__ = {\n")
        for roi_name, funcs in roi_func_map.items():
            lines.append(f"    '{roi_name}': {{\n")
            for attr_name, func in funcs.items():
                lines.append(f"        '{attr_name}': {func.__name__},\n")
            lines.append("    },\n")
        lines.append("}\n")

        with open(py_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))

    def save(self, path: str | None = None) -> str:
        """Alias for :meth:`dump`."""
        return self.dump(path)

    # ------------------------------------------------------------------ #
    #  Representation                                                    #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lines = ["PreparedDataSource("]
        if self._datasource is not None:
            lines.append(f"  datasource: {self._datasource.sample_name}")
        lines.append(f"  rois: {list(self.roi_configs.keys())}")
        lines.append(")")
        return "\n".join(lines)
    
