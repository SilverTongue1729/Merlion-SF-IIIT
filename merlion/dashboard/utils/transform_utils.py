#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Utilities for handling data preprocessing transforms in the dashboard.
"""
from collections import OrderedDict
from enum import Enum
import inspect
from typing import Dict, List, Any

from merlion.transform.factory import TransformFactory
from merlion.transform.sequence import TransformSequence


# Commonly used transforms with sensible defaults for dashboard
COMMON_TRANSFORMS = {
    "MeanVarNormalize": {},
    "MinMaxNormalize": {},
    "DifferenceTransform": {},
    "MovingAverage": {"n_steps": 5},
    "ExponentialMovingAverage": {"alpha": 0.3},
    "SpikeAmplification": {"spike_threshold_quantile": 0.75, "amplification_factor": 1.5, "use_power": False},
    "BoxCoxTransform": {"lmbda": None},
    "LowerUpperClip": {"lower": -100.0, "upper": 100.0},  # Provide reasonable defaults instead of None
    "ConvexHullMethod": {"slope_limit": 1000000000.0, "relaxation_tp": 10},
    "RollingMeans": {"window_size": 5},
    "PeakMultiplier": {"peak_multiplier": 1.5, "peak_threshold": 8.0, "hill_dist": 5},
}


def get_available_transforms():
    """Get list of available transforms for dashboard."""
    return list(COMMON_TRANSFORMS.keys())


def get_all_transforms():
    """Get all available transforms from factory."""
    return list(TransformFactory.import_alias.keys())


def get_transform_param_info(transform_name: str) -> OrderedDict:
    """
    Get parameter information for a transform.

    :param transform_name: Name of the transform class
    :return: OrderedDict with parameter info
    """

    def is_enum(t):
        return isinstance(t, type) and issubclass(t, Enum)

    def is_valid_type(t):
        return t in [int, float, str, bool, list, tuple, dict, type(None)] or is_enum(t)

    transform_class = TransformFactory.get_transform_class(transform_name)
    param_info = OrderedDict()
    signature = inspect.signature(transform_class.__init__).parameters

    for name, param in signature.items():
        if name in ["self", "kwargs"]:
            continue

        value = param.default
        if value == param.empty:
            value = ""

        # Handle None defaults - use annotation for type, or default to float
        if value is None:
            if param.annotation != param.empty and param.annotation is not None:
                param_type = param.annotation
            else:
                # For numeric parameters with None defaults (like lower/upper bounds), default to float
                param_type = float
            param_info[name] = {"type": param_type, "default": None}
        elif is_valid_type(type(param.default)):
            value = value.name if isinstance(value, Enum) else value
            param_info[name] = {"type": type(param.default), "default": value}
        elif is_valid_type(param.annotation):
            value = value.name if isinstance(value, Enum) else value
            param_info[name] = {"type": param.annotation, "default": value}

    return param_info


def parse_transform_parameters(transform_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate transform parameters from dashboard input.

    :param transform_name: Name of the transform
    :param params: Dictionary of parameter name -> value (as strings from UI)
    :return: Dictionary of parsed parameters ready for transform creation
    """
    param_info = get_transform_param_info(transform_name)
    kwargs = {}

    for name, value in params.items():
        if name not in param_info:
            continue

        info = param_info[name]
        value_type = info["type"]

        # Handle None/null values
        if value is None or (isinstance(value, str) and value.lower() in ["none", "null", ""]):
            kwargs[name] = None
        elif value_type in [int, float, str]:
            try:
                kwargs[name] = value_type(value)
            except (ValueError, TypeError):
                # Use default if conversion fails
                if info["default"] != "":
                    kwargs[name] = info["default"]
        elif issubclass(value_type, Enum):
            valid_enum_values = value_type.__members__.keys()
            if value in valid_enum_values:
                kwargs[name] = value_type[value]
        elif value_type == bool:
            if isinstance(value, bool):
                kwargs[name] = value
            elif isinstance(value, str):
                kwargs[name] = value.lower() in ["true", "1", "yes"]
        else:
            # For complex types, try to use as-is
            kwargs[name] = value

    return kwargs


def create_transform_from_config(transform_name: str, params: Dict[str, Any]):
    """
    Create a transform instance from configuration.

    :param transform_name: Name of the transform class
    :param params: Dictionary of parameters (already parsed)
    :return: Transform instance
    """
    return TransformFactory.create(transform_name, **params)


def create_transform_sequence(transform_configs: List[Dict[str, Any]]):
    """
    Create a sequence of transforms from configurations.

    :param transform_configs: List of dicts, each with 'name' and 'params' keys
    :return: TransformSequence instance or single transform if only one
    """
    if not transform_configs:
        return None

    transforms = []
    for config in transform_configs:
        transform_name = config.get("name")
        params = config.get("params", {})

        # Parse parameters
        parsed_params = parse_transform_parameters(transform_name, params)

        # Create transform
        transform = create_transform_from_config(transform_name, parsed_params)
        transforms.append(transform)

    if len(transforms) == 1:
        return transforms[0]
    else:
        return TransformSequence(transforms)


def get_transform_defaults(transform_name: str) -> Dict[str, Any]:
    """
    Get default parameter values for a transform.

    :param transform_name: Name of the transform
    :return: Dictionary of default values
    """
    if transform_name in COMMON_TRANSFORMS:
        return COMMON_TRANSFORMS[transform_name].copy()

    # Get defaults from signature
    param_info = get_transform_param_info(transform_name)
    defaults = {}
    for name, info in param_info.items():
        if info["default"] != "":
            defaults[name] = info["default"]

    return defaults
