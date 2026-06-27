# -*- coding: utf-8 -*-
"""
Fractionation task package API.
Provides list_methods, list_elements and run(...) that dispatch to method-specific modules.
"""
from typing import Dict, List

try:
    from . import internal_standard  # type: ignore
except Exception:
    internal_standard = None

try:
    from . import double_spike  # type: ignore
except Exception:
    double_spike = None


def list_methods() -> Dict[str, str]:
    return {
        "internal_standard": "Internal standard method",
        "double_spike": "Double-spike (double-diluent) method",
    }


def list_elements(method: str) -> List[str]:
    if method == "internal_standard":
        if internal_standard and hasattr(internal_standard, "list_elements"):
            return internal_standard.list_elements()
        return []
    if method == "double_spike":
        if double_spike and hasattr(double_spike, "list_elements"):
            return double_spike.list_elements()
        return []
    return []


def run(method: str, element: str, input_path: str, out_dir: str, **kwargs):
    """
    Dispatch to the selected method module.
    return method {"status":"success","out_path":...}）。
    """
    if method == "internal_standard":
        if not internal_standard or not hasattr(internal_standard, "run"):
            raise NotImplementedError("internal_standard method not implemented in algo_fractionation package.")
        return internal_standard.run(element, input_path, out_dir, **kwargs)
    if method == "double_spike":
        if not double_spike or not hasattr(double_spike, "run"):
            raise NotImplementedError("double_spike method not implemented in algo_fractionation package.")
        return double_spike.run(element, input_path, out_dir, **kwargs)
    raise ValueError(f"Unknown method: {method}")
