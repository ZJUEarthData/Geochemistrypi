# -*- coding: utf-8 -*-
"""
Auto Export Tool module for chemical modeling.
Provides automated data export from geochemical instrument software.
"""


def list_methods():
    """Return available methods for this task."""
    return {"batch_export": "Batch export data from geochemical instruments"}


def list_elements(method):
    """Return available elements for the given method."""
    if method == "batch_export":
        return ["general"]
    return []


def run(method, element, input_path, out_dir, **kwargs):
    """Run the specified method with the given element."""
    if method == "batch_export" and element == "general":
        from .auto_export import run_auto_export

        return run_auto_export(input_path, out_dir, **kwargs)
    else:
        raise ValueError(f"Unknown method/element combination: {method}/{element}")
