# -*- coding: utf-8 -*-
"""
internal_standard method wrapper.
Wrap your Hg_Internal_standard_method.py logic here.
"""
from typing import Dict, List


def list_elements() -> List[str]:
    # list of elements currently supported by this method
    return ["Hg"]


def run(element: str, input_path: str, out_dir: str, **kwargs) -> Dict:
    # Dispatch by element; for now only Hg
    if element != "Hg":
        raise NotImplementedError(f"{element} not supported by internal_standard yet.")
    # You can either import your existing function or reimplement here.
    # For example:
    from geochemistrypi.chemical_modeling.process.hg_internal import run as hg_run

    out_path = hg_run(input_path, out_dir)
    return {"status": "success", "out_path": out_path}
