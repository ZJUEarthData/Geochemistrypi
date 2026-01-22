# -*- coding: utf-8 -*-
"""
double_spike method wrapper.
Wrap your Mo_Double_spike_method_* logic here.
"""
from typing import Dict, List


def list_elements() -> List[str]:
    return ["Mo"]


def run(element: str, input_path: str, out_dir: str, solver: str = "fsolve", **kwargs) -> Dict:
    if element != "Mo":
        raise NotImplementedError(f"{element} not supported by double_spike yet.")
    from geochemistrypi.chemical_modeling.process.mo_double_spike import run as mo_run

    out_path = mo_run(input_path, out_dir, solver=solver)
    return {"status": "success", "out_path": out_path}
