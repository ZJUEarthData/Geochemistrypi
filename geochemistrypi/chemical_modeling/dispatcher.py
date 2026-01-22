# -*- coding: utf-8 -*-
"""
Dispatcher: discover task modules under model.func and provide introspection helpers.
"""
import importlib
import pkgutil
from typing import Dict, List

BASE_PKG = "geochemistrypi.chemical_modeling.model.func"


def discover_tasks() -> List[str]:
    """Return list of task package names (e.g. 'algo_fractionation')."""
    pkg = importlib.import_module(BASE_PKG)
    return [name for _, name, ispkg in pkgutil.iter_modules(pkg.__path__) if ispkg]


def load_task_module(task_name: str):
    """Import and return the task module (e.g. 'geochemistrypi.chemical_modeling.model.func.algo_fractionation')."""
    fullname = f"{BASE_PKG}.{task_name}"
    return importlib.import_module(fullname)


def list_task_methods(task_name: str) -> Dict[str, str]:
    mod = load_task_module(task_name)
    return getattr(mod, "list_methods")()


def list_method_elements(task_name: str, method: str) -> List[str]:
    mod = load_task_module(task_name)
    return getattr(mod, "list_elements")(method)


def run_task_method(task_name: str, method: str, element: str, input_path: str, out_dir: str, **kwargs):
    mod = load_task_module(task_name)
    return getattr(mod, "run")(method, element, input_path, out_dir, **kwargs)
