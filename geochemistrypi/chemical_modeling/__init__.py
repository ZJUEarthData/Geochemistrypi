# -*- coding: utf-8 -*-
"""Chemical-modeling package.

The CLI pulls in the data-mining runtime for shared console helpers.  Import it
only when the CLI is actually used so API callers can load the lightweight
dispatcher without installing the full machine-learning stack.
"""


def cli_pipeline(*args, **kwargs):
    """Load and run the interactive CLI pipeline on demand."""
    from .cli_pipeline import cli_pipeline as run_cli_pipeline

    return run_cli_pipeline(*args, **kwargs)

__all__ = ["cli_pipeline"]
