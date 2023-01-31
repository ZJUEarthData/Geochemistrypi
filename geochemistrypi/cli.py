# -*- coding: utf-8 -*-
import typer
from .data_mining.pipeline import pipeline


app = typer.Typer()


@app.callback()
def callback():
    """
    Geochemistry π is a Python framework for data-driven geochemistry discovery.
    It automates data mining process with frequently-used machine learning algorithm by providing the users with options to choose.
    """


@app.command()
def data_mining(data: str = ""):
    """Apply data mining technique with supervised learning and unsupervised learning methods."""
    pipeline(data)

