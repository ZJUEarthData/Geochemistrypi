import os

from ..enum import DataSource
from .base import list_excel_files


def toggle_data_source(data_source: DataSource = None) -> list:
    """Toggle the training data path and output path based on the provided status.

    Parameters
    ----------
    status : str, optional
        The status value, which can be "1" or "2".
        - "1": Use the input and output paths in command line mode.
        - "2": Retrieves all Excel files from the "data" folder on the desktop as the training data path, and sets the output path to the desktop.

    training_data_path : str, optional
        The path to the training data. This parameter is used when `status` is "1".

    Returns
    -------
    paths : list
        A list containing the training data path and the output path.

    """

    if data_source == DataSource.BUILT_IN:
        working_path = os.path.dirname(os.getcwd())
    elif data_source == DataSource.DESKTOP:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        existing_excel_files = list_excel_files(os.path.join(desktop_path, "geopi_input"))
        working_path = desktop_path

    return [existing_excel_files, working_path]
