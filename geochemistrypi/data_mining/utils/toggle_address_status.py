import os


def list_excel_files(directory: str) -> list:
    """Recursively lists all Excel files (including .xlsx, .xls, and .csv) in the specified directory and its subdirectories.

    Parameters
    ----------
    directory : str
        The path to the directory to search for Excel files.

    Returns
    -------
    excel_files : list
        A list of file paths for all Excel files found.

    Notes
    -----
    (1) The function uses `os.walk` to traverse the directory and its subdirectories.
    (2) Only files with extensions .xlsx, .xls, and .csv are considered as Excel files.
    """
    excel_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".csv"):
                excel_files.append(os.path.join(root, file))
    return excel_files


def toggle_address_status(status: str = None, training_data_path: str = None) -> list:
    """Toggles the training data path and output path based on the provided status.

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

    if int(status) == 1:
        working_path = os.path.dirname(os.getcwd())
    elif int(status) == 2:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        training_data_path = list_excel_files(os.path.join(desktop_path, "data"))
        working_path = desktop_path
    else:
        raise ValueError("Invalid status value. It should be '1' or '2'.")

    return [training_data_path, working_path]
