import os


def list_excel_files(directory):
    """Recursively lists all Excel files (including .xlsx, .xls, and .csv) in the specified directory and its subdirectories.

    Parameters:
    directory (str): The path to the directory to search for Excel files.

    Returns:
    list: A list of file paths for all Excel files found.
    """
    excel_name = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".csv"):
                excel_name.append(os.path.join(root, file))
    return excel_name


def toggle_address_status(status: str = None, training_data_path: str = None):
    """Toggles the training data path and output path based on the provided status.

    Args:
    status (str): The status value, which can be "1" or "2".
        - "1": Uses the provided `training_data_path` as the training data path, and sets the output path to the parent directory of the current working directory.
        - "2": Retrieves all Excel files from the "data" folder on the desktop as the training data path, and sets the output path to the desktop.

    Returns:
    list: A list containing the training data path and the output path.
    """

    if int(status) == 1:
        training_data_path = training_data_path
        working_path = os.path.dirname(os.getcwd())
    elif int(status) == 2:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        training_data_path = list_excel_files(os.path.join(desktop_path, "data"))
        working_path = desktop_path
    else:
        pass
    return [training_data_path, working_path]
