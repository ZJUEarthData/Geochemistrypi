import os
from typing import Optional


def ensure_directories_exist(directory_paths: Optional[list], base_path: str = None) -> None:
    """Ensure that the specified directories exist. If not, create them.

    Parameters
    ----------
    directory_paths : list of str
        List of directory paths to check and create if necessary.
    base_path : str, optional
        Base path to prepend to each directory path in `directory_paths`.

    Returns
    -------
    None
    """
    for directory_path in directory_paths:
        full_path = os.path.join(base_path, directory_path) if base_path else directory_path
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        else:
            pass


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


def toggle_address_status(status: str = None, training_data_path: str = None, user_conformation: int = 0) -> list:
    """Toggles the training data path and output path based on the provided status.

    Parameters
    ----------
    status : str, optional
        The status value, which can be "1" or "2".
        - "1": Use the input and output paths in command line mode.
        - "2": Retrieves all Excel files from the "data" folder on the desktop as the training data path, and sets the output path to the desktop.
    training_data_path : str, optional
        The path to the training data. This parameter is used when `status` is "1".
    user_conformation : int, optional
        Whether the user needs to confirm that the processing file has been placed.

    Returns
    -------
    paths : list
        A list containing the training data path and the output path.

    """

    if int(status) == 1:
        working_path = os.path.dirname(os.getcwd())
    elif int(status) == 2:
        file_name = ["geopi_data_input", "geopi_data_output"]
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        ensure_directories_exist(file_name, desktop_path)
        if int(user_conformation) == 1:
            while True:
                confirmation_parameter = input("Place the data you need to process in the geopi_data_input folder on the desktop [y]: ")
                if confirmation_parameter.lower() == "y":
                    break
                else:
                    print("Please confirm again and enter 'y'!")
        training_data_path = list_excel_files(os.path.join(desktop_path, "geopi_data_input"))
        working_path = desktop_path
    else:
        raise ValueError("Invalid status value. It should be '1' or '2'.")

    return [training_data_path, working_path]
