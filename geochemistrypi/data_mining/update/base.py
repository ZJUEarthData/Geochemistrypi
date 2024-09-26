import subprocess
from typing import Optional

import requests


def is_software_installed(software_name: str) -> bool:
    """
    Check if a specified software is installed on the system.

    Parameters
    ----------
    software_name : str
        The name of the software to check for installation.

    Returns
    -------
    bool
        True if the software is installed, False otherwise.

    """

    try:
        # Use the where command to find the executable file of the software
        result = subprocess.run(["where", software_name], capture_output=True, text=True)

        # If the where command returns 0, it means the software was found
        if result.returncode == 0:
            print(f"{software_name} is installed.")
            return True
        else:
            print(f"{software_name} is not installed.")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def run_bat_file(bat_file_path: Optional[list]) -> None:
    """
    Execute a specified batch file.

    Parameters
    ----------
    bat_file_path : list
        The path to the batch file to be executed.
    """

    try:
        subprocess.run([bat_file_path], check=True, shell=True)
        print(f"Bat file executed successfully: {bat_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the bat file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_latest_release_version(repo_owner: str, repo_name: str) -> str:
    """
    Retrieve the latest release version of a GitHub repository.

    Parameters
    ----------
    repo_owner : str
        The owner of the GitHub repository.
    repo_name : str
        The name of the GitHub repository.

    Returns
    -------
    str
        The latest release version number, or None if the version cannot be retrieved.
    """

    # GitHub API URL for the latest release
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"

    try:
        # Send a GET request to the GitHub API
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        release_info = response.json()

        # Extract the version number from the tag_name
        if "tag_name" in release_info:
            version_number = release_info["tag_name"]
            print(f"Latest release version: {version_number}")
            return version_number
        else:
            print("No tag_name found in the latest release.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
