import os

from ..constants import PACKAGEDIR
from .base import is_software_installed, run_bat_file


def anoconda_installer() -> None:
    """
    Install Anaconda if it is not already installed.
    """

    detection = is_software_installed("conda.exe")
    if detection:
        pass
    else:
        conda_installer_path = os.path.join(PACKAGEDIR, "bat", "pre-installer.bat")
        run_bat_file(conda_installer_path)
