# *************************************************************************** #
#                                                                              #
#    repository.py                                                             #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/19 11:09:47 by Widium                                    #
#    Updated: 2023/04/19 11:09:47 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import subprocess
from pathlib import Path
from urllib.parse import urlparse

def clone_repository(
    repository_url : str,
    destination_folder: str = "Project/src/",
):
    """
    Clone git repository inside a destination folder

    Args:
        repository_url (str): https url of the repository
        destination_folder (str, optional): destination folder. Defaults to "Project/src/".

    Raises:
        FileNotFoundError: if the destination folder doesn't exist
        FileExistsError: if the repository already exist in destination

    Returns:
        Path: path of cloned repository
    """
    destination_folder = Path(destination_folder)
    url_path = urlparse(repository_url).path
    repository_name = Path(url_path).stem
    
    # Ensure the destination folder exists
    if not destination_folder.exists():
        raise FileNotFoundError(f"destination_folder doesn't exist : [{destination_folder}]")

    # Create a new folder for the repository inside the destination folder
    repository_dest_path = destination_folder / repository_name

    # Check if the new folder is empty
    if repository_dest_path.exists():
        raise FileExistsError(f"{repository_dest_path} already exists. Please remove it or choose a different location.")

    # Clone the repository into the new folder
    subprocess.run(["git", "clone", repository_url, str(repository_dest_path)])
    
    return (repository_dest_path)
