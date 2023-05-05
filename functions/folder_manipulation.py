# *************************************************************************** #
#                                                                              #
#    folder_manipulation.py                                                    #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/19 11:09:44 by Widium                                    #
#    Updated: 2023/04/19 11:09:44 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import shutil
from pathlib import Path

def extract_folder(parent_folder : str, target_folder_name : str, destination : str):
    """
    Extract Folder and his content to other location and remove the parent folder

    Args:
        `parent_folder` (str): parent folder of target folder
        `target_folder_name` (str): target folder we want to extract
        `destination` (str): destination of target folder extracted
    
    Return :
        new path of target folder
    """
    parent_folder = Path(parent_folder)
    destination = Path(destination)
    
    folders = [
        item.name 
        for item in parent_folder.iterdir() 
        if item.is_dir()
    ]
    
    for name in folders:
        
        if name == target_folder_name:
            target_folder_path = parent_folder / name
            shutil.move(src=str(target_folder_path), dst=str(destination))
            print(f"[INFO] : Moove [{target_folder_path}] to [{destination / target_folder_name}]")
    
    shutil.rmtree(str(parent_folder))
    print(f"[INFO] : Remove [{parent_folder}]")
    
    return (str(destination / target_folder_name))