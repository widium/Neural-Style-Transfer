# *************************************************************************** #
#                                                                              #
#    environment.py                                                            #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/05/02 15:00:12 by Widium                                    #
#    Updated: 2023/05/02 15:00:12 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import venv
from pathlib import Path

def create_virtual_env(env_name : str):
    """
    Create Virtual Environment Programmaticaly and retrun path of executable
    

    Args:
        env_name (str): name of virtual environment directory

    Returns:
        Path: path of Python executable [env_name/bin/python]
    """
    virtual_env_path = Path(env_name)

    venv.create(
        env_dir=virtual_env_path,
        with_pip=True
    )
    
    python_executable = virtual_env_path / "bin" / "python"
    
    return (python_executable)