# *************************************************************************** #
#                                                                              #
#    file_ops.py                                                               #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/05/05 14:01:27 by Widium                                    #
#    Updated: 2023/05/05 14:01:27 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import shutil

from pathlib import Path

# ============================================================================== #

def create_simple_app_file(repository_path : Path):
    """
    Create simple app.py file with Gradio code for building the app in Hugging Face space hub

    Args:
        repository_path (Path): repository path 
    """

    app_filepath = repository_path / "app.py"
    app_filepath.touch()
    
    with app_filepath.open("w") as file :
        app_content = "import gradio as gr\n"
        app_content += "\ndef greet(name):\n"
        app_content += "    return 'Hello ' + name + '!!'\n"
        app_content += "iface = gr.Interface(fn=greet, inputs='text', outputs='text')\n"
        app_content += "iface.launch()\n"
    
        file.write(app_content)
        
    print(f"[INFO] : App file not found -> create simple app file here [{app_filepath}]")
    
# ============================================================================== #

def duplicate_app_file(app_filepath : str, repository_path : Path):
    """
    Duplicate python app file with gradio code to the repository path

    Args:
        app_filepath (str): filepath of python app file
        repository_path (Path): repository path
    """
    new_app_filepath = repository_path / f"app.py"
    
    app_filepath = shutil.copy2(
        src=app_filepath, 
        dst=new_app_filepath,
    )
    print(f"[INFO] : App file found -> Move app file here [{app_filepath}]")

# ============================================================================== #