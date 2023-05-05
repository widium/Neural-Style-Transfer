# *************************************************************************** #
#                                                                              #
#    file_manage.py                                                            #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/05/05 13:56:30 by Widium                                    #
#    Updated: 2023/05/05 13:56:30 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from pathlib import Path
from huggingface_hub import Repository

from .app_file_manage import create_simple_app_file, duplicate_app_file

# ============================================================================== #

def define_python_version_in_readme(
    readme_path : Path,
    python_version : str = "3.8.0", 
)->None:
    """
    Insert Line inside config line in top of README.md file of Hugging Face Space

    Args:
        readme_path (Path): Path of readme file inside repository
        python_version (str, optional): python version to setup. Defaults to "3.8.0".

    Raises:
        FileNotFoundError: if readme file doesn't exist
    """
    
    python_version_config = f"python_version: {python_version}\n"
    if readme_path.exists():
        
        # Extract content of file in list of lines
        with readme_path.open("r") as file:
            content = file.readlines()
        
        # add line in specific index
        for index, line in enumerate(content):
            if (line.startswith("title")): 
                content.insert(index, python_version_config) 
                break
        
        # Rewrite the file
        with readme_path.open("w") as file:
            print(f"[INFO] : Setup the Python version of Space : [python : {python_version}]")
            file.writelines(content)
            
    else :
        raise FileNotFoundError(f"[ERROR] : file [{readme_path}] doesn't exist...")


# ============================================================================== #
     
def setup_space_repository(
    repository : Repository,
    python_version : str,
    app_filepath : str = None,
)->None:
    """
    Set up the repository with initial files and folder structure.
    
    Args:
        repository (Repository): Repository object for the local cloned repository
        python_version (str): Python version to use for the Space
        app_filepath (str, optional): Filepath of the Python app file, if None, a simple app file will be created
    """
    repository_path = Path(repository.local_dir)
        
    examples_path = repository_path / "examples"
    requirement_path = repository_path / "requirements.txt"
    readme_path = repository_path / "README.md"

    examples_path.mkdir(parents=True, exist_ok=True)
    requirement_path.touch()
    
    define_python_version_in_readme(
        readme_path=readme_path,
        python_version=python_version,
    )
    
    if app_filepath == None:
        create_simple_app_file(repository_path=repository_path)
    else :
        duplicate_app_file(
            app_filepath=app_filepath,
            repository_path=repository_path
        )
    
    print(f"[INFO] : Create [{requirement_path} File and {examples_path} Directory].")
    
    repository.git_add()
    repository.git_commit(commit_message="Initialize Repository App")
    repository.git_push()
    
    print(f"[INFO] : First Commit Successfully Initialized.")      