# *************************************************************************** #
#                                                                              #
#    hugging_face.py                                                           #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/25 07:53:57 by Widium                                    #
#    Updated: 2023/04/25 07:53:57 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo

from .hub import create_repository_on_hub
from .hub import clone_repository_from_hub

from .space_manage import setup_space_repository

# ============================================================================== #

class HuggingFaceRepositoryCreator:
    """
    Create Hugging Face repository in hub
    Clone inside local folder
    Setup it with basic file structure
    Push it to hugging face hub 
    
    Initialize Creator with write token
    """
# ============================================================================== #

    def __init__(self, api_token: str):
        """
        Initialize HuggingFaceRepositoryCreator with an API token.
        
        Args:
            api_token (str): Hugging Face API token
        """
        self.api = HfApi()
        self.api_token = api_token
        
        user = self.api.whoami(token=api_token)
        print(f"[INFO] : Successfully Logged in Hugging Face Hub.")
         
        self.namespace = user['name']
    
    # ============================================================================== #
      
    def create_repository(
        self, 
        repo_name : str,
        app_filepath : str = None,
        destination_path : str = ".",
        repo_type : str = "space",
        space_sdk : str = "gradio",
        python_version : str = "3.8.9",
        space_hardware : str = "cpu-basic",
        private: bool = False,
    )-> Repository:
        """
        Create a Hugging Face Hub repository, clone it, and set up its initial structure.
        
        Args:
            `repo_name` (str): Repository name.
            `app_filepath` (str, optional): Path to the app file.
            `destination_path` (str, optional): Local destination path for the cloned repository. Defaults to ".".
            `repo_type` (str, optional): Repository type. Defaults to "space".
            `space_sdk` (str, optional): Space SDK. Defaults to "gradio".
            `python_version` (str, optional): Python version for the space. Defaults to "3.8.9".
            `space_hardware` (str, optional): Space hardware. Defaults to "cpu-basic".
            `private` (bool, optional): Set repository private or public. Defaults to False.
        
        Returns:
            `Repository`: Created repository object.
        """
        repo_url = create_repository_on_hub(
            api_token=self.api_token,
            namespace=self.namespace,
            repo_name=repo_name,
            repo_type=repo_type,
            space_sdk=space_sdk,
            space_hardware=space_hardware,
            private=private,
        )
        
        repository = clone_repository_from_hub(
            repo_url=repo_url, 
            repo_name=repo_name,
            api_token=self.api_token, 
            destination_path=destination_path
        )
        
        setup_space_repository(
            repository=repository,
            app_filepath=app_filepath,
            python_version=python_version,
        )
        
        return (repository)
    
    # ============================================================================== #