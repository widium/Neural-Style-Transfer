# *************************************************************************** #
#                                                                              #
#    repository.py                                                             #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/05/05 13:56:54 by Widium                                    #
#    Updated: 2023/05/05 13:56:54 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from pathlib import Path
from huggingface_hub import Repository
from huggingface_hub import create_repo


# ============================================================================== #

def create_repository_on_hub(
    api_token :  str,
    namespace : str,
    repo_name : str,
    repo_type : str = "space",
    space_sdk : str = "gradio",
    space_hardware : str = "cpu-basic",
    private: bool = False,
)->str:
    """
    Create a repository on Hugging Face Hub with the given parameters.

    Args:
        api_token (str): API token for Hugging Face Hub.
        namespace (str): Namespace for the repository.
        repo_name (str): Repository name.
        repo_type (str, optional): Repository type. Defaults to "space".
        space_sdk (str, optional): Space SDK. Defaults to "gradio".
        space_hardware (str, optional): Space hardware. Defaults to "cpu-basic".
        private (bool, optional): Set repository private or public. Defaults to False.
        
    Return:
        str : Url of Hugging face space repository
    """
        
    repo_full_name = f"{namespace}/{repo_name}"
    
    repo_url = create_repo(
        repo_id=repo_full_name,
        token=api_token,
        repo_type=repo_type,
        space_sdk=space_sdk,
        space_hardware=space_hardware,
        private=private,
        exist_ok=True,
    )
    print(f"[INFO] : Repository Successfully Created in Hugging Face Hub [{repo_url}].") 
    
    return (repo_url)
    
# ============================================================================== #

def clone_repository_from_hub(
    repo_url : str,
    repo_name : str,
    api_token : str,
    destination_path : str = ".",
)->Repository:
    """
    Clone the repository from the given URL to the destination path.

    Args:
        repo_url (str): Repository URL
        repo_name (str): Repository name
        api_token (str): Hugging Face API token
        destination_path (str): Destination path for the cloned repository
        
    Return:
        Repository : repository object who represent cloned repository
    """
    destination_path = Path(destination_path)
        
    if destination_path.exists():
        repo_path = destination_path / repo_name
    else :
        repo_path = Path(".") / repo_name
            
    repository = Repository(
        local_dir=repo_path,
        clone_from=repo_url,
        use_auth_token=api_token
    )
    
    print(f"[INFO] : Repository Successfully Cloned in [{destination_path}].")
    
    return (repository)
    
    # ============================================================================== #