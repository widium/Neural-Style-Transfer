# *************************************************************************** #
#                                                                              #
#    package.py                                                                #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/05/05 15:08:55 by Widium                                    #
#    Updated: 2023/05/05 15:08:55 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from repository import clone_repository
from folder_manipulation import extract_folder

def import_package(destination_folder : str = "Project/src/"):
    
    repository = {
        # "repository url" : "package name"
        "https://github.com/widium/Pytorch-Model-Archiver.git" : "archive",
        "https://github.com/widium/Pytorch-Training-Toolkit.git" : "training",
        "https://github.com/widium/Pytorch_Experiment_Framework.git" : "saver",
        "https://github.com/widium/MlOps-Toolkit.git" : "deployment",
    }
    
    for repos_url, package_name in repository.items():
        
        repos_path = clone_repository(
            repository_url=repos_url,
            destination_folder=destination_folder,
        )
        
        # extract package inside repos and remove repository folder 
        extract_folder(
            parent_folder=repos_path,
            target_folder_name=package_name,
            destination=destination_folder,
        )
        
        