from setuptools import find_packages, setup
from typing import List

hyphen_e = "-e ."

def get_req(file_path:str)->List[str]:
    req = []
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        req = [re.replace("\n", "") for re in req]
    if hyphen_e in req:
        req.remove(hyphen_e)
    return req

setup(
    name = 'ml_project_with_azure',
    version = '0.0.1',
    author = 'Rishee',
    author_email = 'rishrash2712@gmail.com',
    install_requires = get_req('requirements.txt')
)
