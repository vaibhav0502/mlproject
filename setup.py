from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This fun will return the list of requirements
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [r.replace("\n", '') for r in requirements]
    
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(
    name="mlproject",
    version='0.0.1',
    author='vaibhav',
    author_email='kumbharvaibhav007@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
