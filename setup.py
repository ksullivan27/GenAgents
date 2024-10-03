from setuptools import setup, find_packages  # Importing setup and find_packages from setuptools for packaging
import os  # Importing os for file path operations
import re  # Importing re for regular expression operations

from text_adventure_games import __version__  # Importing the version of the text_adventure_games package

# Base installation requirements for the game
base_game_install_reqs = ['jupyter', 'graphviz']

# Reading the long description from the README file
with open("README.md", "r") as fh:
    long_description = fh.read()  # Storing the content of README.md in long_description

# Reading the requirements from requirements.txt
with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), 'r') as reqs:
    # Filtering out comments and empty lines from the requirements file
    install_packages = [req for req in reqs.read().split('\n') if not re.match(r"#\s?", req) and req]
    install_packages.extend(base_game_install_reqs)  # Adding base game installation requirements to the list


setup(
    name='text adventure games',
    version=__version__,
    author='Generative Agents Framework: Samuel Thudium, Federico Cimini; Base game Framework: Chris Callison-Burch, Jms Dnns',
    author_email='sam.thudium1@gmail.com; ccb@upenn.edu',
    # description='A framework for building text based RPGs',
    description='A framework for building episodic and competitive text-based RPGs with generative agents.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://interactive-fiction-class.org/',
    # license='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_packages,
    extras_require={
        'dev': [
            'black',
            'nbformat'
        ],
    },
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
