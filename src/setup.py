import os
from pathlib import Path
import shutil
import configparser
from setuptools import setup, Extension
from platform import python_version_tuple
import numpy as np
from glob import glob

SETUP_CFG = configparser.ConfigParser()
SETUP_CFG.read('setup.cfg')
PKG_NAME = SETUP_CFG['metadata']['name']
ENTRYPOINT_FUNCTION = f'{PKG_NAME}.cli:main'
ENTRYPOINT_COMMANDS = [
    'tissue-model-analysis-tools',
    'tmat',
    PKG_NAME,
    PKG_NAME.replace('_', '-')
]

PROJECT_ROOT = Path('.').resolve().parent
PKG_ROOT = Path(PKG_NAME).resolve()
CFG_FILE = PKG_ROOT / 'package.cfg'

# copy scripts, model_training and config directories to package directory
for dir_name in ['scripts', 'model_training', 'config']:
    src_dir = PROJECT_ROOT / dir_name
    dest_dir = Path(PKG_NAME) / dir_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)

config = configparser.ConfigParser()

# Default base dir is relative to user's home directory, expanded at runtime
# However on first run, the configuration script will ask for a base dir
config['metadata'] = {
    'name': PKG_NAME
}

config[PKG_NAME] = {
    'base_dir': str(Path('~') / PKG_NAME)
}

with open(CFG_FILE, 'w', encoding='utf-8') as config_file:
    config.write(config_file)

# Boost libraries
py_version = python_version_tuple()
libraries = [f"boost_python{py_version[0]}{py_version[1]}",
             f"boost_numpy{py_version[0]}{py_version[1]}"]

# Add lib and include dirs if boost is installed in a conda environment
env_path = os.environ.get("CONDA_PREFIX")
if env_path:
    env_library_paths = (env_path, os.path.join(env_path, "Library"))
    library_dirs = [os.path.join(p, "lib") for p in env_library_paths]
    include_dirs = [os.path.join(p, "include") for p in env_library_paths]
else:
    library_dirs = None
    include_dirs = None


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = [str(CFG_FILE)]
for dir_name in ['scripts', 'model_training', 'config']:
    extra_files.extend(package_files(PKG_ROOT / dir_name))

setup(
    name=PKG_NAME,
    version='0.1.0',
    author='Fogg Lab',
    packages=[PKG_NAME],
    package_data={PKG_NAME: extra_files},
    include_package_data=True,
    url='https://github.com/fogg-lab/tissue-model-analysis-tools',
    license='MIT',
    setup_requires=[
        'Cython',
        'numpy'
    ],
    ext_modules=[
        Extension(
            name=f'{PKG_NAME}.gwdt_impl',
            sources=[os.path.join(PKG_NAME, 'gwdt_impl.pyx')],
            include_dirs=[np.get_include()],
            language='c++'
        ),
        Extension(
            name="pydmtgraph.dmtgraph",
            sources=["pydmtgraph/src/pydmtgraph/dmtgraph/DMTGraph.cpp"],
            library_dirs=library_dirs,
            include_dirs=include_dirs,
            extra_compile_args=["--std=c++11"]
        )
    ],
    entry_points={
        'console_scripts': [f'{command}={ENTRYPOINT_FUNCTION}' for command in ENTRYPOINT_COMMANDS]
    }
)
