import os
from pathlib import Path
import shutil
import configparser
from platform import python_version_tuple
from setuptools import find_namespace_packages, Extension, setup

setup_cfg = configparser.ConfigParser()
setup_cfg.read('setup.cfg')

pkg_name = setup_cfg['metadata']['name']

entrypoint_commands = [
    'tissue-model-analysis-tools',
    'tmat',
    pkg_name,
    pkg_name.replace('_', '-')
]

pkg_root = Path(pkg_name).resolve()
cfg_file = pkg_root / 'package.cfg'
cwd = Path('.').resolve()

# copy scripts, model_training and config directories to package directory
project_root = cwd.parent
for dir_name in ['scripts', 'model_training', 'config']:
    src_dir = project_root / dir_name
    dest_dir = Path(pkg_name) / dir_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)

config = configparser.ConfigParser()

# Default base dir is relative to user's home directory, expanded at runtime
# However on first run, the configuration script will ask for a base dir
config['metadata'] = {
    'name': pkg_name
}

config[pkg_name] = {
    'base_dir': str(Path('~') / pkg_name)
}

with open(cfg_file, 'w', encoding='utf-8') as config_file:
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

extra_files = [str(cfg_file)]
for dir_name in ['scripts', 'model_training', 'config']:
    extra_files.extend(package_files(pkg_root / dir_name))

exclude_packages = ['pydmtgraph.notebooks', 'pydmtgraph.figures', 'pydmtgraph.data']

setup(
    name=pkg_name,
    version='0.1.0',
    author='Fogg Lab',
    # pkg_name and 'packages' without __init__.py using find_namespace_packages here:
    packages=find_namespace_packages(exclude=exclude_packages),
    package_data={pkg_name: extra_files},
    include_package_data=True,
    url='https://github.com/fogg-lab/tissue-model-analysis-tools',
    license='MIT',
    setup_requires=[
        'Cython',
        'numpy'
    ],
    ext_modules=[
        Extension(
            name="pydmtgraph.dmtgraph",
            sources=["pydmtgraph/src/pydmtgraph/dmtgraph/DMTGraph.cpp"],
            library_dirs=library_dirs,
            include_dirs=include_dirs,
            libraries=libraries,
            extra_compile_args=["--std=c++11"]
        )
    ],
    entry_points={
        'console_scripts': [f'{command}={pkg_name}.cli:main' for command in entrypoint_commands]
    }
)
