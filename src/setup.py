import os
from pathlib import Path
import configparser
from setuptools import setup, Extension
import numpy as np

SETUP_CFG = configparser.ConfigParser()
SETUP_CFG.read(Path(__file__).resolve().parent / 'setup.cfg')
PKG_NAME = SETUP_CFG['metadata']['name']
ENTRYPOINT_FUNCTION = f'{PKG_NAME}.cli:main'
ENTRYPOINT_COMMANDS = [
    'tissue-model-analysis-tools',
    'tmat',
    PKG_NAME,
    PKG_NAME.replace('_', '-')
]
CFG_FILE = Path(__file__).resolve().parent / PKG_NAME / 'package.cfg'

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

setup(
    name=PKG_NAME,
    version='0.1.0',
    author='Fogg Lab',
    packages=[PKG_NAME],
    package_data={'': [CFG_FILE]},
    include_package_data=True,
    url='https://github.com/fogg-lab/tissue-model-analysis-tools',
    license='MIT',
    setup_requires=[
        'Cython',
        'numpy'
    ],
    ext_modules=[Extension(
        name=f'{PKG_NAME}.gwdt_impl',
        sources=[os.path.join(PKG_NAME, 'gwdt_impl.pyx')],
        include_dirs=[np.get_include()],
        language='c++'
    )],
    entry_points={
        'console_scripts': [f'{command}={ENTRYPOINT_FUNCTION}' for command in ENTRYPOINT_COMMANDS]
    }
)
