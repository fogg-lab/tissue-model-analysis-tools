'''
Create or move the base directory to store config files, custom scripts, data, and output.
Creates the following directory structure:
    fl_tissue_model_tools/
        config/
            default_branching_computation.json
            default_cell_area_computation.json
            default_invasion_depth_computation.json
        model_training/
        output/
        scripts/
            compute_branches.py
            compute_cell_area.py
            compute_inv_depth.py
            compute_zproj.py
'''

from pathlib import Path
import sys
import configparser

from fl_tissue_model_tools import defs

CFG_FILES = [
    'default_branching_computation.json',
    'default_cell_area_computation.json',
    'default_invasion_depth_computation.json'
]

SCRIPTS = [
    'compute_branches.py',
    'compute_cell_area.py',
    'compute_inv_depth.py',
    'compute_zproj.py'
]

CFG_SUBDIR = 'config'
SCRIPTS_SUBDIR = 'scripts'

BASE_DIR_SUBDIRS = [
    CFG_SUBDIR,
    SCRIPTS_SUBDIR,
    'model_training',
    'output'
]

USER_HOME = Path.home().resolve()

def configure(base_dir: str='', print_help: bool=False):
    '''Create or move the base directory for config files, scripts, data, and output.'''

    if print_help:  # print help message and exit
        print(f'Usage: Pass in the preferred base directory location for {defs.PKG_NAME}.')
        print('If no base directory is specified, you will be prompted to enter one ' +
              'or accept the default.')
        sys.exit(0)

    default_base_dir = str(defs.BASE_DIR)
    if not default_base_dir:
        default_base_dir = str(USER_HOME / defs.PKG_NAME)

    print(f'\nEnter the preferred base directory location for {defs.PKG_NAME}.')
    print(f'The default base directory is \'{defs.PKG_NAME}\' in the current user\'s home folder.')

    if base_dir == '':
        base_dir = input(f'Base directory [{default_base_dir}]: ') or default_base_dir
        print('')

    prev_base_dir = str(defs.BASE_DIR)

    # Ensure that base_dir.parent exists
    if not Path(base_dir).parent.is_dir():
        print(f'Error - Parent directory does not exist: {base_dir.parent}')
        return

    if Path(base_dir).exists():
        print(f'Base directory already exists: {base_dir}')
    elif Path(prev_base_dir).exists():
        print(f'Moving base directory from {prev_base_dir} to {base_dir}')
        try:
            Path(prev_base_dir).rename(base_dir)
        except PermissionError:
            print(f'Cannot move directory {prev_base_dir} to {base_dir}: Permission denied')
            sys.exit(1)
    else:
        print(f'Creating base directory: {base_dir}')
        try:
            Path(base_dir).mkdir(parents=True)
        except PermissionError:
            print(f'Cannot create directory {base_dir}: Permission denied')
            sys.exit(1)

    for subdir in BASE_DIR_SUBDIRS:
        subdir_path = Path(base_dir) / subdir
        if not subdir_path.exists():
            print(f'Creating {subdir} directory')
            subdir_path.mkdir(parents=True)

    # Copy config files and scripts to base_dir
    for src_dir, dest_dir, filenames in [(defs.PKG_CONFIG_DIR, CFG_SUBDIR, CFG_FILES),
                                         (defs.PKG_SCRIPTS_DIR, SCRIPTS_SUBDIR, SCRIPTS)]:
        for filename in filenames:
            src_path = src_dir / filename
            dest_path = Path(base_dir) / dest_dir / filename
            if not dest_path.exists():
                print(f'Copying {filename} to {dest_path}')
                dest_path.write_bytes(src_path.read_bytes())

    if base_dir != prev_base_dir:
        # Update package config file with new base_dir

        config = configparser.ConfigParser()
        config.read(defs.PKG_CFG_PATH)

        # Keep the base directory relative to current user's home directory
        config.set(defs.PKG_NAME, 'base_dir', base_dir.replace(str(USER_HOME), '~'))
        with open(defs.PKG_CFG_PATH, 'w', encoding='utf-8') as config_file:
            config.write(config_file)

    print(f'\nThe base directory for {defs.PKG_NAME} is now: {base_dir}')

    if str(USER_HOME) in base_dir and str(USER_HOME) not in str(defs.PKG_BASE_DIR):
        # Make note about multiple users if package is installed system-wide
        print(f'Note: For other users, \"{USER_HOME}\" will be replaced with their home directory.')
