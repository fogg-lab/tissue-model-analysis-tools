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

import argparse
from pathlib import Path
import sys
import configparser
import pkg_resources

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

def configure():
    description ='Create or move the base directory for config files, scripts, data, and output.'
    parser = argparse.ArgumentParser(description=description)

    default_base_dir = str(defs.BASE_DIR)
    if not default_base_dir:
        default_base_dir = str(USER_HOME / defs.PKG_NAME)

    parser.add_argument('--base_dir', type=str, default='', help='Base directory')

    args = parser.parse_args()

    print(f'\nEnter the preferred base directory location for {defs.PKG_NAME}.')
    print(f'The default base directory is \'{defs.PKG_NAME}\' in the current user\'s home folder.')

    if args.base_dir == '':
        args.base_dir = input(f'Base directory [{default_base_dir}]: ')
        print('')

    if args.base_dir == '':
        args.base_dir = default_base_dir

    base_dir = Path(args.base_dir)

    # Ensure that base_dir.parent exists
    if not base_dir.parent.is_dir():
        print(f'Error - Parent directory does not exist: {base_dir.parent}')
        return

    # Get previous base_dir from config file
    cfg_path = pkg_resources.resource_filename('fl_tissue_model_tools', 'fl_tissue_model_tools.cfg')
    config = configparser.ConfigParser()
    config.read(cfg_path)
    prev_base_dir = Path(config['fl_tissue_model_tools']['base_dir'])

    if base_dir.exists():
        print(f'Base directory already exists: {base_dir}')
    elif prev_base_dir.exists():
        print(f'Moving base directory from {prev_base_dir} to {base_dir}')
        try:
            prev_base_dir.rename(base_dir)
        except PermissionError:
            print(f'Cannot move directory {prev_base_dir} to {base_dir}: Permission denied')
            sys.exit(1)
    else:
        print(f'Creating base directory: {base_dir}')
        try:
            base_dir.mkdir(parents=True)
        except PermissionError:
            print(f'Cannot create directory {base_dir}: Permission denied')
            sys.exit(1)

    for subdir in BASE_DIR_SUBDIRS:
        if not (base_dir / subdir).exists():
            print(f'Creating {subdir} directory')
            (base_dir / subdir).mkdir(parents=True)

    # Copy config files and scripts to base_dir
    for src_dir, dest_dir, filenames in [(defs.PKG_CONFIG_DIR, CFG_SUBDIR, CFG_FILES),
                                         (defs.PKG_SCRIPTS_DIR, SCRIPTS_SUBDIR, SCRIPTS)]:
        for filename in filenames:
            src_path = src_dir / filename
            dest_path = base_dir / dest_dir / filename
            if not dest_path.exists():
                print(f'Copying {filename} to {dest_path}')
                dest_path.write_bytes(src_path.read_bytes())

    # Keep the base directory relative to current user's home directory
    args.base_dir = args.base_dir.replace(str(USER_HOME), '~')

    config.set('fl_tissue_model_tools', 'base_dir', args.base_dir)
    with open(cfg_path, 'w', encoding='utf-8') as config_file:
        config.write(config_file)

    print(f'\nThe base directory for {defs.PKG_NAME} is now: {base_dir}')

    if str(USER_HOME) in str(base_dir):
        print(f'Note: For other users, \"{str(USER_HOME)}\" will be replaced with ' +
              'that user\'s home directory.')
