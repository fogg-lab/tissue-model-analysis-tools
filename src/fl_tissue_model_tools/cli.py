import argparse
import subprocess
import sys
from glob import glob
from pathlib import Path

from fl_tissue_model_tools import defs
from fl_tissue_model_tools.configure import configure
from fl_tissue_model_tools.update_package import update_package

USAGE = '''Usage: tissue-model-analysis-tools [SUBCOMMAND] [OPTIONS]
Shorthand: tmat [SUBCOMMAND] [OPTIONS]

If no subcommand is given, the interactive mode will be used. For example, run: tmat

Available subcommands:
    configure: Set the location of the base directory for scripts and model training data.
    update: Update the package from GitHub.
    [SCRIPT_NAME]: Run a script from the scripts directory. Do not include the .py extension.

Get available options:
    -h, --help: Show this help message and exit.
    [SUBCOMMAND] -h: Show help (including available options) for a particular subcommand.

Examples:
    tmat configure -h
    tmat configure "C:\\Users\\Quinn\\Desktop\\some_folder_name"
    tmat update
    tmat compute_inv_depth -h
    tmat compute_zproj -v "C:\\Users\\Quinn\\input_folder_name" "C:\\Users\\Quinn\\out_folder_name"
'''

def main():
    commands = ['configure'] + [Path(script).stem for script in glob(str(defs.SCRIPT_DIR / '*.py'))]

    def print_usage_and_exit():
        print(USAGE)
        sys.exit(1)

    # for '-h' or '--help', print the usage and exit
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print_usage_and_exit()

    # Arguments are the command and any arguments for the command
    parser = argparse.ArgumentParser(description='Tissue model image analysis tools',
                                     usage=USAGE, add_help=False)

    if len(sys.argv[1:]) > 0:
        parser.add_argument('command', type=str, choices=commands+['update'], default=None)

    parser.add_argument('command_args', nargs=argparse.REMAINDER,
                        help='Arguments for the command')

    args = parser.parse_args()

    args.command = args.command if len(sys.argv[1:]) > 0 else None

    if args.command is None and len(commands) == 1:
        configure()
        print(f'{defs.PKG_NAME} configured successfully.')
        return

    def in_args(query_args, actual_args):
        # Check if any of the given args from a list are in the actual list of arguments
        for arg_name in query_args:
            if arg_name in actual_args:
                return True
        return False

    def get_flag_value(flag_name, flag_alias, args):
        # Get the value of a flag (e.g. '--local') from a list of arguments
        flag_index = args.index(flag_name) if flag_name in args else args.index(flag_alias)
        if flag_index == len(args) - 1:
            print(f'Error: No value given for {flag_name} flag.')
            sys.exit(1)
        return args[flag_index + 1]

    if args.command == 'update':
        force = False
        if in_args(['--help', '-h'], args.command_args):
            print('Update the package from GitHub.')
            print('Usage: update [--help] [--force] [--local SRC_DIR] [--branch BRANCH_NAME]')
            print('  --help, -h: Show this help message and exit.')
            print('  --force, -f: Replace scripts and config files without confirmation.')
            print('  --local, -l: Update the package from a local src directory.')
            print('  --branch, -b: Update the package from a specific branch.')
            return

        force =  in_args(['--force', '-f'], args.command_args)
        local_update = in_args(['--local', '-l'], args.command_args)
        branch_update = in_args(['--branch', '-b'], args.command_args)

        if local_update and branch_update:
            print('Error: Cannot use both --local and --branch')
            sys.exit(1)

        if local_update:
            src_dir = get_flag_value('--local', '-l', args.command_args)
            update_package(src_dir=src_dir)
        elif branch_update:
            branch_name = get_flag_value('--branch', '-b', args.command_args)
            update_package(branch_name=branch_name)
        else:
            update_package()
        configure_cmd = 'tmat configure' if force else 'tmat configure --force'
        subprocess.run(configure_cmd, shell=True, check=True)
        return

    if args.command is None:
        print('Command options:')

        for i, command in enumerate(commands):
            print(f'  {i+1}. {command}')

        command_num = input('Enter the number of the command to run (or q to quit): ')
        if command_num == '':
            print('No command entered')
            sys.exit(1)

        if command_num == 'q':
            print('Quitting')
            sys.exit(0)

        try:
            command_num = int(command_num)
        except ValueError:
            print(f'Invalid command number: \'{command_num}\'')
            sys.exit(1)

        if command_num < 1 or command_num > len(commands):
            print(f'Invalid command number: {command_num}')
            sys.exit(1)

        args.command = commands[command_num - 1]

    if not args.command_args and args.command != 'configure':
        args.command_args = input('Arguments, if any (or -h to list options): ').split()

    if args.command == 'configure':
        if in_args(['--help', '-h'], args.command_args):
            print('Set the base directory for the package.')
            print('Usage: configure [--help] [--force] [BASE_DIR]')
            print('  --help, -h: Show this help message and exit.')
            print('  --force, -f: Replace scripts and config files without confirmation.')
            print('  BASE_DIR: The base directory for the package.')
            return
        force =  in_args(['--force', '-f'], args.command_args)
        other_args = [arg for arg in args.command_args if arg not in ['--force', '-f']]
        if len(other_args) > 1:
            print('Error: Too many arguments. Run \'tmat configure --help\' for more information.')
            print('If the base directory has spaces in the path, enclose it with quotes.')
            sys.exit(1)
        target_base_dir = other_args[0] if other_args else ''
        configure(target_base_dir=target_base_dir, replace_subdirs_auto=force)
        return

    # Make sure the base directory and its subdirectories exist
    required_dirs = [
        defs.BASE_DIR,
        defs.SCRIPT_CONFIG_DIR,
        defs.SCRIPT_DIR,
        defs.MODEL_TRAINING_DIR
    ]

    for required_dir in required_dirs:
        if not required_dir.is_dir():
            print(f'{required_dir} directory not found. Running configure...')
            configure()
            for required_dir in required_dirs:  # Make sure it worked
                assert required_dir.is_dir(), f"Configuration failed: {required_dir} not found"
            print(f'Configuration complete. Proceeding with {args.command}...')
            break

    if args.command:
        # Run the command as a subprocess
        script_path = defs.SCRIPT_DIR / f'{args.command}.py'
        command = [sys.executable, str(script_path), *args.command_args]
        if args.command_args == ['-h']:
            print(f'Getting options for: {args.command}')
        else:
            command_printout = ' '.join(command)
            print(f'Executing: {command_printout}')
        subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
