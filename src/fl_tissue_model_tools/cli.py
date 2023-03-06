import argparse
import subprocess
import sys
from glob import glob
from pathlib import Path

from fl_tissue_model_tools import defs
from fl_tissue_model_tools.configure import configure
from fl_tissue_model_tools.update_package import update_package


def main():
    commands = ['configure'] + [Path(script).stem for script in glob(str(defs.SCRIPT_DIR / '*.py'))]

    # Arguments are the command and any arguments for the command
    parser = argparse.ArgumentParser(description='Description the command-line interface')

    if len(sys.argv[1:]) > 0:
        parser.add_argument('command', type=str, choices=commands+['update'], default=None,
                            help='Command to run')

    parser.add_argument('command_args', nargs=argparse.REMAINDER,
                        help='Arguments for the command')

    args = parser.parse_args()

    args.command = args.command if len(sys.argv[1:]) > 0 else None

    if args.command is None and len(commands) == 1:
        configure()
        print(f'{defs.PKG_NAME} configured successfully.')
        return

    if args.command == 'update':
        force = False
        if len(args.command_args) == 1:
            if args.command_args[0] in ['--help', '-h']:
                print('Update the package from GitHub.')
                print('Usage: update [--help] [--force]')
                print('  --help, -h: Show this help message and exit')
                print('  --force, -f: Replace scripts and config files without confirmation')
                return
            elif args.command_args[0] in ['--force', '-f']:
                force = True
        elif len(args.command_args) > 1:
            print('Error: Too many arguments for the update command (expected between 0 and 1). '
                  'Use --help to list options.')
            return
        update_package()
        configure(replace_subdirs=True, force_replace=force)
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
        if len(args.command_args) > 1:
            print('Error: Too many arguments for the configure command (expected between 0 and 1).'
                  ' Use --help to list options.')
            print('If you entered a path with spaces, enclose the path in quotes'
                  ' or prepend each space with a backslash (\\) to escape it.')
            sys.exit(1)
        if args.command_args and args.command_args[0] in ['-h', '--help']:
            print('Usage: configure [base_dir]')
        elif args.command_args:
            configure(args.command_args[0])
        else:
            configure()
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
        command = [sys.executable, script_path, *args.command_args]
        if args.command_args == ['-h']:
            print(f'Getting options for: {args.command}')
        else:
            command_printout = ' '.join(command)
            print(f'Executing: {command_printout}')
        subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
