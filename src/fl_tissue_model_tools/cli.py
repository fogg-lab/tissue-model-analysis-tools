import argparse
import subprocess
import sys
from glob import glob
from pathlib import Path

from fl_tissue_model_tools import configure, defs

COMMANDS = ['configure'] + [Path(script).stem for script in glob(str(defs.SCRIPT_DIR / '*.py'))]

def main():

    # Arguments are the command and any arguments for the command
    parser = argparse.ArgumentParser(description='Description the command-line interface')

    cmdline_args = sys.argv[1:]

    if len(cmdline_args) > 0:
        parser.add_argument('command', type=str, choices=COMMANDS, default=None,
                            help='Command to run')

    parser.add_argument('command_args', nargs=argparse.REMAINDER,
                        help='Arguments for the command')

    args = parser.parse_args()

    args.command = args.command if len(cmdline_args) > 0 else None

    if args.command is None:
        print('Command options:')

        for i, command in enumerate(COMMANDS):
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

        if command_num < 1 or command_num > len(COMMANDS):
            print(f'Invalid command number: {command_num}')
            sys.exit(1)

        args.command = COMMANDS[command_num - 1]

    if args.command == 'configure':
        configure.configure()
        return

    if not args.command_args:
        args.command_args = input('Arguments, if any (or -h to list options): ').split()

    # Make sure the base directory and its subdirectories exist
    required_dirs = [
        defs.BASE_DIR,
        defs.SCRIPT_CONFIG_DIR,
        defs.SCRIPT_DIR,
        defs.OUTPUT_DIR
    ]

    for required_dir in required_dirs:
        if not required_dir.exists():
            configure.configure()
            break

    if args.command:
        # Run the command as a subprocess
        script_path = defs.SCRIPT_DIR / f'{args.command}.py'
        command = [sys.executable, script_path, *args.command_args]
        if args.command_args == ['-h']:
            print(f'Getting options for: {args.command}')
        else:
            print(f'Executing: {command}')
        subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
