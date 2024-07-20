import argparse
import subprocess
import sys
from glob import glob
from pathlib import Path

from fl_tissue_model_tools import defs
from fl_tissue_model_tools.configure import configure

USAGE = """Usage: tissue-model-analysis-tools [SUBCOMMAND] [OPTIONS]
Shorthand: tmat [SUBCOMMAND] [OPTIONS]

If no subcommand is given, the interactive mode will be used. For example, run: tmat

Available subcommands:
    configure: Set the location of the base directory for scripts and model training data.
    [SCRIPT_NAME]: Run a script from the scripts directory. Do not include the .py extension.

Get available options:
    -h, --help: Show this help message and exit.
    [SUBCOMMAND] -h: Show help (including available options) for a particular subcommand.

Examples:
    tmat configure -h
    tmat configure "C:\\Users\\Quinn\\Desktop\\some_folder_name"
    tmat compute_inv_depth -h
    tmat compute_zproj "C:\\Users\\Quinn\\input_folder_name" "C:\\Users\\Quinn\\out_folder_name"
    tmat compute_branches "path/to/my/input_folder" "path/to/my/output_folder" --image-width-microns 1200
"""


def main():
    commands = ["configure"] + [
        Path(script).stem for script in glob(str(defs.SCRIPT_DIR / "*.py"))
    ]

    def print_usage_and_exit():
        print(USAGE)
        sys.exit(1)

    # for '-h' or '--help', print the usage and exit
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_usage_and_exit()

    # Arguments are the command and any arguments for the command
    parser = argparse.ArgumentParser(
        description="Tissue model image analysis tools", usage=USAGE, add_help=False
    )

    if len(sys.argv[1:]) > 0:
        parser.add_argument("command", type=str, choices=commands, default=None)

    parser.add_argument(
        "command_args", nargs=argparse.REMAINDER, help="Arguments for the command"
    )

    args = parser.parse_args()

    args.command = args.command if len(sys.argv[1:]) > 0 else None

    if args.command is None and len(commands) == 1:
        configure()
        print(f"{defs.PKG_NAME} configured successfully.")
        return

    def in_args(query_args, actual_args):
        # Check if any of the given args from a list are in the actual list of arguments
        for arg_name in query_args:
            if arg_name in actual_args:
                return True
        return False

    if args.command is None:
        print("Command options:")

        for i, command in enumerate(commands):
            print(f"  {i+1}. {command}")

        command_num = input("Enter the number of the command to run (or q to quit): ")
        if command_num == "":
            print("No command entered")
            sys.exit(1)

        if command_num == "q":
            print("Quitting")
            sys.exit(0)

        try:
            command_num = int(command_num)
        except ValueError:
            print(f"Invalid command number: '{command_num}'")
            sys.exit(1)

        if command_num < 1 or command_num > len(commands):
            print(f"Invalid command number: {command_num}")
            sys.exit(1)

        args.command = commands[command_num - 1]

    if not args.command_args and args.command != "configure":
        args.command_args = input("Arguments, if any (or -h to list options): ").split()

    if args.command == "configure":
        if in_args(["--help", "-h"], args.command_args):
            print("Set the base directory for the package.")
            print("Usage: configure [--help] [BASE_DIR]")
            print("  --help, -h: Show this help message and exit.")
            print("  BASE_DIR: The base directory for the package.")
            return
        if len(args.command_args) > 1:
            print(
                "Error: Too many arguments. Run 'tmat configure --help' for more information."
            )
            print(
                "If the base directory has spaces in the path, enclose it with quotes."
            )
            sys.exit(1)
        target_base_dir = args.command_args[0] if len(args.command_args) > 0 else ""
        configure(target_base_dir=target_base_dir)
        return

    # Make sure the base directory and its subdirectories exist
    required_dirs = [
        defs.BASE_DIR,
        defs.SCRIPT_CONFIG_DIR,
        defs.SCRIPT_DIR,
        defs.MODEL_TRAINING_DIR,
    ]

    for required_dir in required_dirs:
        if not required_dir.is_dir():
            print(f"{required_dir} directory not found. Running configure...")
            configure()
            for required_dir in required_dirs:  # Make sure it worked
                assert (
                    required_dir.is_dir()
                ), f"Configuration failed: {required_dir} not found"
            print(f"Configuration complete. Proceeding with {args.command}...")
            break

    if args.command:
        # Run the command as a subprocess
        script_path = defs.SCRIPT_DIR / f"{args.command}.py"
        command = [sys.executable, str(script_path), *args.command_args]
        if args.command_args == ["-h"]:
            print(f"Getting options for: {args.command}")
        else:
            command_printout = " ".join(command)
            print(f"Executing: {command_printout}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            sys.exit(1)


if __name__ == "__main__":
    main()
