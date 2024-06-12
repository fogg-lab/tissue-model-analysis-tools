import argparse
from gooey import Gooey, GooeyParser, local_resource_path
from fl_tissue_model_tools.scripts import (
    compute_branches,
    compute_cell_area,
    compute_inv_depth,
    compute_zproj,
)


@Gooey(
    program_name="Tissue Model Analysis Tools",
    image_dir=local_resource_path("."),
    navigation="TABBED",
)
def main():
    parser = GooeyParser(description="Tissue model analysis tools")
    subparsers = parser.add_subparsers(dest="command")

    compute_branches_parser = subparsers.add_parser("Analyze ​Microvessels")
    compute_zproj_parser = subparsers.add_parser("Z ​Project")
    compute_cell_area_parser = subparsers.add_parser("Estimate ​Cell ​Coverage ​Area")
    compute_inv_depth_parser = subparsers.add_parser("Predict Depth ​of ​Invasion")

    for subparser in [
        compute_branches_parser,
        compute_cell_area_parser,
        compute_inv_depth_parser,
        compute_zproj_parser,
    ]:
        subparser.add_argument(
            "in_root",
            type=str,
            help=(
                "Full path to root directory of input images. "
                "Ex: [...]/my_data/images/experiment_1_yyyy_mm_dd/"
            ),
            widget="DirChooser",
        )

        subparser.add_argument(
            "out_root",
            type=str,
            help=(
                "Full path to root directory to save output "
                "(HINT: choose or create an empty directory for this)."
            ),
            widget="DirChooser",
        )

        subparser.add_argument(
            "-C",
            "--channel",
            type=int,
            default=None,
            help=(
                "Index of color channel (starting from 0) to read from images. "
                "If no argument is supplied, images must be single channel."
            ),
        )

        subparser.add_argument(
            "-T",
            "--time",
            type=int,
            default=None,
            help=(
                "Index of time (starting from 0) to read from images. "
                "If no argument is supplied, images must not be time series."
            ),
        )

    compute_branches_parser.add_argument(
        "--image-width-microns",
        type=float,
        default=1000,
        help=(
            "Physical width in microns of the region captured by each image. "
            "For instance, if 1 pixel in the image corresponds to 0.8 microns, "
            "this value should equal to 0.8x the horizontal resolution of the image. "
        ),
    )

    compute_branches_parser.add_argument(
        "--graph-thresh-1",
        nargs="+",
        type=float,
        default=5,
        help=(
            "This threshold controls how much of the morse graph is used to compute the number of branches. "
            "Lower values include more of the graph, and more branches are detected. "
            "Higher values include less of the graph, and fewer branches are detected."
            "You can provide multiple values (separated by space characters) to test multiple thresholds."
        ),
    )

    compute_branches_parser.add_argument(
        "--graph-thresh-2",
        nargs="+",
        type=float,
        default=10,
        help=(
            "This is the threshold for connecting branches, e.g. where it is "
            "ambiguous whether two branches are part of the same component. Lower "
            "values result in more connected branches, and higher values result in "
            "more disconnections.\n"
            "You can provide multiple values (separated by space characters) to test multiple thresholds."
        ),
    )

    compute_branches_parser.add_argument(
        "--min-branch-length",
        type=float,
        default=12,
        help=("The minimum branch length (in microns) to consider."),
    )

    compute_branches_parser.add_argument(
        "--max-branch-length",
        type=float,
        default=None,
        help=(
            "This is the maximum branch length (in microns) to consider. By default, "
            "this parameter is not included. If it is not specified, no maximum branch "
            "will be enforced."
        ),
    )

    compute_branches_parser.add_argument(
        "--remove-isolated-branches",
        action="store_true",
        help=(
            "Whether to remove branches that are not connected to any other branches "
            "after the network is trimmed per the branch length constraints "
            "(enforcing minimum and maximum branch lengths might isolate some "
            "branches, which may or may not be desired)."
        ),
    )

    compute_branches_parser.add_argument(
        "--graph-smoothing-window",
        type=float,
        default=12,
        help=("This is the window size (in microns) for smoothing the branch paths."),
    )

    compute_branches_parser.add_argument(
        "-w",
        "--detect-well",
        action="store_true",
        help="Auto detect the well boundary and exclude regions outside the well.",
    )

    compute_cell_area_parser.add_argument(
        "-w",
        "--detect-well",
        action="store_true",
        help="Auto detect the well boundary and exclude regions outside the well.",
    )

    compute_zproj_parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="max",
        choices=["min", "max", "med", "avg", "fs"],
        help=(
            "Z projection method. Defaults to 'max'.\n"
            "min = Minimum intensity projection\n"
            "max = Maximum intensity projection\n"
            "med = Median intensity projection\n"
            "avg = Average intensity projection\n"
            "fs = Focus stacking"
        ),
    )

    args = parser.parse_args()

    if args.command == "Analyze ​Microvessels":
        compute_branches.main(args)
    elif args.command == "Z ​Project":
        compute_zproj.main(args)
    elif args.command == "Estimate Cell​ Coverage​ Area":
        compute_cell_area.main(args)
    elif args.command == "Predict ​Depth ​of ​Invasion":
        compute_inv_depth.main(args)


if __name__ == "__main__":
    main()
