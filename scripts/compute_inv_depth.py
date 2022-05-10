import sys

from fl_tissue_model_tools import script_util as su

def main():
    args = su.parse_inv_depth_args({})
    verbose = args.verbose


    ### Tidy up paths ###
    in_root = args.in_root.replace("\\", "/")
    out_root = args.out_root.replace("\\", "/")


    ### Verify input source ###
    extension = args.extension.replace(".", "")
    try:
        zstack_paths = su.inv_depth_verify_input_dir(in_root, verbose=verbose)
    except FileNotFoundError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


    ### Verify output destination ###
    try:
        su.inv_depth_verify_output_dir(out_root, verbose=verbose)
    except PermissionError as e:
        print(f"{su.SFM.failure} {e}")
        sys.exit()


if __name__ == "__main__":
    main()