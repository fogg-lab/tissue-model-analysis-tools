# Fogg Lab Tissue Model Analysis Tools

A command-line application for automated high-throughput analysis of cancer and endothelial cell dynamics in hydrogels.

**Try the interactive demo notebook in Google Colab**

<a target="_blank" href="https://colab.research.google.com/github/fogg-lab/tissue-model-analysis-tools/blob/main/notebooks/analysis_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Table of Contents
**[Setup](#setup)**<br>
**[Capabilities](#capabilities)**<br>
**[Usage](#usage)**<br>

## Setup

Windows, MacOS, and Linux are all supported for regular installation. Installation with CUDA (GPU acceleration) requires Linux or Windows Subsystem for Linux (WSL), Python **3.10**, and an NVidia CUDA-capable GPU.

### Prerequisite: Install Python and pipx

_Note: As an alternative option to using `pipx`, you could install Tissue Model Analysis Tools in a Conda environment. Otherwise, follow the instructions below._

**1**. Install a version of Python in the range **>=3.9,<3.12** (such as [Python 3.11.9](https://www.python.org/downloads/release/python-3119)). Confirm that the correct Python version was installed by running each of these commands in a terminal or command prompt window, and find out which command is recognized (depending on your system configuration, it could be installed as either `python`, `python3`, or `py`, or something version-specific such as `python3.10`):
```bash
python --version
```
```bash
python3 --version
```
```bash
py --version
```
**2**. In your terminal or command prompt window, install [pipx](https://github.com/pypa/pipx). To do so, run the two commands below that correspond to your Python installation (starting with either `python -m ...`, `python3 -m ...`, or `py -m ...`):
```bash
python -m pip install --user pipx
python -m pipx ensurepath
```
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```
```bash
py -m pip install --user pipx
py -m pipx ensurepath
```

After you run these commands, close the terminal or command prompt window. `pipx` will be available the next time you open a terminal or command prompt window, and you can proceed with the setup.

### Setup

Run the following commands in a terminal or command prompt window.

**1**. Install `fl_tissue_model_tools` command-line utility, `tmat` for short (est. time 5 minutes):

**Note**: You need to use the "Regular Installation" option unless you have an NVidia GPU and are running in Linux or WSL.

**Regular Installation**
```bash
pipx install git+https://github.com/fogg-lab/tissue-model-analysis-tools.git#egg=fl_tissue_model_tools
```
**Installation with CUDA (GPU Acceleration)**
```bash
pipx install 'git+https://github.com/fogg-lab/tissue-model-analysis-tools.git#egg=fl_tissue_model_tools[and-cuda]'
```
**2**. Configure base directory to store data, scripts, and script configuration files:
```bash
tmat configure
```
**3**. Note that commands will follow this layout (more details in [usage](#usage)):
```bash
tmat [SUBCOMMAND] [OPTIONS]
```

Or you can use the interactive mode:
```bash
tmat
```

---

#### Uninstall `fl_tissue_model_tools` package
Execute
```bash
pipx uninstall fl_tissue_model_tools
```

---

#### Update `fl_tissue_model_tools` package
To update `tmat`, just reinstall it with the `--force` flag:
```bash
pipx install git+https://github.com/fogg-lab/tissue-model-analysis-tools.git@#egg=fl_tissue_model_tools --force
tmat configure
```

## Capabilities

For a detailed description of analysis capabilities, see the [capabilities overview notebook](notebooks/capabilities_overview.ipynb).

## Usage

### Tools
The `tmat` command-line utility consists of four separate automated image analysis tools:
* Z projection of image Z stacks. The input is a directory of Z stacks. The output is a directory of Z projections.
* Cell coverage area computation. The input is a directory of images (for instance, Z projections). The output is a CSV file and a directory of binary masks, one mask per image, to visually show what was detected as cells.
* Invasion depth computation (of Z stacks). The input is a directory of Z stacks. The output is a CSV file containing invasion predictions for each Z position in each Z stack.
* Quantify microvessel formation (number of branches, lengths of branches). The input is a directory of images (for instance, Z projections). The output is a CSV file containing the total number of branches, total branch length, and average branch length. Additionally, this tool outputs a directory of intermediate outputs, which are all visualizations that you can use to confirm the validity of the analysis. These visualizations can also help you tweak the configuration parameters and run the tool again if it doesn't do a very good job.

To use `tmat`, open a terminal or command prompt window and execute commands in the following format:
```bash
# For non-interactive use, specify all arguments in a single command
tmat [command_script] [-flags] [arguments]
```

For interactive use, just execute `tmat`.
```bash
# Interactive mode can be useful if you forget what command line arguments are available
tmat
```

For input data paths on Windows, it is usually easiest to copy the path from the file explorer search bar.

For a description of all parameters that each commandline tool accepts, execute one of the following:
```bash
tmat [command_script] -h
```
```bash
# or
tmat [command_script] --help
```
```bash
# or (get help at the interactive prompt)
tmat
```

#### Cell Area
**Basic usage (accept the default configuration)**
```bash
tmat compute_cell_area "/path/to/input/folder" "/path/to/output/folder"
```

Here, `/path/to/input/folder` is the full path to a directory of images which will be analyzed.

If your images are not cropped to the region inside the well, you can have the script automatically detect the well region by adding the `--detect-well` flag (or `-w` for short). For instance, if your wells are circular and you add the `--detect-well` flag, the script will detect and mask out the region outside of this circular well. Also works for "squircle" shaped (i.e. square with rounded corners) wells. Example usage:
```bash
tmat compute_cell_area --detect-well "/path/to/input/folder" "/path/to/output/folder"
```

**Custom usage (customize the analysis configuration)**

* Create custom configuration `.json` file, using `config/default_cell_area_computation.json` as a template. The following parameters can be customized:
    * `dsamp_size` (int): Size that input images will be downsampled to for analysis. Smaller sizes mean faster, less accurate analysis. Default is 512, meaning the image will be downscaled so that the maximum dimension is 512 (e.g., 1000x1500 is downsampled to 341x512).
    * `sd_coef` (float): Strictness of thresholding. Positive numbers are more strict, negative numbers are less strict. This is a multiplier of the foreground pixel standard deviation, so values in the range (-2, 2) are the most reasonable.
    * `rs_seed` (integer): A random seed for the algorithm. Allows for reproducability since the Gaussian curves are randomly initialized. Default is 0.
    * `batch_size` (integer): Number of images to process at once. Larger numbers are faster but require more memory. Default is 4.

**Run with custom configuration file:**
```bash
tmat compute_cell_area --config "/path/to/config/file.json" "/path/to/input/folder" "/path/to/output/folder"
```

#### Z Projection
**Basic usage (accept the default configuration)**
```bash
tmat compute_zproj "/path/to/input/directory" "/path/to/output/folder"
```
Here, "/path/to/input/directory" is the full path to a directory of Z stacks. Each Z stack should either be an OME-TIFF format file, or subdirectory of individual TIFF format files. 

If you create subdirectories of indidual TIFFs for each Z stack, you need to assign filenames to the images containing the following pattern: `...Z[pos]_...` to indicate the Z position for each image. For example `...Z01_...` denotes Z position 1 for a given image (or you can start at 0 if you like).

For N Z stacks, the input directory structure would be:
```
Root directory
|
|----Z Stack 1 subdirectory
|    |    Z position 1 image
|    |    Z position 2 image
|    |    ...
|
|----Z Stack 2 subdirectory
|    |    Z position 1 image
|    |    Z position 2 image
|    |    ...
|    ...
|
|----Z Stack N subdirectory
|    |    Z position 1 image
|    |    Z position 2 image
|    |    ...
```

**To compute Z-projections and their cell area, add the --area flag:**
```bash
tmat compute_zproj --area "/path/to/input/folder" "/path/to/output/folder"
```

* Use the `--method` flag to select custom Z projection method, from:
    * Minimum: Minimum intensity projection, use `--method min`
    * Maximum (**default**): Maximum intensity projection, use `--method max`
    * Median: Median intensity projection, use `--method med`
    * Average: Average intensity projection, use `--method avg`
    * Focus Stacking: Focus stacking projection, use `--method fs`.

**Example: Compute Z projections and cell coverage area with the focus stacking method**
```bash
tmat compute_zproj --area --method fs "/path/to/input/folder" "/path/to/output/folder"
```

See [Capabilities](#capabilities) for details.

#### Invasion Depth
**Usage**
```
tmat compute_inv_depth "/path/to/input/folder" "/path/to/output/folder"
```

For a description of the input directory structure, see [Z Projection](#z-projection).

#### Branches (quantify vessel formation)
**Basic usage (accept the default configuration)**
```bash
tmat compute_branches "/path/to/input/folder" "/path/to/output/folder"
```

Here, `/path/to/input/folder` is the full path to a directory of images which will be analyzed.

If your images are not cropped to the region inside the well, you can have the script automatically detect the well region by adding the `--detect-well` flag (or `-w` for short). For instance, if your wells are circular and you add the `--detect-well` flag, the script will detect and mask out the region outside of this circular well. Also works for "squircle" shaped (i.e. square with rounded corners; lens) wells. Example usage:
```bash
tmat compute_branches --detect-well "/path/to/input/folder" "/path/to/output/folder"
```

**Custom usage (customize the analysis configuration)**

Customize configuration variables (you can edit `config/default_branching_computation.json` in your base directory, or refer to `src/fl_tissue_model_tools/config` in this repository):

- `image_width_microns` (float): Physical width in microns of the region captured by each image. For instance, if 1 pixel in the image corresponds to 0.8 microns, this value should equal to 0.8x the horizontal resolution of the image. The default is 1000.0. **Important**: Make sure that the value for `image_width_microns` is set to the correct value for your images. If it is not, the branch lengths and other length-based configuration parameters will be not be accurate.
- `model_cfg_path` (string): Optional. This is the path to the configuration file of the segmentation model. This parameter is not included in the default configuration file. If it is not specified, the latest pretrained model in the `model_training` folder will be used.
- `vessel_probability_thresh` (float): May require some experimentation to find the best value for your data. This is the segmentation probability threshold above which a pixel will be classified as part of a microvessel. The default is 0.5. Lower values such as 0.1 may work well if you want to detect vessels that are visibly faint, dim, or have dark spots along them. Higher values such as 0.9 may work well if the script is detecting too many objects as vessels. *Tunable*\*
- `graph_thresh_1` (float): May require some experimentation to find the best value for your data. This threshold controls how much of the morse graph is used to compute the number of branches. Lower values include more of the graph, and more branches are detected. Higher values include less of the graph, and fewer branches are detected. The default is 2. If the default value does not work well, try different values like 0.25, 0.5, 1, 2, 4, etc. up to around 64. *Tunable*\*
- `graph_thresh_2` (float): Also could use some tuning. This is the threshold for connecting branches, e.g. where it is ambiguous whether two branches are part of the same component. Lower values result in more connected branches, and higher values result in more disconnections. The default is 4. If the default value does not work well, try values like 0.0, 0.25, 0.5, 1, 2, 4, etc. up to around 64. *Tunable*\*
- `min_branch_length` (integer): The minimum branch length (in microns) to consider. The default is 10. *Tunable*\*
- `max_branch_length` (integer): Optional. This is the maximum branch length (in microns) to consider. By default, this parameter is not included in the configuration file. If it is not in the configuration, no maximum branch length will be enforced. *Tunable*\*
- `remove_isolated_branches` (boolean): Whether to remove branches that are not connected to any other branches *after* the network is trimmed per the branch length constraints (enforcing minimum and maximum branch lengths might isolate some branches, which may or may not be desired). The default is "false". To tune this parameter, you can simply leave it at "false", run the analysis, and then inspect the Morse tree visualization to see whether or not it should be set to "true" instead. *Tunable*\*
- `graph_smoothing_window` (float): This is the window size (in microns) for smoothing the branch paths. The default is 10. *Tunable*\*

\*Trying out a few different values for the tunable parameters tends to yield more accurate quantification of vessel formation. An efficient way to do this is to specify a list of values directly in the configuration file, for example:
```json
{
    "image_width_microns": 1000.0,
    "vessel_probability_thresh": [0.02, 0.1, 0.5],
    "graph_thresh_1": [0.5, 2, 10],
    "graph_thresh_2": [0, 4, 16],
    "graph_smoothing_window": 12,
    "min_branch_length": [10, 25],
    "remove_isolated_branches": false
}
```

The example configuration above runs the analysis for all 54 parameter combinations.
