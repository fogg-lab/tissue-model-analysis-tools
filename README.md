# Fogg Lab Tissue Model Analysis Tools

A Python package for the high-throughput analysis of cancer and endothelial cell dynamics in hydrogels.

**Try the standalone interactive demo notebook in Google Colab**

<a target="_blank" href="https://colab.research.google.com/github/fogg-lab/tissue-model-analysis-tools/blob/main/notebooks/analysis_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Table of Contents
**[Setup](#setup)**<br>
**[Capabilities](#capabilities)**<br>
**[Usage](#usage)**<br>

## Setup

*Create the conda environment and install the fl_tissue_model_tools package*

### Prerequisites:
- [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) package manager
   - Install a Conda distribution such as **Miniconda** and choose the right installer for your operating system.
   - Specifically for devices with ARM processors (e.g. a MacBook with an M series chip), install **Miniforge** and choose the **arm64** architecture.
- Build tools, according to your operating system:
   - Linux or Windows Subsystem for Linux (WSL): g++ (likely already installed)
   - Windows (if using WSL, see above bullet instead): [Build tools for visual studio](https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2022). At the installer, select "Desktop development with C++" with the following individual components (in the details pane) selected:
     - MSVC C++ x64/86 build tools
     - Windows SDK
   - Mac: Clang (run `xcode-select --install` in the terminal)

### Quick setup

Run the following commands in a terminal/command prompt window.

```bash
conda env update -f https://raw.githubusercontent.com/fogg-lab/tissue-model-analysis-tools/main/environment.yml
conda activate tissue-model-analysis
pip install -I fl_tissue_model_tools@git+https://github.com/fogg-lab/tissue-model-analysis-tools.git#subdirectory=src
tmat configure
```

### Detailed setup

#### Optional: Clone repository (for development)
```
git clone --recurse-submodules https://github.com/fogg-lab/tissue-model-analysis-tools.git
```

#### Create conda environment
Build a conda environment using the `environment.yml` file.

For more info on Conda environments, see [environment management reference](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

If you cloned the repo, `cd` to the project directory and run:
```bash
conda env update -f environment.yml
```

If you didn't clone the repo, run:
```bash
conda env update -f https://raw.githubusercontent.com/fogg-lab/tissue-model-analysis-tools/main/environment.yml
```

Next, activate the environment:
```
conda activate tissue-model-analysis
```

---

#### Install `fl_tissue_model_tools` Package

1. To install the `fl_tissue_model_tools` package, ensure that your conda environment is activated.

2. Install with pip:

If you cloned the repo, `cd` into the `src` directory and run:
```bash
pip install -e .
```
If you didn't clone the repo, run this command:
```bash
pip install -I fl_tissue_model_tools@git+https://github.com/fogg-lab/tissue-model-analysis-tools.git#subdirectory=src
```

3. The `fl_tissue_model_tools` package should now be accessible from any code within the python environment in which it was installed. Additionally, the `tissue-model-analysis-tools` (or `tmat` for short) command has been added to your terminal as an entrypoint for running the package scripts. Commands will follow this layout (more details in [usage](#usage)):
```bash
tissue-model-analysis-tools [SUBCOMMAND] [OPTIONS]
```

4. Configure base directory to store data, scripts, and script configuration files:

```bash
# example
tmat configure "C:\Users\Quinn\Desktop\some_folder_name"
```

```bash
# or use the interactive prompt
tmat
```

---

#### Uninstall `fl_tissue_model_tools` package
Execute
```bash
pip uninstall fl_tissue_model_tools
```

---

#### Update `fl_tissue_model_tools` package
To update `tmat`, just run the setup commands again (with one caveat: if you cloned the repo, you should `cd` into it and run `git pull` instead of cloning again). When you run `tmat configure`, make sure to select `y` at every prompt to overwrite the old scripts.

#### Jupyter + `fl_tissue_model_tools`
If changes are made to the `fl_tissue_model_tools` code, the notebook kernel must be restarted for those changes to be reflected.

## Capabilities

For a detailed description of analysis capabilities, see the [capabilities overview notebook](notebooks/capabilities_overview.ipynb).

## Usage

### Tools
Four command-line tools which handle the following operations:
* Cell area computation (of Z projected images)
* Z projection of image Z stacks
* Invasion depth computation (of Z stacks)
* Quantify blood vessel formation (number of branches, lengths of branches)

To use these tools:

1. Ensure conda environment is active & all setup procedures have been followed (see [Install `fl_tissue_model_tools` Package](#install-fl_tissue_model_tools-package) )
2. Within a terminal window, execute the commandline tools via (see sections below for details)
```bash
# non-interactive
tmat [command_script] [-flags] [arguments]
```
```bash
# interactive (all scripts and options are available via this command)
tmat
```

For input data paths, it is usually easiest to copy the path from the file explorer search bar.

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
**Basic usage:**
```bash
tmat compute_cell_area "path/to/input/folder" "path/to/output/folder"
```

Here, `path/to/input/folder` is the full path to a directory of images which will be analyzed.

If your images are not cropped to the region inside the well, you can have the script automatically detect the well region by adding the `--detect-well` flag (or `-w` for short). For instance, if your wells are circular and you add the `--detect-well` flag, the script will detect and mask out the region outside of this circular well. Also works for "squircle" shaped (i.e. square with rounded corners) wells. Example usage:
```bash
tmat compute_cell_area --detect-well "path/to/input/folder" "path/to/output/folder"
```

**Advanced usage:**

* Create custom configuration `.json` file, using `config/default_cell_area_computation.json` as a template. The following parameters can be customized:
    * `dsamp_size` (int): Size that input images will be downsampled to for analysis. Smaller sizes mean faster, less accurate analysis. Default is 512, meaning the image will be downscaled so that the 
    * `sd_coef` (float): Strictness of thresholding. Positive numbers are more strict, negative numbers are less strict. This is a multiplier of the foreground pixel standard deviation, so values in the range (-2, 2) are the most reasonable.
    * `rs_seed` (integer): A random seed for the algorithm. Allows for reproducability since the Gaussian curves are randomly initialized. Default is 0.
    * `batch_size` (integer): Number of images to process at once. Larger numbers are faster but require more memory. Default is 4.

**Run with custom configuration file:**
```bash
tmat compute_cell_area --config "path/to/config/file.json" "path/to/input/folder" "path/to/output/folder"
```

#### Z Projection
**Basic usage:**
```bash
tmat compute_zproj "path/to/input/directory" "path/to/output/folder"
```
Here, "path/to/input/directory" is the full path to a directory of Z stacks. Each Z stack should either be an OME-TIFF format file, or subdirectory of individual TIFF format files. 

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
tmat compute_zproj --area "path/to/input/folder" "path/to/output/folder"
```

**Advanced usage:**

* Select custom Z projection method from:
    * Minimum
    * Maximum
    * Median
    * Average
    * Focus Stacking

See [Capabilities](#capabilities) for details.

#### Invasion Depth
**Basic usage:**
```
tmat compute_inv_depth "path/to/input/folder" "path/to/output/folder"
```

For a description of the input directory structure, see [Z Projection](#z-projection).

#### Branches (quantify vessel formation)
**Basic usage:**
```bash
tmat compute_branches "path/to/input/folder" "path/to/output/folder"
```

Here, `path/to/input/folder` is the full path to a directory of images which will be analyzed.

If your images are not cropped to the region inside the well, you can have the script automatically detect the well region by adding the `--detect-well` flag (or `-w` for short). For instance, if your wells are circular and you add the `--detect-well` flag, the script will detect and mask out the region outside of this circular well. Also works for "squircle" shaped (i.e. square with rounded corners) wells. Example usage:
```bash
tmat compute_branches --detect-well "path/to/input/folder" "path/to/output/folder"
```

**Advanced usage:**

Customize configuration variables (you can edit `config/default_branching_computation.json` in your base directory, or refer to `src/fl_tissue_model_tools/config` in this repository):

- `well_width_microns` (float): Physical width of the image in microns.
- `use_latest_model_cfg` (boolean): If `True`, the most recently trained model in the `model_training` folder will be used. If `False`, the model specified in `model_cfg_path` will be used.
- `model_cfg_path` (string): Path to the configuration file of the segmentation model. You can leave this blank (`''`) to use the most recently trained model in the `model_training` folder.
- `graph_thresh_1` (float): May require some experimentation to find the best value for your data. This threshold controls how much of the morse graph is used to compute the number of branches. Lower values include more of the graph, and more branches are detected. Higher values include less of the graph, and fewer branches are detected. Try different values like 0.25, 0.5, 1, 2, 4, etc. up to around 64.
- `graph_thresh_2` (float): Also could use some tuning. This is the threshold for connecting branches, e.g. where it is ambiguous whether two branches are part of the same component. Lower values result in more connected branches, and higher values result in more disconnections. Try values like 0.0, 0.25, 0.5, 1, 2, 4, etc. up to around 64.
- `graph_smoothing_window` (integer): This is the window size (in pixels) for smoothing the branch paths. Values in the range of 5-20 tend to work well.
- `min_branch_length` (integer): This is the minimum branch length (in pixels) to consider.
