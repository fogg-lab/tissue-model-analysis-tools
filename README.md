# fogg-lab-tissue-model-analysis-tools

## Table of Contents
**[Setup](#setup)**<br>
**[Capabilities](#capabilities)**<br>
**[Usage](#usage)**<br>

## Setup

*Create the conda environment and install the fl_tissue_model_tools package*

### Prerequisite: [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

**Tip**: Enable [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) to create the conda environment faster.

### Quick setup

Run these commands on the command line:
```bash
conda env create -f https://raw.githubusercontent.com/fogg-lab/tissue-model-analysis-tools/main/environment.yml
conda activate tissue-model-analysis
pip install -I fl_tissue_model_tools@git+https://github.com/fogg-lab/tissue-model-analysis-tools.git#subdirectory=src
tmat configure
```

### Detailed setup

#### Optional: Clone repository
```
git clone --recurse-submodules git@github.com:fogg-lab/tissue-model-analysis-tools.git
```

#### Create conda environment
Build a conda environment using the `environment.yml` file. If you have a CUDA-capable (NVIDIA) GPU, use `environment_gpu.yml` for GPU-accelerated training and inference.

For more information on how to manage conda environments, see [environment management reference](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

If you cloned the repo, `cd` to the project directory and run:
```bash
conda env create -f environment.yml
```

If you didn't clone the repo, run:
```bash
conda env create -f https://raw.githubusercontent.com/fogg-lab/tissue-model-analysis-tools/main/environment.yml
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
1. Execute

```bash
pip uninstall fl_tissue_model_tools
```

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

It is recommended that the `-v` (verbose) flag be used for each command.

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
tmat compute_cell_area -v [input_root_path] [output_root_path]
```
Here, `input_path` is the full path to a directory of images which will be analyzed.

**Advanced usage:**

* Create custom configuration `.json` file, using `config/default_cell_area_computation.json` as a template.
    * `dsamp_size`: Size that input images will be downsampled to for analysis. Smaller sizes mean faster, less accurate analysis.
    * `sd_coef`: Strictness of thresholding. Positive numbers are more strict, negative numbers are less strict. This is a multiplier of the foreground pixel standard deviation, so values in the range (-2, 2) are the most reasonable.
    * `pinhole_buffer`: Used to compute radius of circular mask. For a 250 pixel downsampled image, `pinhole_buffer` = 0.04 means the circular mask radius will extend from the center of the image to 0.04 * 250 = 10 pixels from the top of the image. In other words, the diameter of the circular mask will be 250 - (10*2) = 230 pixels.
    * `rs_seed`: A random seed for the algorithm. Allows for reproducability since the Gaussian curves are randomly initialized.

#### Z Projection
**Basic usage:**
```bash
tmat compute_zproj -v [input_root_path] [output_root_path]
```
Here, `input_root_path` is the full path to a directory of Z stack subdirectories which will be analyzed. Each Z stack subdirectory should contain all images for a given Z stack with files containing the pattern `...Z[pos]_...` in their name. For example `...Z01_...` denotes Z position 1 for a given image.

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
tmat compute_zproj -v --area [input_root_path] [output_root_path]
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
tmat compute_inv_depth -v [input_root_path] [output_root_path]
```

For a description of `input_root_path` directory structure, see [Z Projection](#z-projection).

#### Branches (quantify vessel formation)
**Basic usage:**
```bash
# the -i flag saves intermediate results in the visualizations folder of the output directory
tmat compute_branches -v -i [input_root_path] [output_root_path]
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
