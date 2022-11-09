# fogg-lab-tissue-model-analysis-tools

## Table of Contents
**[Environment](#environment-setup)**<br>
**[Capabilities](#capabilities)**<br>
**[Usage](#usage)**<br>

## Environment Setup

### Use with `conda`
Build a `conda` environment using the appropriate `environment_[OS].yml` file (located in the root project directory). For information on how to manage `conda` environments, see [environment management reference](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

#### Install `fl_tissue_model_tools` Package

1. To install the `fl_tissue_model_tools` package, ensure that your `conda` environment is activated.
2. Navigate into the `src` directory
3. Execute

```
pip install -e .
```
4. The `fl_tissue_model_tools` package should now be accessible from any code within the python environment in which int was installed

#### Uninstall `fl_tissue_model_tools` package
1. Execute

```
pip uninstall fl_tissue_model_tools
```

#### Jupyter + `fl_tissue_model_tools`
If changes are made to the `fl_tissue_model_tools` code, the notebook kernel must be restarted for those changes to be reflected.

### Other Dependencies

### Data, Analysis, and Figures Directories
Ensure that a file called dev_paths.txt is located in the root project directory. The file should be laid out as follows (do not include comments):

```
root/path/for/storing/data      # top-level directory of data to be analyzed
root/path/for/storing/analysis  # top-level directory of analysis output
root/path/for/storing/figures   # top-level directory of figure and visualization output
# blank line
```

See `dev_paths_example.txt` for an example file.

### Topological Analysis
To utilize the `fl_tissue_model_tools.topology` module, the `graph_recon_DM` dependency must be set up. To do so, follow the installation instructions in `fl_tissue_model_tools/graph_recon_DM/README.md`.

#### For Windows users
It is easiest to install `g++` via `MYSYS2`, following this [guide](https://www.msys2.org/). Verify installation of `g++` by restarting terminal after installation and executing
```
g++ --version
```

## Capabilities

For a detailed description of analysis capabilities, see the [capabilities overview notebook](notebooks/capabilities_overview.ipynb).

## Usage

### Tools
There commandline tools which handle the following common operations:

* Cell area computation (of Z projected images)
* Z projection of image Z stacks
* Invasion depth computation (of Z stacks)

To use these tools:

1. Ensure `conda` environment is active & all setup procedures have been followed (see [Install `fl_tissue_model_tools` Package](#install-fltissuemodeltools-package) )
2. Within a terminal window, change into the `scripts` directory
3. Execute the commandline tools via (see sections below for details)
```
python [command_script].py [-flags] [arguments]
```
It is strongly recommeded that the `-v` (verbose) flag be used for each command.

For input data paths, it is usually easiest to copy the path from the file explorer search bar.

For a description of all parameters that each commandline tool accepts, execute the command using
```
python [command_script].py -h
```
or
```
python [command_script].py --help
```

#### Cell Area
**Basic usage:**
```
python compute_cell_area.py -v [input_root_path] [output_root_path]
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
```
python compute_zproj.py -v [input_root_path] [output_root_path]
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
```
python compute_zproj.py -v --area [input_root_path] [output_root_path]
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
python compute_inv_depth.py -v [input_root_path] [output_root_path]
```

For a description of `input_root_path` directory structure, see [Z Projection](#z-projection).
