# fogg-lab-tissue-model-analysis-tools

## Table of Contents
**[Environment](#environment-setup)**<br>
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

For a detailed description of analysis capabilities, see the [capabilities overview notebook](notebooks/capabilities_overview.ipynb)

## Usage

### Tools
There commandline tools which handle the following common operations:

* Cell area computation (of Z projected images)
* Z projection of image Z stacks
* Invasion depth computation (of Z stacks)

#### Cell Area
1. Change into the `scripts` directory
```
python compute_cell_area.py -v ...
```