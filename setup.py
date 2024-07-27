import os
from pathlib import Path
import shutil
import configparser
from setuptools import find_namespace_packages, setup

setup_cfg = configparser.ConfigParser()
setup_cfg.read("setup.cfg")

pkg_name = setup_cfg["metadata"]["name"]
pkg_root = Path(pkg_name).resolve()
cfg_file = pkg_root / "package.cfg"
project_root = pkg_root.parent

# copy scripts, model_training and config directories to package directory
for dir_name in ["scripts", "model_training", "config"]:
    src_dir = project_root / dir_name
    dest_dir = Path(pkg_name) / dir_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)

config = configparser.ConfigParser()

# Default base dir is relative to user's home directory, expanded at runtime
# However on first run, the configuration script will ask for a base dir
config["metadata"] = {"name": pkg_name}

config[pkg_name] = {"base_dir": str(Path("~") / pkg_name)}

with open(cfg_file, "w", encoding="utf-8") as config_file:
    config.write(config_file)


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = [str(cfg_file)]
for dir_name in ["scripts", "model_training", "config"]:
    extra_files.extend(package_files(pkg_root / dir_name))

exclude_packages = ["pydmtgraph.notebooks", "pydmtgraph.figures", "pydmtgraph.data"]

setup(
    name=pkg_name,
    version="1.0.6",
    description="Automatic image processing software for 3D cancer models",
    author="Fogg Lab",
    packages=find_namespace_packages(exclude=exclude_packages),
    package_data={pkg_name: extra_files},
    include_package_data=True,
    url="https://github.com/fogg-lab/tissue-model-analysis-tools",
    install_requires=[
        "aicsimageio[nd2]==4.14.0",
        "dask==2024.5.2",
        "imagecodecs==2024.1.1",
        "matplotlib==3.8.4",
        "networkx==3.3",
        "numba==0.59.1",
        "numpy==1.26.4",
        "opencv-python>=4.9.0,<4.10",
        "pandas==2.2.2",
        "pillow==10.3.0",
        "pip==24.0",
        "scikit-image==0.22.0",
        "scikit-learn==1.5.0",
        "scipy==1.13.1",
        "silence-tensorflow==1.2.2",
        "tqdm==4.66.4",
        "keras-tuner==1.4.7",
        # Using patched version of Gooey for proper dark mode support, especially on Linux and macOS
        "gooey@git+https://github.com/wigginno/Gooey.git",
        "pyinstaller==6.9.0",
    ]
    + ["tensorflow-cpu==2.14.1" if os.name == "nt" else "tensorflow==2.14.1"],
    entry_points={"console_scripts": [f"tmat={pkg_name}.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="License Notice for Free Academic Research Use",
    python_requires=">=3.9, <3.12",
)
