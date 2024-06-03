import requests
from zipfile import ZipFile
import os.path as osp
from pathlib import Path
from typing import Union

REPO_ORG = "fogg-lab"
REPO_NAME = "tissue-model-analysis-tools-data"

def download_example_images(dataset_name: str, save_directory: str='.'):
    """ Download example images from the sample data repo:
        https://github.com/fogg-lab/tissue-model-analysis-tools-data
        Args:
            dataset_name (str): Name of the dataset to download, one of:
                - "branching": For the microvessel analysis script.
                - "cell_coverage_area": For the cell area script.
                - "invasion_depth": For the invasion depth script.
                - "microvessel_segmentation": For binary segmentation training.
                - "zprojection": For the Z-projection script.
    """
    dsets = ["branching", "cell_coverage_area", "invasion_depth",
             "microvessel_segmentation", "zprojection"]
    if dataset_name not in dsets:
        raise ValueError(f"Dataset name must be one of: {dsets}")

    if dataset_name == "microvessel_segmentation":
        dataset_name += "_training"
    else:
        dataset_name += "_input"

    manifest_url = (
        f"https://raw.githubusercontent.com/{REPO_ORG}/"
        f"{REPO_NAME}/main/image_paths.json"
    )
    manifest = requests.get(manifest_url).json()

    # Download the dataset
    dataset_manifest = manifest[dataset_name]

    # Create the save directory if it doesn't exist
    save_directory = Path(save_directory) / dataset_name
    save_directory.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset_manifest, dict):
        for subdir, dset_paths in dataset_manifest.items():
            dset_paths = [osp.join(dataset_name, subdir, p) for p in dset_paths]
            download_images(dset_paths, save_directory / subdir)
    else:
        download_images(dataset_manifest, save_directory)

def download_images(repo_data_paths: list, save_directory: Union[str, Path]):
    """ Download images from a list of relative paths in the tmat data repo.
        Args:
            repo_data_paths (list): List of data paths from the repo to download.
            save_directory (str or pathlib.Path): Directory to save the images to.
    """
    # Create the save directory if it doesn't exist
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    for image_path in repo_data_paths:
        url = data_repo_path_to_url(image_path)
        filename = Path(url).name.replace("%20", " ")
        save_path = osp.join(save_directory, filename)
        if not osp.exists(save_path):
            print(f"Downloading {url}")
            r = requests.get(url)
            with open(save_path, "wb") as f:
                f.write(r.content)

def data_repo_path_to_url(rel_path: str):
    """ Convert a path to a file in the data repo to a URL.
        Args:
            rel_path (str): Path to the file relative to the root of the data repo.
        Returns:
            str: URL to the file.
    """
    rel_path = rel_path.replace("\\", "/")
    rel_path = rel_path.replace(" ", "%20")
    return (
        f"https://raw.githubusercontent.com/{REPO_ORG}/"
        f"{REPO_NAME}/main/" + rel_path
    )
