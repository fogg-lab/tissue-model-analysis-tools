import os
import shutil
from typing import Sequence

def make_dir(path: str) -> None:
    # Path(path).mkdir(parents=True, exist_ok=True)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def save_class_imgs(img_paths: Sequence[str], split_list: Sequence[int], split_map: dict[int, str], img_class: str, dset_path: str) -> None:
    for i, img_p, in enumerate(img_paths):
        img_n = img_p.split("/")[-1]
        shutil.copy(img_p, f"{dset_path}/{split_map[split_list[i]]}/{img_class}/{img_n}")
