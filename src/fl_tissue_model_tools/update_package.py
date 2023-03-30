import subprocess

PKG_NAME = 'fl_tissue_model_tools'
GITHUB_LINK = 'git+https://github.com/fogg-lab/tissue-model-analysis-tools.git'
SUBDIR = 'src'

def update_package(src_dir=None, branch_name=None):
    """Update the package from GitHub."""
    subprocess.run(f'pip uninstall -y {PKG_NAME}', shell=True, check=True)
    if src_dir is not None:
        cmd = f'pip install -I -e {src_dir}'
    elif branch_name is not None:
        cmd = f'pip install -I {PKG_NAME}@{GITHUB_LINK}@{branch_name}#subdirectory={SUBDIR}'
    else:
        cmd = f'pip install -I {PKG_NAME}@{GITHUB_LINK}#subdirectory={SUBDIR}'
    print(f'Running command: {cmd}')
    subprocess.run(cmd, shell=True, check=True)
