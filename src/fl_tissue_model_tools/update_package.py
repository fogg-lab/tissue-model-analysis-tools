import subprocess

# TODO: Before pulling into main, remove "@packaging" from GITHUB_LINK

PKG_NAME = 'fl_tissue_model_tools'
GITHUB_LINK = 'git+https://github.com/fogg-lab/tissue-model-analysis-tools.git@packaging'
SUBDIR = 'src'

def update_package():
    """Update the package from GitHub."""
    cmd = f'pip install -I {PKG_NAME}@{GITHUB_LINK}#subdirectory={SUBDIR}'
    print(f'Running command: {cmd}')
    subprocess.run(cmd, shell=True, check=True)
