from fl_tissue_model_tools.colored_messages import SFM

FILES_AND_DIRS_ERR_MSG = (
    SFM.failure,
    "Found both files and directories in input directory: {input_dir}\n"
    "To fix, you can check your input directory and make sure it conforms "
    " to the expected input directory structure. Documentation can be found here:\n"
    "https://github.com/fogg-lab/tissue-model-analysis-tools/tree/main?tab=readme-ov-file#image-input-directory-structure",
)


class ZStackInputException(Exception):
    """Exception to raise when encountering issues with zstack input directories."""
