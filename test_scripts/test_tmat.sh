echo "----------------------------------------"
echo "Command:"
echo "cd ~"
echo "----------------------------------------"
cd ~
echo "----------------------------------------"
echo "Command:"
echo "mkdir -p repos"
echo "----------------------------------------"
mkdir -p repos
echo "----------------------------------------"
echo "Command:"
echo "cd repos"
echo "----------------------------------------"
cd repos
echo "----------------------------------------"
echo "Command:"
echo "git clone --recurse-submodules https://github.com/fogg-lab/tissue-model-analysis-tools.git || true"
echo "----------------------------------------"
git clone --recurse-submodules https://github.com/fogg-lab/tissue-model-analysis-tools.git || true
echo "----------------------------------------"
echo "Command:"
echo "cd tissue-model-analysis-tools"
echo "----------------------------------------"
cd tissue-model-analysis-tools
echo "----------------------------------------"
echo "Command:"
echo "git pull"
echo "----------------------------------------"
git pull
echo "----------------------------------------"
echo "Command:"
echo "conda env update -f environment.yml"
echo "----------------------------------------"
conda env update -f environment.yml
base_env_path=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $base_env_path/etc/profile.d/conda.sh
echo "----------------------------------------"
echo "Command:"
echo "conda activate tissue-model-analysis"
echo "----------------------------------------"
conda activate tissue-model-analysis
echo "----------------------------------------"
echo "Command:"
echo "pip install -e src"
echo "----------------------------------------"
pip install -e src
echo "----------------------------------------"
echo "Command:"
echo "tmat configure -f ~/fl_tissue_model_tools"
echo "----------------------------------------"
tmat configure -f ~/fl_tissue_model_tools
echo "----------------------------------------"
echo "Command:"
echo "cd .."
echo "----------------------------------------"
cd ..
echo "----------------------------------------"
echo "Command:"
echo "git clone tissue-model-analysis-tools-data || true"
echo "----------------------------------------"
git clone tissue-model-analysis-tools-data || true
echo "----------------------------------------"
echo "Command:"
echo "cd tissue-model-analysis-tools-data"
echo "----------------------------------------"
cd tissue-model-analysis-tools-data
echo "----------------------------------------"
echo "Command:"
echo "git pull"
echo "----------------------------------------"
git pull
echo "----------------------------------------"
echo "Command:"
echo "mkdir -p branching_in2"
echo "----------------------------------------"
mkdir -p branching_in2
echo "----------------------------------------"
echo "Command:"
echo "cp ./branching_input/A2_001.tif ./branching_in2"
echo "----------------------------------------"
cp ./branching_input/A2_001.tif ./branching_in2
echo "----------------------------------------"
echo "Command:"
echo "cp ./branching_input/B1_001-1.tif ./branching_in2"
echo "----------------------------------------"
cp ./branching_input/B1_001-1.tif ./branching_in2
echo "----------------------------------------"
echo "Command:"
echo "tmat compute_branches ./branching_in2 ./branching_out2"
echo "----------------------------------------"
tmat compute_branches ./branching_in2 ./branching_out2
echo "----------------------------------------"
echo "Command:"
echo "tmat compute_cell_area ./cell_coverage_area_input ./cell_coverage_area_out"
echo "----------------------------------------"
tmat compute_cell_area ./cell_coverage_area_input ./cell_coverage_area_out
echo "----------------------------------------"
echo "Command:"
echo "tmat compute_inv_depth ./invasion_depth_input ./invasion_depth_out"
echo "----------------------------------------"
tmat compute_inv_depth ./invasion_depth_input ./invasion_depth_out
echo "----------------------------------------"
echo "Command:"
echo "tmat compute_zproj ./zprojection_input ./zprojection_out"
echo "----------------------------------------"
tmat compute_zproj ./zprojection_input ./zprojection_out
