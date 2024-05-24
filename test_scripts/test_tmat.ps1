Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "cd ~"
Write-Host "----------------------------------------"
Set-Location $HOME

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "mkdir -p repos"
Write-Host "----------------------------------------"
New-Item -ItemType Directory -Force -Path repos
Set-Location repos

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "git clone --recurse-submodules https://github.com/fogg-lab/tissue-model-analysis-tools.git || true"
Write-Host "----------------------------------------"
git clone --recurse-submodules https://github.com/fogg-lab/tissue-model-analysis-tools.git

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "cd tissue-model-analysis-tools"
Write-Host "----------------------------------------"
Set-Location tissue-model-analysis-tools

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "git pull"
Write-Host "----------------------------------------"
git pull

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "conda env update -f environment.yml"
Write-Host "----------------------------------------"
conda env update -f environment.yml

$base_env_path = (conda info | Select-String -Pattern 'base environment' | ForEach-Object { $_.Line.Split(': ')[1] }).Trim()
& "$base_env_path\Scripts\conda.bat" "activate" "base"

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "conda activate tissue-model-analysis"
Write-Host "----------------------------------------"
conda activate tissue-model-analysis

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "pip install -e src"
Write-Host "----------------------------------------"
pip install -e src

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "tmat configure -f $HOME\fl_tissue_model_tools"
Write-Host "----------------------------------------"
tmat configure -f "$HOME\fl_tissue_model_tools"

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "cd .."
Write-Host "----------------------------------------"
Set-Location ..

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "git clone tissue-model-analysis-tools-data || true"
Write-Host "----------------------------------------"
git clone tissue-model-analysis-tools-data

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "cd tissue-model-analysis-tools-data"
Write-Host "----------------------------------------"
Set-Location tissue-model-analysis-tools-data

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "git pull"
Write-Host "----------------------------------------"
git pull

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "mkdir -p branching_in2"
Write-Host "----------------------------------------"
New-Item -ItemType Directory -Force -Path branching_in2

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "cp ./branching_input/A2_001.tif ./branching_in2"
Write-Host "----------------------------------------"
Copy-Item -Path .\branching_input\A2_001.tif -Destination .\branching_in2

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "cp ./branching_input/B1_001-1.tif ./branching_in2"
Write-Host "----------------------------------------"
Copy-Item -Path .\branching_input\B1_001-1.tif -Destination .\branching_in2

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "tmat compute_branches ./branching_in2 ./branching_out2"
Write-Host "----------------------------------------"
tmat compute_branches .\branching_in2 .\branching_out2

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "tmat compute_cell_area ./cell_coverage_area_input ./cell_coverage_area_out"
Write-Host "----------------------------------------"
tmat compute_cell_area .\cell_coverage_area_input .\cell_coverage_area_out

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "tmat compute_inv_depth ./invasion_depth_input ./invasion_depth_out"
Write-Host "----------------------------------------"
tmat compute_inv_depth .\invasion_depth_input .\invasion_depth_out

Write-Host "----------------------------------------"
Write-Host "Command:"
Write-Host "tmat compute_zproj ./zprojection_input ./zprojection_out"
Write-Host "----------------------------------------"
tmat compute_zproj .\zprojection_input .\zprojection_out
